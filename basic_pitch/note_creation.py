#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import pathlib
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
import mir_eval
import librosa
import resampy
import numpy as np
import numpy.typing as npt
import pretty_midi
import scipy
from scipy.io import wavfile

from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    ANNOTATIONS_N_SEMITONES,
    ANNOTATIONS_BASE_FREQUENCY,
    AUDIO_N_SAMPLES,
    ANNOT_N_FRAMES,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    N_FREQ_BINS_CONTOURS,
)

MIDI_OFFSET = 21
SONIFY_FS = 3000
N_PITCH_BEND_TICKS = 8192
MAX_FREQ_IDX = 87
MAX_MIDI_PITCH_BEND_INSTRUMENTS = 15  # 16 MIDI channels, minus the percussion channel

DEFAULT_MIN_NOTE_LEN = 11
ENERGY_TOLERANCE = 11
MAGIC_ALIGNMENT_OFFSET = 0.0018
MIDI_VELOCITY_SCALE = 127
MIDI_MIN_PITCH = 0
MIDI_MAX_PITCH = 127
MIDI_MIN_NOTE_VELOCITY = 1  # a note-on with velocity 0 is a note-off, so the note would vanish from the file
PITCH_BEND_SCALE = 4096
MICROSECONDS_PER_MINUTE = 60_000_000
MIDI_MAX_TEMPO_MICROSECONDS = 0xFFFFFF  # a MIDI tempo event stores the microseconds per beat in 3 bytes
MIDI_MAX_TICK = 10_000_000  # pretty_midi treats a file with a later event as corrupt


def _is_number(value: object) -> bool:
    """True for ints and floats (also numpy ones), but not for bools, strings or None."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def verify_note_options(
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: float,
    min_freq: Optional[float],
    max_freq: Optional[float],
    midi_tempo: float,
    melodia_peak_thresh: float = 0.5,
    note_peak_thresh: float = 0.4,
) -> None:
    """Raise a ValueError naming the first note option that is outside its allowed range."""
    for name, value in (
        ("onset threshold", onset_thresh),
        ("frame threshold", frame_thresh),
        ("melodia peak threshold", melodia_peak_thresh),
        ("note peak threshold", note_peak_thresh),
    ):
        if not _is_number(value) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"The {name} must be a number between 0 and 1, got {value}.")
    if not _is_number(min_note_len) or not math.isfinite(min_note_len) or min_note_len < 0:
        raise ValueError(f"The minimum note length must be a number of at least 0, got {min_note_len}.")
    for name, frequency in (("minimum frequency", min_freq), ("maximum frequency", max_freq)):
        if frequency is not None and (not _is_number(frequency) or not math.isfinite(frequency) or frequency <= 0):
            raise ValueError(f"The {name} must be a number above 0 Hz, got {frequency}.")
    if min_freq is not None and max_freq is not None and min_freq > max_freq:
        raise ValueError(f"The minimum frequency ({min_freq}) must not be above the maximum frequency ({max_freq}).")
    if not _is_number(midi_tempo) or not math.isfinite(midi_tempo) or midi_tempo <= 0:
        raise ValueError(f"The MIDI tempo must be a number above 0, got {midi_tempo}.")
    if not 1 <= round(MICROSECONDS_PER_MINUTE / midi_tempo) <= MIDI_MAX_TEMPO_MICROSECONDS:
        raise ValueError(
            "The MIDI tempo must be a number of beats per minute that a MIDI file can store, "
            f"between {MICROSECONDS_PER_MINUTE / MIDI_MAX_TEMPO_MICROSECONDS:.2f} and {MICROSECONDS_PER_MINUTE}, "
            f"got {midi_tempo}."
        )


def amplitude_to_velocity(amplitude: float) -> int:
    """Convert a note amplitude to a MIDI velocity, limited to the audible range 1-127 (NaN becomes 1)."""
    if np.isnan(amplitude):
        return MIDI_MIN_NOTE_VELOCITY
    return int(np.clip(np.round(MIDI_VELOCITY_SCALE * amplitude), MIDI_MIN_NOTE_VELOCITY, MIDI_VELOCITY_SCALE))


def verify_sonification_samplerate(samplerate: int) -> None:
    """Raise a ValueError if the sonification sample rate cannot be written to a wav file."""
    if isinstance(samplerate, bool) or not isinstance(samplerate, (int, np.integer)) or not 0 < samplerate < 2**32:
        raise ValueError(f"The sonification sample rate must be a whole number of Hz above 0, got {samplerate}.")


def verify_outputs_requested(save_midi: bool, sonify_midi: bool, save_model_outputs: bool, save_notes: bool) -> None:
    """Raise a ValueError if no output file would be written."""
    if not (save_midi or sonify_midi or save_model_outputs or save_notes):
        raise ValueError(
            "No output was requested, so nothing would be saved. Do not turn off the MIDI file "
            "(--no-save-midi) unless another output is requested: --sonify-midi, --save-model-outputs "
            "or --save-note-events."
        )


def model_output_to_notes(
    output: Dict[str, npt.NDArray[Any]],
    onset_thresh: float,
    frame_thresh: float,
    infer_onsets: bool = True,
    min_note_len: int = 11,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    include_pitch_bends: bool = True,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    midi_tempo: float = 120,
    melodia_peak_thresh: float = 0.5,
    note_peak_thresh: float = 0.4,
    suppress_harmonics: bool = True,
) -> Tuple[pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]]]:
    """Convert model output to MIDI

    Args:
        output: A dictionary with shape
            {
                'frame': array of shape (n_times, n_freqs),
                'onset': array of shape (n_times, n_freqs),
                'contour': array of shape (n_times, 3*n_freqs)
            }
            representing the output of the basic pitch model.
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        min_note_len: The minimum allowed note length in frames.
        min_freq: Minimum allowed output frequency, in Hz. If None, all frequencies are used.
        max_freq: Maximum allowed output frequency, in Hz. If None, all frequencies are used.
        include_pitch_bends: If True, include pitch bends.
        multiple_pitch_bends: If True, allow overlapping notes in midi file to have pitch bends.
        melodia_trick: Use the melodia post-processing step.
        midi_tempo: The tempo of the MIDI file.
        melodia_peak_thresh: A note found by the melodia step only (without an onset of at least onset_thresh)
            is kept only if its frame activation reaches this value.
        note_peak_thresh: A note is dropped if its frame activation never reaches this value. Use 0 to keep every note.
        suppress_harmonics: Drop weaker notes that sound 12, 19 or 24 semitones above a note starting at the same time.

    Returns:
        midi : pretty_midi.PrettyMIDI object
        note_events: A list of note event tuples (start_time_s, end_time_s, pitch_midi, amplitude)
    """
    verify_note_options(
        onset_thresh,
        frame_thresh,
        min_note_len,
        min_freq,
        max_freq,
        midi_tempo,
        melodia_peak_thresh,
        note_peak_thresh,
    )

    frames = output["note"]
    onsets = output["onset"]
    contours = output["contour"]

    estimated_notes = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        infer_onsets=infer_onsets,
        min_note_len=min_note_len,
        min_freq=min_freq,
        max_freq=max_freq,
        melodia_trick=melodia_trick,
        melodia_peak_thresh=melodia_peak_thresh,
        note_peak_thresh=note_peak_thresh,
        suppress_harmonics=suppress_harmonics,
    )
    if include_pitch_bends:
        estimated_notes_with_pitch_bend = get_pitch_bends(contours, estimated_notes)
    else:
        estimated_notes_with_pitch_bend = [(note[0], note[1], note[2], note[3], None) for note in estimated_notes]

    times_s = model_frames_to_time(contours.shape[0] + 1)
    estimated_notes_time_seconds = [
        (times_s[note[0]], times_s[note[1]], note[2], note[3], note[4]) for note in estimated_notes_with_pitch_bend
    ]

    return (
        note_events_to_midi(estimated_notes_time_seconds, multiple_pitch_bends, midi_tempo),
        estimated_notes_time_seconds,
    )


def sonify_midi(midi: pretty_midi.PrettyMIDI, save_path: Union[pathlib.Path, str], sr: int = 44100) -> None:
    """Sonify a pretty_midi midi object and save to a file.

    Args:
        midi: A pretty_midi.PrettyMIDI object that will be sonified.
        save_path: Where to save the sonified midi.
        sr: Sample rate for rendering audio from midi, a whole number of Hz above 0.

    Raises:
        ValueError: If the sample rate cannot be written to a wav file.
    """
    verify_sonification_samplerate(sr)
    y = midi.synthesize(sr)
    wavfile.write(save_path, sr, y)


def sonify_salience(
    gram: npt.NDArray[Any], semitone_resolution: int, save_path: Optional[str] = None, thresh: float = 0.2
) -> Tuple[npt.NDArray[Any], int]:
    """Sonify a salience matrix.

    Args:
        gram: A matrix of pitch salience values with range 0-1, with shape (n_freqs, n_times),
            where n_freqs is 88 times semitone_resolution.
            The frequencies are logarithmically spaced. The matrix is not modified.
        semitone_resolution: The number of bins per semitone in gram.
        save_path: Optional location to save the sonified salience.
        thresh: Salience values below thresh will not be sonified. Used to increase the speed of this function.

    Returns:
        A tuple of the sonified salience as an audio signal and the associated sample rate.
    """
    if isinstance(semitone_resolution, bool) or not isinstance(semitone_resolution, (int, np.integer)):
        raise ValueError(f"The semitone resolution must be a whole number of at least 1, got {semitone_resolution}.")
    if semitone_resolution < 1:
        raise ValueError(f"The semitone resolution must be a whole number of at least 1, got {semitone_resolution}.")
    if gram.ndim != 2 or gram.shape[0] != ANNOTATIONS_N_SEMITONES * semitone_resolution:
        raise ValueError(
            f"The gram must have {ANNOTATIONS_N_SEMITONES * semitone_resolution} rows (88 semitones times "
            f"{semitone_resolution} bins per semitone) and one column per time step, got shape {gram.shape}."
        )
    freqs = librosa.core.cqt_frequencies(
        n_bins=ANNOTATIONS_N_SEMITONES * semitone_resolution,
        fmin=ANNOTATIONS_BASE_FREQUENCY,
        bins_per_octave=12 * semitone_resolution,
    )
    # this function is slow - for speed, only sonify frequencies below
    # sonify_fs/2 Hz (e.g. 1000 Hz)
    max_freq_idx = np.where(freqs > SONIFY_FS / 2)[0][0]
    times = librosa.core.frames_to_time(
        np.arange(gram.shape[1]),
        sr=AUDIO_SAMPLE_RATE,
        hop_length=AUDIO_N_SAMPLES / ANNOT_N_FRAMES,  # type: ignore[call-overload]  # THE CORRECT HOP IS FRACTIONAL
    )
    gram = np.where(gram < thresh, 0, gram)
    if gram.shape[1] < 2:
        # mir_eval cannot infer a duration from fewer than two time stamps, so a gram this short is silent
        frame_s = (AUDIO_N_SAMPLES / ANNOT_N_FRAMES) / AUDIO_SAMPLE_RATE
        y = np.zeros(int(np.round(gram.shape[1] * frame_s * SONIFY_FS)))
    else:
        y = mir_eval.sonify.time_frequency(gram[:max_freq_idx, :], freqs[:max_freq_idx], times, fs=SONIFY_FS)
    if save_path:
        y_resamp = resampy.resample(y, SONIFY_FS, 44100) if len(y) else y
        wavfile.write(save_path, 44100, y_resamp)

    return y, SONIFY_FS


def midi_pitch_to_contour_bin(pitch_midi: int) -> float:
    """Convert midi pitch to corresponding index in contour matrix

    Args:
        pitch_midi: pitch in midi

    Returns:
        index in contour matrix

    """
    pitch_hz = librosa.midi_to_hz(pitch_midi)
    return float(12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(pitch_hz / ANNOTATIONS_BASE_FREQUENCY))


def get_pitch_bends(
    contours: npt.NDArray[Any], note_events: List[Tuple[int, int, int, float]], n_bins_tolerance: int = 25
) -> List[Tuple[int, int, int, float, Optional[List[int]]]]:
    """Given note events and contours, estimate pitch bends per note.
    Pitch bends are represented as a sequence of evenly spaced midi pitch bend control units.
    The time stamps of each pitch bend can be inferred by computing an evenly spaced grid between
    the start and end times of each note event.

    Args:
        contours: Matrix of estimated pitch contours
        note_events: note event tuple
        n_bins_tolerance: Pitch bend estimation range. Defaults to 25.

    Returns:
        note events with pitch bends
    """
    window_length = n_bins_tolerance * 2 + 1
    freq_gaussian = scipy.signal.windows.gaussian(window_length, std=5)
    note_events_with_pitch_bends = []
    for start_idx, end_idx, pitch_midi, amplitude in note_events:
        freq_idx = int(np.round(midi_pitch_to_contour_bin(pitch_midi)))
        freq_start_idx = np.max([freq_idx - n_bins_tolerance, 0])
        freq_end_idx = np.min([N_FREQ_BINS_CONTOURS, freq_idx + n_bins_tolerance + 1])

        pitch_bend_submatrix = (
            contours[start_idx:end_idx, freq_start_idx:freq_end_idx]
            * freq_gaussian[
                np.max([0, n_bins_tolerance - freq_idx]) : window_length
                - np.max([0, freq_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1)])
            ]
        )
        pb_shift = n_bins_tolerance - np.max([0, n_bins_tolerance - freq_idx])

        # a frame with no contour activity near the note has no bend, instead of the first bin of the window
        peak_bins = np.where(pitch_bend_submatrix.max(axis=1) > 0, np.argmax(pitch_bend_submatrix, axis=1), pb_shift)
        bends: Optional[List[int]] = list(peak_bins - pb_shift)  # this is in units of 1/3 semitones
        note_events_with_pitch_bends.append((start_idx, end_idx, pitch_midi, amplitude, bends))
    return note_events_with_pitch_bends


def note_events_to_midi(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]],
    multiple_pitch_bends: bool = False,
    midi_tempo: float = 120,
) -> pretty_midi.PrettyMIDI:
    """Create a pretty_midi object from note events

    Args:
        note_events : list of tuples [(start_time_seconds, end_time_seconds, pitch_midi, amplitude)]
            where amplitude is a number between 0 and 1
        multiple_pitch_bends : If True, allow overlapping notes to have pitch bends
            Note: this will assign each pitch to its own midi instrument, as midi does not yet
            support per-note pitch bends. A midi file has 15 usable channels, so with more pitches
            than that, notes that do not sound at the same time share an instrument.

    Returns:
        pretty_midi.PrettyMIDI() object

    """
    for _, _, note_number, *_ in note_events_with_pitch_bends:
        if (
            isinstance(note_number, bool)
            or not isinstance(note_number, (int, np.integer))
            or not MIDI_MIN_PITCH <= note_number <= MIDI_MAX_PITCH
        ):
            raise ValueError(
                f"The pitch of a note must be a whole number between {MIDI_MIN_PITCH} and {MIDI_MAX_PITCH}, "
                f"got {note_number}."
            )
    mid = pretty_midi.PrettyMIDI(initial_tempo=midi_tempo)
    last_tick = max((end_time for _, end_time, *_ in note_events_with_pitch_bends), default=0) * (
        midi_tempo * mid.resolution / 60
    )
    if last_tick + 1 >= MIDI_MAX_TICK:
        raise ValueError(
            f"The MIDI tempo must be lower for audio this long: at {midi_tempo} beats per minute the last note "
            f"ends after {MIDI_MAX_TICK} ticks, more than a MIDI file can hold."
        )
    if not multiple_pitch_bends:
        note_events_with_pitch_bends = drop_overlapping_pitch_bends(note_events_with_pitch_bends)

    piano_program = pretty_midi.instrument_name_to_program("Electric Piano 1")
    instruments: DefaultDict[int, pretty_midi.Instrument] = defaultdict(
        lambda: pretty_midi.Instrument(program=piano_program)
    )
    bent_notes: DefaultDict[int, List[Tuple[float, float]]] = defaultdict(list)
    instrument_keys = (
        assign_pitch_bend_instruments(note_events_with_pitch_bends)
        if multiple_pitch_bends
        else [0] * len(note_events_with_pitch_bends)
    )
    for (start_time, end_time, note_number, amplitude, pitch_bend), instrument_key in zip(
        note_events_with_pitch_bends, instrument_keys
    ):
        instrument = instruments[instrument_key]
        note = pretty_midi.Note(
            velocity=amplitude_to_velocity(amplitude),
            pitch=note_number,
            start=start_time,
            end=end_time,
        )
        instrument.notes.append(note)
        if not pitch_bend:
            continue
        pitch_bend_times = np.linspace(start_time, end_time, len(pitch_bend))
        pitch_bend_midi_ticks = np.round(np.array(pitch_bend) * PITCH_BEND_SCALE / CONTOURS_BINS_PER_SEMITONE).astype(
            int
        )
        # This supports pitch bends up to 2 semitones
        # If we estimate pitch bends above/below 2 semitones, crop them here when adding them to the midi file
        pitch_bend_midi_ticks[pitch_bend_midi_ticks > N_PITCH_BEND_TICKS - 1] = N_PITCH_BEND_TICKS - 1
        pitch_bend_midi_ticks[pitch_bend_midi_ticks < -N_PITCH_BEND_TICKS] = -N_PITCH_BEND_TICKS
        for pb_time, pb_midi in zip(pitch_bend_times, pitch_bend_midi_ticks):
            instrument.pitch_bends.append(pretty_midi.PitchBend(pb_midi, pb_time))
        bent_notes[instrument_key].append((start_time, end_time))

    # a pitch bend stays in effect until the next one, so return to neutral when a bent note ends
    # (unless the next bent note starts right then and sets its own value). The reset is placed one tick
    # after the note's last bend, since events at the same tick are not guaranteed to keep their order.
    for instrument_key, spans in bent_notes.items():
        spans.sort()
        for i, (_, end_time) in enumerate(spans):
            reset_time = mid.tick_to_time(int(mid.time_to_tick(end_time)) + 1)
            if i + 1 < len(spans) and spans[i + 1][0] <= reset_time:
                continue
            instruments[instrument_key].pitch_bends.append(pretty_midi.PitchBend(0, reset_time))
    mid.instruments.extend(instruments.values())

    return mid


def assign_pitch_bend_instruments(
    note_events: List[Tuple[float, float, int, float, Optional[List[int]]]],
) -> List[int]:
    """Choose an instrument for every note, so that notes sharing a MIDI channel do not sound at the same time.

    A MIDI file has 15 usable channels and a pitch bend applies to a whole channel, so each pitch gets its own
    instrument while there are channels left. After that, a note goes to an instrument that is idle when the
    note starts. When every instrument is busy, the note goes to the one that becomes idle first.

    Args:
        note_events: Note event tuples (start_time_seconds, end_time_seconds, pitch_midi, amplitude, pitch_bends).

    Returns:
        The instrument key of each note, in the order of note_events.
    """
    keys = [0] * len(note_events)
    busy_until: Dict[int, float] = {}  # instrument key -> time its latest note ends
    pitch_key: Dict[int, int] = {}
    for index in sorted(range(len(note_events)), key=lambda i: note_events[i][0]):
        start_time, end_time, note_number = note_events[index][:3]
        key = pitch_key.get(note_number)
        if key is None or busy_until[key] > start_time:
            if len(busy_until) < MAX_MIDI_PITCH_BEND_INSTRUMENTS:
                key = len(busy_until)
            else:
                key = min(busy_until, key=lambda k: (busy_until[k] > start_time, busy_until[k], k))
        pitch_key.setdefault(note_number, key)
        busy_until[key] = max(busy_until.get(key, end_time), end_time)
        keys[index] = key
    return keys


def drop_overlapping_pitch_bends(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]],
) -> List[Tuple[float, float, int, float, Optional[List[int]]]]:
    """Drop pitch bends from any notes that overlap in time with another note"""
    # the pitch bends are left out of the sort key, as a list cannot be compared with None
    note_events = sorted(note_events_with_pitch_bends, key=lambda note: note[:4])
    for i in range(len(note_events) - 1):
        for j in range(i + 1, len(note_events)):
            if note_events[j][0] >= note_events[i][1]:  # start j > end i
                break
            note_events[i] = note_events[i][:-1] + (None,)  # last field is pitch bend
            note_events[j] = note_events[j][:-1] + (None,)

    return note_events


def get_inferred_onsets(onsets: npt.NDArray[Any], frames: npt.NDArray[Any], n_diff: int = 2) -> npt.NDArray[Any]:
    """Infer onsets from large changes in frame amplitudes.

    Args:
        onsets: Array of note onset predictions.
        frames: Audio frames.
        n_diff: Differences used to detect onsets.

    Returns:
        The maximum between the predicted onsets and its differences.
    """
    if frames.shape[0] == 0:
        return onsets

    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    max_frame_diff = np.max(frame_diff)
    if max_frame_diff <= 0:  # no change in the frames, so there is nothing to infer
        return onsets
    frame_diff = np.max(onsets) * frame_diff / max_frame_diff  # rescale to have the same max as onsets

    max_onsets_diff = np.max([onsets, frame_diff], axis=0)  # use the max of the predicted onsets and the differences

    return np.asarray(max_onsets_diff)


def constrain_frequency(
    onsets: npt.NDArray[Any], frames: npt.NDArray[Any], max_freq: Optional[float], min_freq: Optional[float]
) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Zero out activations above or below the max/min frequencies

    Args:
        onsets: Onset activation matrix (n_times, n_freqs)
        frames: Frame activation matrix (n_times, n_freqs)
        max_freq: The maximum frequency to keep.
        min_freq: the minimum frequency to keep.

    Returns:
       The onset and frame activation matrices, with frequencies outside the min and max
       frequency set to 0.
    """
    n_freqs = onsets.shape[1]
    min_freq_idx = 0
    max_freq_idx = n_freqs

    if min_freq is not None:
        min_freq_idx = int(np.clip(np.round(librosa.hz_to_midi(min_freq) - MIDI_OFFSET), 0, n_freqs))
    if max_freq is not None:
        # the bin nearest to max_freq is kept, just as the bin nearest to min_freq is
        max_freq_idx = int(np.clip(np.round(librosa.hz_to_midi(max_freq) - MIDI_OFFSET) + 1, 0, n_freqs))

    onsets = onsets.copy()
    frames = frames.copy()
    onsets[:, :min_freq_idx] = 0
    frames[:, :min_freq_idx] = 0
    onsets[:, max_freq_idx:] = 0
    frames[:, max_freq_idx:] = 0

    return onsets, frames


def model_frames_to_time(n_frames: int) -> npt.NDArray[Any]:
    original_times = librosa.core.frames_to_time(
        np.arange(n_frames),
        sr=AUDIO_SAMPLE_RATE,
        hop_length=FFT_HOP,
    )
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + MAGIC_ALIGNMENT_OFFSET
    times = original_times - (window_offset * window_numbers)
    return np.asarray(times)


HARMONIC_INTERVALS = (12, 19, 24)
HARMONIC_MAX_AMPLITUDE_RATIO = 0.7
HARMONIC_ONSET_TOLERANCE_FRAMES = 5


def suppress_harmonic_notes(
    note_events: List[Tuple[int, int, int, float]],
    max_amplitude_ratio: float = HARMONIC_MAX_AMPLITUDE_RATIO,
    onset_tolerance_frames: int = HARMONIC_ONSET_TOLERANCE_FRAMES,
) -> List[Tuple[int, int, int, float]]:
    """Drop notes that are the overtone of a stronger note starting at the same time.

    A note is dropped when another note starts within `onset_tolerance_frames` of it, sounds 12, 19 or 24
    semitones lower, and is louder than it by more than `max_amplitude_ratio`.

    Args:
        note_events: List of (start_frame, end_frame, pitch_midi, amplitude) tuples.
        max_amplitude_ratio: A note is only dropped if its amplitude is at most this fraction of the lower note's.
        onset_tolerance_frames: Largest start difference, in frames, for two notes to count as simultaneous.

    Returns:
        The note events without the overtones, in their original order.
    """
    by_pitch: DefaultDict[int, List[Tuple[int, int, int, float]]] = defaultdict(list)
    for note in note_events:
        by_pitch[note[2]].append(note)

    kept = []
    for note in note_events:
        start, end, pitch, amplitude = note
        is_overtone = False
        for interval in HARMONIC_INTERVALS:
            for lower in by_pitch.get(pitch - interval, ()):
                if abs(lower[0] - start) <= onset_tolerance_frames and amplitude <= max_amplitude_ratio * lower[3]:
                    is_overtone = True
                    break
            if is_overtone:
                break
        if not is_overtone:
            kept.append(note)
    return kept


def find_onset_peaks(onsets: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Find the frames where an onset activation peaks, per frequency.

    A peak is a frame that is higher than the frame before it and at least as high as the frame after it, so a
    flat top counts once, at its first frame. The first frame can be a peak (the audio may start with a note).
    Frames with no activation are never peaks.

    Args:
        onsets: Onset activation matrix (n_times, n_freqs).

    Returns:
        A tuple of the time indices and the frequency indices of the peaks, like np.where.
    """
    if onsets.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    below = np.full((1, onsets.shape[1]), -np.inf)
    previous = np.concatenate([below, onsets[:-1]])
    following = np.concatenate([onsets[1:], below])
    peaks = np.where((onsets > 0) & (onsets > previous) & (onsets >= following))
    return peaks[0], peaks[1]


def output_to_notes_polyphonic(
    frames: npt.NDArray[Any],
    onsets: npt.NDArray[Any],
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    max_freq: Optional[float],
    min_freq: Optional[float],
    melodia_trick: bool = True,
    energy_tol: int = 11,
    melodia_peak_thresh: float = 0.5,
    note_peak_thresh: float = 0.4,
    suppress_harmonics: bool = True,
) -> List[Tuple[int, int, int, float]]:
    """Decode raw model output to polyphonic note events

    Args:
        frames: Frame activation matrix (n_times, n_freqs).
        onsets: Onset activation matrix (n_times, n_freqs).
        onset_thresh: Minimum amplitude of an onset activation to be considered an onset.
        frame_thresh: Minimum amplitude of a frame activation for a note to remain "on".
        min_note_len: Minimum allowed note length in frames.
        infer_onsets: If True, add additional onsets when there are large differences in frame amplitudes.
        max_freq: Maximum allowed output frequency, in Hz.
        min_freq: Minimum allowed output frequency, in Hz.
        melodia_trick : Whether to use the melodia trick to better detect notes.
        energy_tol: Drop notes below this energy.
        melodia_peak_thresh: A note found by the melodia trick without an onset of at least onset_thresh is dropped
            unless its frame activation reaches this value.
        note_peak_thresh: A note is dropped if its frame activation never reaches this value. Use 0 to keep every note.
        suppress_harmonics: Drop weaker notes that sound 12, 19 or 24 semitones above a note starting at the same time.

    Returns:
        list of tuples [(start_time_frames, end_time_frames, pitch_midi, amplitude)]
        representing the note events, where amplitude is a number between 0 and 1,
        sorted by start time, then pitch
    """

    n_frames = frames.shape[0]
    if n_frames == 0:
        return []

    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    # use onsets inferred from frames in addition to the predicted onsets
    if infer_onsets:
        onsets = get_inferred_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = find_onset_peaks(onsets)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]  # sort to go backwards in time
    onset_freq_idx = onset_idx[1][::-1]  # sort to go backwards in time

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

    # frames already taken by a note, so that a later note of the same pitch cannot run over them
    claimed = np.zeros(frames.shape, dtype=bool)

    # loop over onsets
    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        # if we're too close to the end of the audio, continue
        if note_start_idx >= n_frames - 1:
            continue

        # a note ends where it reaches a note of the same pitch that was already found, as the energy of those frames
        # only counts as used up when it is below the frame threshold
        claimed_after = np.flatnonzero(claimed[note_start_idx + 1 :, freq_idx])
        note_limit = note_start_idx + 1 + claimed_after[0] if len(claimed_after) else n_frames

        # find time index at this frequency band where the frames drop below an energy threshold
        i = note_start_idx + 1
        k = 0  # number of frames since energy dropped below threshold
        while i < note_limit and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1

        i -= k  # go back to frame above threshold

        # if the note is too short, skip it
        if i - note_start_idx < max(min_note_len, 1):
            continue

        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0

        # a note whose frames never become clearly active is not backed by the model's frame output
        if np.max(frames[note_start_idx:i, freq_idx]) < note_peak_thresh:
            continue

        # add the note
        claimed[note_start_idx:i, freq_idx] = True
        amplitude = np.mean(frames[note_start_idx:i, freq_idx])
        note_events.append(
            (
                note_start_idx,
                i,
                freq_idx + MIDI_OFFSET,
                amplitude,
            )
        )

    if melodia_trick:
        energy_shape = remaining_energy.shape

        while np.max(remaining_energy) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0

            # a pass ends where it reaches a note of the same pitch that was already found
            claimed_after = np.flatnonzero(claimed[i_mid + 1 :, freq_idx])
            forward_limit = i_mid + 1 + claimed_after[0] if len(claimed_after) else n_frames
            claimed_before = np.flatnonzero(claimed[:i_mid, freq_idx])
            backward_limit = claimed_before[-1] if len(claimed_before) else -1

            # forward pass
            i = i_mid + 1
            k = 0
            while i < forward_limit and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0

                i += 1

            i_end = i - k

            # backward pass
            i = i_mid - 1
            k = 0
            while i > backward_limit and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0

                i -= 1

            i_start = i + 1 + k  # go back to frame above threshold
            assert i_start >= 0, "{}".format(i_start)
            assert i_end <= n_frames

            if i_end - i_start < max(min_note_len, 1):
                # note is too short, skip it
                continue

            # a note found only by following frame energy needs more than a weak activation to be kept
            peak = np.max(frames[i_start:i_end, freq_idx])
            if peak < note_peak_thresh or (
                peak < melodia_peak_thresh and np.max(onsets[i_start:i_end, freq_idx]) < onset_thresh
            ):
                continue

            # add the note
            claimed[i_start:i_end, freq_idx] = True
            amplitude = np.mean(frames[i_start:i_end, freq_idx])
            note_events.append(
                (
                    i_start,
                    i_end,
                    freq_idx + MIDI_OFFSET,
                    amplitude,
                )
            )

    if suppress_harmonics:
        note_events = suppress_harmonic_notes(note_events)

    # onsets are decoded from the end of the audio backwards and melodia notes come last
    note_events.sort(key=lambda note: (note[0], note[2], note[1]))
    return note_events
