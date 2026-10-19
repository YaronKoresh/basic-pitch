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

import pathlib
import warnings
from typing import List, Optional, Tuple

import librosa
import mido
import numpy as np
import pretty_midi
import pytest

from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
from basic_pitch.inference import DEFAULT_MINIMUM_NOTE_LENGTH_MS

from basic_pitch.note_creation import (
    amplitude_to_velocity,
    assign_pitch_bend_instruments,
    constrain_frequency,
    drop_overlapping_pitch_bends,
    find_onset_peaks,
    get_inferred_onsets,
    get_pitch_bends,
    model_output_to_notes,
    note_events_to_midi,
    output_to_notes_polyphonic,
    sonify_salience,
    suppress_harmonic_notes,
)


def test_drop_overlapping_pitch_bends() -> None:
    # events are: (start, end, pitch, amplitude, pitch_bend)
    note_events_with_pitch_bends = [
        (0.0, 0.1, 60, 1.0, None),
        (2.0, 2.1, 62, 1.0, [0, 1, 2]),  # οverlaps w next
        (2.0, 2.1, 64, 1.0, [0, 1, 2]),  # overlaps w prev
        (1.0, 1.1, 65, 1.0, [0, 1, 2]),
        (1.1, 1.2, 67, 1.0, [0, 1, 2]),
        (3.0, 3.2, 69, 1.0, [0, 1, 2]),  # overlaps w next
        (3.1, 3.3, 71, 1.0, [0, 1, 2]),  # overlaps w prev
        (5.0, 5.1, 72, 1.0, [0, 1, 2]),  # overlaps w next
        (5.0, 5.2, 74, 1.0, [0, 1, 2]),  # overlaps w prev
        (4.0, 4.2, 76, 1.0, [0, 1, 2]),  # overlaps w next
        (4.1, 4.2, 77, 1.0, [0, 1, 2]),  # overlaps w prev
    ]
    expected = [
        (0.0, 0.1, 60, 1.0, None),
        (2.0, 2.1, 62, 1.0, None),  # overlaps w next
        (2.0, 2.1, 64, 1.0, None),  # overlaps w prev
        (1.0, 1.1, 65, 1.0, [0, 1, 2]),
        (1.1, 1.2, 67, 1.0, [0, 1, 2]),
        (3.0, 3.2, 69, 1.0, None),  # overlaps w next
        (3.1, 3.3, 71, 1.0, None),  # overlaps w prev
        (5.0, 5.1, 72, 1.0, None),  # overlaps w next
        (5.0, 5.2, 74, 1.0, None),  # overlaps w prev
        (4.0, 4.2, 76, 1.0, None),  # overlaps w next
        (4.1, 4.2, 77, 1.0, None),  # overlaps w prev
    ]
    result = drop_overlapping_pitch_bends(note_events_with_pitch_bends)
    assert sorted(result) == sorted(expected)


def _single_note_activations() -> Tuple[np.ndarray, np.ndarray]:
    onsets = np.zeros((100, 88))
    frames = np.zeros((100, 88))
    frames[10:60, 40] = 0.9
    onsets[10, 40] = 0.9
    return onsets, frames


@pytest.mark.parametrize("min_freq", [1.0, 10.0, 20.0, 27.5])
def test_constrain_frequency_min_below_lowest_key_keeps_notes(min_freq: float) -> None:
    onsets, frames = _single_note_activations()
    onsets, frames = constrain_frequency(onsets, frames, None, min_freq)
    assert frames[:, 40].sum() > 0
    assert onsets[10, 40] > 0


@pytest.mark.parametrize("max_freq", [1.0, 15.0, 20.0])
def test_constrain_frequency_max_below_lowest_key_removes_notes(max_freq: float) -> None:
    onsets, frames = _single_note_activations()
    onsets, frames = constrain_frequency(onsets, frames, max_freq, None)
    assert frames.sum() == 0
    assert onsets.sum() == 0


def test_constrain_frequency_keeps_bin_nearest_to_max_freq() -> None:
    onsets, frames = _single_note_activations()
    max_freq = librosa.midi_to_hz(40 + 21)
    onsets, frames = constrain_frequency(onsets, frames, max_freq, None)
    assert frames[:, 40].sum() > 0


def test_constrain_frequency_does_not_modify_inputs() -> None:
    onsets, frames = _single_note_activations()
    onsets_before, frames_before = onsets.copy(), frames.copy()
    constrain_frequency(onsets, frames, 100.0, 5000.0)
    np.testing.assert_array_equal(onsets, onsets_before)
    np.testing.assert_array_equal(frames, frames_before)


def _bends(midi: pretty_midi.PrettyMIDI) -> List[Tuple[float, int]]:
    return sorted((float(b.time), int(b.pitch)) for b in midi.instruments[0].pitch_bends)


def test_note_events_to_midi_resets_pitch_bend_when_bent_note_ends() -> None:
    events: List[Tuple[float, float, int, float, Optional[List[int]]]] = [
        (0.0, 1.0, 60, 0.8, [3, 3, 3]),
        (1.5, 2.5, 64, 0.8, None),
        (3.0, 3.5, 67, 0.8, [0, 0]),
    ]
    bends = _bends(note_events_to_midi(events))
    after_first_note = [(time, pitch) for time, pitch in bends if 1.0 < time < 3.0]
    assert len(after_first_note) == 1
    assert after_first_note[0][1] == 0
    assert after_first_note[0][0] == pytest.approx(1.0, abs=0.01)


def test_note_events_to_midi_does_not_reset_between_back_to_back_bent_notes() -> None:
    events: List[Tuple[float, float, int, float, Optional[List[int]]]] = [
        (0.0, 1.0, 60, 0.8, [3, 3]),
        (1.0, 2.0, 62, 0.8, [-3, -3]),
    ]
    bends = _bends(note_events_to_midi(events))
    assert all(pitch != 0 for time, pitch in bends if 0.9 < time < 1.1)
    assert bends[-1][1] == 0
    assert bends[-1][0] == pytest.approx(2.0, abs=0.01)


def test_note_events_to_midi_resets_pitch_bend_per_instrument() -> None:
    events: List[Tuple[float, float, int, float, Optional[List[int]]]] = [
        (0.0, 1.0, 60, 0.8, [3, 3]),
        (0.0, 2.0, 64, 0.8, [3, 3]),
    ]
    midi = note_events_to_midi(events, multiple_pitch_bends=True)
    ends = {inst.notes[0].pitch: float(inst.pitch_bends[-1].time) for inst in midi.instruments}
    assert ends[60] == pytest.approx(1.0, abs=0.01)
    assert ends[64] == pytest.approx(2.0, abs=0.01)
    assert all(inst.pitch_bends[-1].pitch == 0 for inst in midi.instruments)


def test_note_events_to_midi_keeps_simultaneous_bent_notes_on_separate_channels_with_many_pitches(
    tmp_path: pathlib.Path,
) -> None:
    # 15 short notes fill every usable channel; the 16th pitch then sounds together with the 1st pitch
    events: List[Tuple[float, float, int, float, Optional[List[int]]]] = [
        (0.1 * i, 0.1 * i + 0.05, 40 + i, 0.8, [0, 0]) for i in range(15)
    ]
    events.append((10.0, 11.0, 40, 0.8, [6, 6, 6]))
    events.append((10.0, 11.0, 55, 0.8, [-6, -6, -6]))
    midi = note_events_to_midi(events, multiple_pitch_bends=True)
    path = tmp_path / "many_pitches.mid"
    midi.write(str(path))
    # a pitch bend applies to a whole channel, so the two late notes need different channels
    channel_of_late_note = {}
    for track in mido.MidiFile(str(path)).tracks:
        elapsed_s = 0.0
        for message in mido.merge_tracks([track]):
            elapsed_s += mido.tick2second(message.time, 220, 500000)
            if message.type == "note_on" and message.velocity > 0 and elapsed_s >= 10.0:
                channel_of_late_note[message.note] = message.channel
    assert set(channel_of_late_note) == {40, 55}
    assert channel_of_late_note[40] != channel_of_late_note[55]


def test_assign_pitch_bend_instruments_uses_at_most_15_instruments_and_never_overlaps_notes_in_one() -> None:
    rng = np.random.default_rng(3)
    starts = np.sort(rng.uniform(0, 30, size=200))
    events: List[Tuple[float, float, int, float, Optional[List[int]]]] = [
        (float(s), float(s + rng.uniform(0.05, 0.4)), int(rng.integers(21, 109)), 0.8, [1, 2]) for s in starts
    ]
    keys = assign_pitch_bend_instruments(events)
    assert len(set(keys)) <= 15
    for key in set(keys):
        notes = sorted(event[:2] for event, k in zip(events, keys) if k == key)
        assert all(previous[1] <= following[0] for previous, following in zip(notes, notes[1:]))


def test_assign_pitch_bend_instruments_gives_each_pitch_its_own_instrument_when_channels_suffice() -> None:
    events: List[Tuple[float, float, int, float, Optional[List[int]]]] = [
        (0.0, 1.0, 60, 0.8, [1]),
        (2.0, 3.0, 64, 0.8, [1]),
        (4.0, 5.0, 60, 0.8, [1]),
    ]
    keys = assign_pitch_bend_instruments(events)
    assert keys[0] == keys[2]
    assert keys[0] != keys[1]


def test_get_inferred_onsets_with_silent_frames_returns_onsets_without_warning() -> None:
    onsets = np.zeros((50, 88))
    frames = np.zeros((50, 88))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = get_inferred_onsets(onsets, frames)
    np.testing.assert_array_equal(result, onsets)


def test_get_inferred_onsets_without_frames_returns_empty_onsets() -> None:
    result = get_inferred_onsets(np.zeros((0, 88)), np.zeros((0, 88)))
    assert result.shape == (0, 88)


def test_output_to_notes_polyphonic_without_frames_returns_no_notes() -> None:
    notes = output_to_notes_polyphonic(
        np.zeros((0, 88)), np.zeros((0, 88)), 0.5, 0.3, 11, True, None, None, melodia_trick=True
    )
    assert notes == []


def test_get_inferred_onsets_still_adds_onset_where_frames_rise() -> None:
    onsets = np.zeros((50, 88))
    frames = np.zeros((50, 88))
    onsets[0, 0] = 0.5
    frames[20:40, 30] = 0.9
    result = get_inferred_onsets(onsets, frames)
    assert result[20, 30] == pytest.approx(0.5)


def test_note_events_to_midi_emits_no_warnings_for_many_bent_notes() -> None:
    rng = np.random.default_rng(1)
    starts = np.sort(rng.uniform(0, 60, size=300))
    events: List[Tuple[float, float, int, float, Optional[List[int]]]] = [
        (float(s), float(s + rng.uniform(0.05, 0.3)), int(rng.integers(21, 109)), 0.8, [1, 2, 1]) for s in starts
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        note_events_to_midi(events, multiple_pitch_bends=True)


def _sustained_activation(peak: float, onset: float) -> Tuple[np.ndarray, np.ndarray]:
    frames = np.zeros((100, 88))
    onsets = np.zeros((100, 88))
    frames[20:70, 40] = peak
    onsets[40:45, 40] = onset
    return onsets, frames


def _decode(onsets: np.ndarray, frames: np.ndarray, melodia_trick: bool = True) -> list:  # type: ignore[type-arg]
    return output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None, melodia_trick=melodia_trick)


def test_melodia_drops_weak_note_without_onset() -> None:
    onsets, frames = _sustained_activation(peak=0.4, onset=0.0)
    assert _decode(onsets, frames) == []


def test_melodia_keeps_strong_note_without_onset() -> None:
    onsets, frames = _sustained_activation(peak=0.7, onset=0.0)
    notes = _decode(onsets, frames)
    assert len(notes) == 1
    assert notes[0][2] == 40 + 21


def test_melodia_keeps_weak_note_with_strong_onset() -> None:
    onsets, frames = _sustained_activation(peak=0.4, onset=0.9)
    assert len(_decode(onsets, frames)) == 1


def test_melodia_peak_threshold_can_be_lowered() -> None:
    onsets, frames = _sustained_activation(peak=0.4, onset=0.0)
    notes = output_to_notes_polyphonic(
        frames, onsets, 0.5, 0.3, 11, False, None, None, melodia_trick=True, melodia_peak_thresh=0.0
    )
    assert len(notes) == 1


def test_note_with_weak_frames_is_dropped_even_with_a_strong_onset() -> None:
    onsets = np.zeros((100, 88))
    frames = np.zeros((100, 88))
    frames[20:70, 40] = 0.35
    onsets[20, 40] = 0.9
    assert output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None) == []


def test_note_with_clear_frames_and_onset_is_kept() -> None:
    onsets = np.zeros((100, 88))
    frames = np.zeros((100, 88))
    frames[20:70, 40] = 0.45
    onsets[20, 40] = 0.9
    notes = output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None)
    assert [(n[0], n[2]) for n in notes] == [(20, 40 + 21)]


def test_note_peak_threshold_can_be_disabled() -> None:
    onsets = np.zeros((100, 88))
    frames = np.zeros((100, 88))
    frames[20:70, 40] = 0.35
    onsets[20, 40] = 0.9
    notes = output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None, note_peak_thresh=0.0)
    assert len(notes) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"min_freq": -5.0},
        {"min_freq": 0.0},
        {"max_freq": -1.0},
        {"min_freq": float("nan")},
        {"max_freq": float("inf")},
        {"min_freq": 500.0, "max_freq": 100.0},
        {"midi_tempo": 0.0},
        {"midi_tempo": -120.0},
        {"midi_tempo": 1.0},
        {"midi_tempo": 1e9},
        {"min_note_len": -1},
        {"onset_thresh": 2.0},
        {"frame_thresh": -0.1},
        {"onset_thresh": float("nan")},
        {"note_peak_thresh": 1.5},
        {"note_peak_thresh": -0.1},
        {"note_peak_thresh": float("nan")},
        {"melodia_peak_thresh": 2.0},
        {"melodia_peak_thresh": -1.0},
        {"melodia_peak_thresh": float("inf")},
    ],
)
def test_model_output_to_notes_rejects_out_of_range_options(options: dict) -> None:
    n_frames = 20
    output = {
        "note": np.zeros((n_frames, 88), dtype=np.float32),
        "onset": np.zeros((n_frames, 88), dtype=np.float32),
        "contour": np.zeros((n_frames, 264), dtype=np.float32),
    }
    kwargs = {"onset_thresh": 0.5, "frame_thresh": 0.3, **options}
    with pytest.raises(ValueError, match="must"):
        model_output_to_notes(output, **kwargs)


def test_sonify_salience_does_not_modify_its_input() -> None:
    gram = np.random.default_rng(0).random((264, 20))
    original = gram.copy()
    sonify_salience(gram, 3)
    np.testing.assert_array_equal(gram, original)


@pytest.mark.parametrize("n_times", [0, 1])
def test_sonify_salience_handles_grams_with_fewer_than_two_frames(n_times: int, tmp_path) -> None:  # type: ignore
    gram = np.ones((264, n_times))
    save_path = tmp_path / "salience.wav"
    audio, sample_rate = sonify_salience(gram, 3, save_path=str(save_path))
    assert sample_rate == 3000
    assert not np.any(audio)
    assert (len(audio) == 0) == (n_times == 0)
    assert save_path.exists()


@pytest.mark.parametrize(
    "amplitude,expected", [(0.0, 1), (0.001, 1), (0.5, 64), (1.0, 127), (1.2, 127), (-0.1, 1), (float("nan"), 1)]
)
def test_amplitude_to_velocity_stays_in_midi_range(amplitude: float, expected: int) -> None:
    assert amplitude_to_velocity(amplitude) == expected


@pytest.mark.parametrize("amplitude", [1.2, -0.1, float("nan")])
def test_note_events_to_midi_output_can_be_written_with_out_of_range_amplitude(amplitude: float, tmp_path) -> None:  # type: ignore
    midi = note_events_to_midi([(0.0, 1.0, 60, amplitude, None)])
    midi.write(str(tmp_path / "out.mid"))
    assert 1 <= midi.instruments[0].notes[0].velocity <= 127


def test_quiet_note_is_kept_when_midi_is_saved_and_loaded(tmp_path) -> None:  # type: ignore
    midi = note_events_to_midi([(0.0, 1.0, 60, 0.002, None)])
    path = tmp_path / "quiet.mid"
    midi.write(str(path))
    loaded = pretty_midi.PrettyMIDI(str(path))
    assert [note.pitch for instrument in loaded.instruments for note in instrument.notes] == [60]


def test_note_events_are_sorted_by_start_time() -> None:
    onsets = np.zeros((300, 88))
    frames = np.zeros((300, 88))
    for start, pitch_idx in [(20, 40), (100, 45), (200, 38)]:
        frames[start : start + 50, pitch_idx] = 0.8
        onsets[start, pitch_idx] = 0.9
    # a note found only by the melodia step, earlier than the last note found by onset
    frames[60:90, 60] = 0.8
    notes = output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None)
    assert len(notes) == 4
    assert [n[0] for n in notes] == sorted(n[0] for n in notes)
    assert [n[2] for n in notes] == [40 + 21, 60 + 21, 45 + 21, 38 + 21]


def test_model_output_to_notes_returns_chronological_note_events() -> None:
    onsets = np.zeros((300, 88))
    frames = np.zeros((300, 88))
    for start, pitch_idx in [(20, 40), (100, 45), (200, 38)]:
        frames[start : start + 50, pitch_idx] = 0.8
        onsets[start, pitch_idx] = 0.9
    output = {"note": frames, "onset": onsets, "contour": np.zeros((300, 264))}
    _, events = model_output_to_notes(output, onset_thresh=0.5, frame_thresh=0.3)
    starts = [e[0] for e in events]
    assert len(starts) == 3
    assert starts == sorted(starts)


def test_suppress_harmonic_notes_drops_weaker_octave_note() -> None:
    events = [(10, 40, 40, 0.8), (10, 30, 52, 0.4)]
    assert suppress_harmonic_notes(events) == [(10, 40, 40, 0.8)]


@pytest.mark.parametrize("interval", [12, 19, 24])
def test_suppress_harmonic_notes_covers_octave_fifth_and_double_octave(interval: int) -> None:
    events = [(10, 40, 40, 0.8), (10, 30, 40 + interval, 0.4)]
    assert suppress_harmonic_notes(events) == [(10, 40, 40, 0.8)]


def test_suppress_harmonic_notes_keeps_octave_of_similar_strength() -> None:
    events = [(10, 40, 40, 0.8), (10, 40, 52, 0.7)]
    assert suppress_harmonic_notes(events) == events


def test_suppress_harmonic_notes_keeps_notes_at_other_times_and_other_intervals() -> None:
    events = [(10, 40, 40, 0.8), (30, 60, 52, 0.2), (10, 40, 47, 0.2)]
    assert suppress_harmonic_notes(events) == events


def test_suppress_harmonic_notes_drops_an_overtone_that_rings_longer_than_the_lower_note() -> None:
    events = [(10, 20, 40, 0.8), (10, 60, 52, 0.4)]
    assert suppress_harmonic_notes(events) == [(10, 20, 40, 0.8)]


def test_output_to_notes_polyphonic_suppresses_harmonics_by_default() -> None:
    frames = np.zeros((100, 88))
    onsets = np.zeros((100, 88))
    for row, peak in ((30, 0.9), (42, 0.5)):
        frames[20:70, row] = peak
        onsets[20, row] = 0.9
    kept = output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None)
    assert [note[2] for note in kept] == [30 + 21]
    both = output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None, suppress_harmonics=False)
    assert [note[2] for note in both] == [30 + 21, 42 + 21]


@pytest.mark.parametrize("pitch", [21, 60, 108])
def test_get_pitch_bends_without_contour_activity_has_no_bend(pitch: int) -> None:
    contours = np.zeros((20, 264))
    (event,) = get_pitch_bends(contours, [(2, 10, pitch, 0.5)])
    assert list(event[4]) == [0] * 8


def test_get_pitch_bends_follows_contour_peak() -> None:
    contours = np.zeros((20, 264))
    centre = int(np.round(12.0 * 3 * np.log2(librosa.midi_to_hz(60) / librosa.midi_to_hz(21))))
    contours[2:10, centre + 3] = 1.0
    (event,) = get_pitch_bends(contours, [(2, 10, 60, 0.5)])
    assert list(event[4]) == [3] * 8


def test_note_events_to_midi_with_notes_tied_on_start_end_pitch_and_amplitude() -> None:
    tied_notes: List[Tuple[float, float, int, float, Optional[List[int]]]] = [
        (0.0, 1.0, 60, 0.5, [1, 2]),
        (0.0, 1.0, 60, 0.5, None),
    ]
    for events in (tied_notes, tied_notes[::-1]):
        midi = note_events_to_midi(events)
        assert len(midi.instruments[0].notes) == 2
        assert not any(bend.pitch for bend in midi.instruments[0].pitch_bends)


def _held_pitch_activations(n_frames: int = 41) -> Tuple[np.ndarray, np.ndarray]:
    frames = np.zeros((n_frames, 88))
    onsets = np.zeros((n_frames, 88))
    frames[10:, 49] = 0.9
    return frames, onsets


def test_melodia_note_does_not_run_over_onset_note_at_end_of_audio() -> None:
    frames, onsets = _held_pitch_activations()
    onsets[32, 49] = 0.9
    frames[40, 49] = 0.95  # the last frame is never part of an onset note, but it is the strongest leftover
    notes = output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 7, False, None, None)
    spans = sorted((start, end) for start, end, _, _ in notes)
    assert (32, 41) in spans
    for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
        assert first_end <= second_start


def test_melodia_note_does_not_run_over_onset_note_in_the_middle() -> None:
    frames = np.zeros((80, 88))
    onsets = np.zeros((80, 88))
    frames[10:70, 49] = 0.6
    frames[40:48, 49] = 0.9
    frames[70, 49] = 0.0
    onsets[40, 49] = 0.9
    notes = output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 5, False, None, None)
    spans = sorted((start, end) for start, end, _, _ in notes)
    assert len(spans) >= 2
    for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
        assert first_end <= second_start


def test_notes_of_one_pitch_never_overlap_on_random_activations() -> None:
    rng = np.random.default_rng(3)
    for _ in range(60):
        n_frames = int(rng.integers(30, 100))
        frames = rng.random((n_frames, 88)) ** 3
        onsets = rng.random((n_frames, 88)) ** 3
        notes = output_to_notes_polyphonic(
            frames, onsets, 0.5, 0.3, int(rng.integers(0, 12)), bool(rng.integers(0, 2)), None, None
        )
        by_pitch = {}
        for start, end, pitch, _ in notes:
            by_pitch.setdefault(pitch, []).append((start, end))
        for spans in by_pitch.values():
            spans.sort()
            for (_, first_end), (second_start, _) in zip(spans, spans[1:]):
                assert first_end <= second_start


def test_note_with_flat_topped_onset_is_found_without_melodia() -> None:
    onsets, frames = _sustained_activation(peak=0.9, onset=0.9)
    notes = _decode(onsets, frames, melodia_trick=False)
    assert [(n[0], n[2]) for n in notes] == [(40, 40 + 21)]


def test_find_onset_peaks_counts_a_flat_top_once_at_its_first_frame() -> None:
    onsets = np.zeros((10, 2))
    onsets[3:6, 0] = 0.8
    onsets[7, 1] = 0.6
    onsets[5, 1] = 0.4
    times, freqs = find_onset_peaks(onsets)
    assert sorted(zip(times.tolist(), freqs.tolist())) == [(3, 0), (5, 1), (7, 1)]


def test_find_onset_peaks_accepts_the_first_frame_and_ignores_silence() -> None:
    onsets = np.zeros((6, 2))
    onsets[0, 0] = 0.9
    onsets[1, 0] = 0.3
    times, freqs = find_onset_peaks(onsets)
    assert list(zip(times.tolist(), freqs.tolist())) == [(0, 0)]
    assert len(find_onset_peaks(np.zeros((0, 88)))[0]) == 0


def _short_note_activation(length: int) -> Tuple[np.ndarray, np.ndarray]:
    frames = np.zeros((100, 88))
    onsets = np.zeros((100, 88))
    frames[20 : 20 + length, 40] = 0.9
    onsets[20, 40] = 0.9
    return onsets, frames


def test_note_exactly_as_long_as_the_minimum_note_length_is_kept() -> None:
    onsets, frames = _short_note_activation(11)
    notes = output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None, melodia_trick=False)
    assert [(n[0], n[1], n[2]) for n in notes] == [(20, 31, 40 + 21)]


def test_note_shorter_than_the_minimum_note_length_is_dropped() -> None:
    onsets, frames = _short_note_activation(10)
    assert output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None, melodia_trick=False) == []


def test_note_extending_to_last_frame_includes_that_frame() -> None:
    n_frames = 40
    onsets = np.zeros((n_frames, 88))
    frames = np.zeros((n_frames, 88))
    frames[5:n_frames, 40] = 0.9
    onsets[5, 40] = 0.9
    notes = output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 11, False, None, None, melodia_trick=False)
    assert len(notes) == 1
    assert notes[0][1] == n_frames


def test_default_minimum_note_length_keeps_a_sixteenth_note_at_115_bpm() -> None:
    sixteenth_s = 60 / 115 / 4
    min_note_len = int(np.round(DEFAULT_MINIMUM_NOTE_LENGTH_MS / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    # a played note sounds a little shorter than its grid slot
    assert int(0.9 * sixteenth_s * AUDIO_SAMPLE_RATE / FFT_HOP) >= min_note_len


def test_note_events_to_midi_rejects_tempo_that_overflows_midi_ticks() -> None:
    with pytest.raises(ValueError, match="must be lower"):
        note_events_to_midi([(0.0, 600.0, 60, 0.5, [0, 3, 0])], midi_tempo=5000)


def test_note_events_to_midi_writes_lowest_and_high_tempos(tmp_path: pathlib.Path) -> None:
    for tempo in (3.58, 120, 1000):
        midi = note_events_to_midi([(0.0, 60.0, 60, 0.5, [0, 3, 0])], midi_tempo=tempo)
        midi.write(str(tmp_path / "tempo.mid"))
