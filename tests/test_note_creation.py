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

import warnings
from typing import List, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
import pytest

from basic_pitch.note_creation import (
    constrain_frequency,
    drop_overlapping_pitch_bends,
    get_infered_onsets,
    note_events_to_midi,
    output_to_notes_polyphonic,
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


def test_get_infered_onsets_with_silent_frames_returns_onsets_without_warning() -> None:
    onsets = np.zeros((50, 88))
    frames = np.zeros((50, 88))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = get_infered_onsets(onsets, frames)
    np.testing.assert_array_equal(result, onsets)


def test_get_infered_onsets_without_frames_returns_empty_onsets() -> None:
    result = get_infered_onsets(np.zeros((0, 88)), np.zeros((0, 88)))
    assert result.shape == (0, 88)


def test_output_to_notes_polyphonic_without_frames_returns_no_notes() -> None:
    notes = output_to_notes_polyphonic(
        np.zeros((0, 88)), np.zeros((0, 88)), 0.5, 0.3, 11, True, None, None, melodia_trick=True
    )
    assert notes == []


def test_get_infered_onsets_still_adds_onset_where_frames_rise() -> None:
    onsets = np.zeros((50, 88))
    frames = np.zeros((50, 88))
    onsets[0, 0] = 0.5
    frames[20:40, 30] = 0.9
    result = get_infered_onsets(onsets, frames)
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
    onsets[40:45, 40] = onset  # a plateau has no local maximum, so the note can only be found by the melodia step
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
    assert _decode(onsets, frames, melodia_trick=False) == []
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
