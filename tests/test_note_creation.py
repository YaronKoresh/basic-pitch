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

from typing import Tuple

import librosa
import numpy as np
import pytest

from basic_pitch.note_creation import constrain_frequency, drop_overlapping_pitch_bends


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
