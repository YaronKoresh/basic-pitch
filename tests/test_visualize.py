#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2024 Spotify AB
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

import numpy as np
import pytest
import tensorflow as tf

from basic_pitch import visualize
from basic_pitch.constants import ANNOT_N_FRAMES, N_FREQ_BINS_NOTES


@pytest.mark.parametrize("n_frames", [ANNOT_N_FRAMES - 5, ANNOT_N_FRAMES, ANNOT_N_FRAMES + 5, 2 * ANNOT_N_FRAMES])
def test_array_to_sonification_handles_grams_shorter_and_longer_than_the_annotation_times(n_frames: int) -> None:
    gram = tf.constant(np.random.default_rng(0).random((2, n_frames, N_FREQ_BINS_NOTES)), dtype=tf.float32)

    audio = visualize._array_to_sonification(gram, visualize.MAX_OUTPUTS)

    assert audio.shape[0] == 2
    assert audio.shape[2] == 1
    assert audio.shape[1] > 0
