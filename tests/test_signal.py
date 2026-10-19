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

from typing import Any, Dict

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from basic_pitch.layers.signal import Spectrogram, Stft  # noqa: E402


def test_stft_defaults_are_derived_from_the_fft_length() -> None:
    layer = Stft(fft_length=256)
    assert layer.window_length == 256
    assert layer.hop_length == 64
    assert layer(tf.zeros((1, 2048))).shape == (1, 33, 129)


def test_stft_can_be_rebuilt_from_its_config() -> None:
    layer = Stft(fft_length=256, window_length=128, hop_length=64)
    rebuilt = Stft.from_config(layer.get_config())
    assert (rebuilt.fft_length, rebuilt.window_length, rebuilt.hop_length) == (256, 128, 64)
    signal = np.random.default_rng(0).random((1, 2048)).astype(np.float32)
    np.testing.assert_allclose(rebuilt(signal), layer(signal))


def test_spectrogram_can_be_rebuilt_from_its_config() -> None:
    layer = Spectrogram(power=1, fft_length=256)
    rebuilt = Spectrogram.from_config(layer.get_config())
    assert rebuilt.power == 1
    assert rebuilt.fft_length == 256


@pytest.mark.parametrize(
    "arguments",
    [
        {"hop_length": 0},
        {"hop_length": -5},
        {"window_length": 0},
        {"window_length": -1},
        {"fft_length": 0},
        {"window_length": 512},
        {"hop_length": 1.5},
    ],
)
def test_stft_rejects_lengths_it_cannot_use(arguments: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Stft(**{"fft_length": 256, **arguments})
