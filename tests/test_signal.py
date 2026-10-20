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

from typing import Any, Dict, cast

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from basic_pitch.layers.signal import NormalizedLog, Spectrogram, Stft  # noqa: E402


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


def test_stft_accepts_numpy_integer_lengths() -> None:
    layer = Stft(**cast(Any, {"fft_length": np.int64(256), "window_length": np.int32(128), "hop_length": np.int64(64)}))
    assert layer(tf.zeros((1, 2048))).shape == (1, 33, 129)


@pytest.mark.parametrize("pad_mode", ["bogus", "", None, 3])
def test_stft_rejects_an_unsupported_pad_mode(pad_mode: Any) -> None:
    with pytest.raises(ValueError, match="pad_mode"):
        Stft(fft_length=256, pad_mode=pad_mode)


@pytest.mark.parametrize("pad_mode", ["CONSTANT", "REFLECT", "SYMMETRIC", "reflect", "Symmetric"])
def test_stft_accepts_the_supported_pad_modes_in_any_letter_case(pad_mode: str) -> None:
    layer = Stft(fft_length=256, pad_mode=pad_mode)
    assert layer(tf.ones((1, 2048))).shape == (1, 33, 129)


@pytest.mark.parametrize("power", [0, -1, float("nan"), float("inf"), True, "2", None])
def test_spectrogram_rejects_a_power_that_is_not_a_finite_number_above_zero(power: Any) -> None:
    with pytest.raises(ValueError, match="power"):
        Spectrogram(power=power, fft_length=256)


@pytest.mark.parametrize("power", [1, 2, 0.5, np.float32(2)])
def test_spectrogram_accepts_positive_powers(power: Any) -> None:
    assert Spectrogram(power=power, fft_length=256)(tf.ones((1, 2048))).shape == (1, 33, 129)


@pytest.mark.parametrize("shape", [(2, 5), (2, 3, 5, 5), (2, 1, 1, 5, 5)])
def test_normalized_log_rejects_unsupported_shapes_with_a_value_error(shape: Any) -> None:
    with pytest.raises(ValueError):
        NormalizedLog()(tf.ones(shape))


@pytest.mark.parametrize("shape", [(2, 4, 5), (2, 1, 4, 5)])
def test_normalized_log_accepts_ranks_three_and_four(shape: Any) -> None:
    assert NormalizedLog()(tf.ones(shape)).shape == (2, 4, 5)
