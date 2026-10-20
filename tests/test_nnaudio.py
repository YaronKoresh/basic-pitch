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

import warnings

from typing import Any, Dict, Tuple

import numpy as np
import pytest
import tensorflow as tf

from basic_pitch.layers.nnaudio import ConstantPad1D, ReflectionPad1D, get_window_dispatch, CQT2010v2


class TestGetWindowDispatch:
    def test_string_window(self) -> None:
        result = get_window_dispatch("hann", 256)
        assert isinstance(result, np.ndarray)
        assert result.shape == (256,)

    def test_gaussian_tuple(self) -> None:
        result = get_window_dispatch(("gaussian", 20), 256)
        assert isinstance(result, np.ndarray)
        assert result.shape == (256,)

    def test_float_kaiser_returns_ndarray(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_window_dispatch(14.0, 256)
        assert isinstance(result, np.ndarray)
        assert result.shape == (256,)
        assert len(w) == 1
        assert "Kaiser" in str(w[0].message)

    def test_non_gaussian_tuple_returns_ndarray(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_window_dispatch(("kaiser", 14), 256)
        assert isinstance(result, np.ndarray)
        assert result.shape == (256,)
        assert len(w) == 1
        assert "Tuple" in str(w[0].message)

    def test_invalid_type_raises(self) -> None:
        window_list: Any = [1, 2, 3]
        with pytest.raises(Exception, match="only supports"):
            get_window_dispatch(window_list, 256)

    def test_gaussian_tuple_with_zero_attenuation_is_flat(self) -> None:
        np.testing.assert_allclose(get_window_dispatch(("gaussian", 0), 8), np.ones(8))

    @pytest.mark.parametrize(
        "window",
        [
            ("gaussian", -1.0),
            ("gaussian", float("nan")),
            ("gaussian", float("inf")),
            ("gaussian", (1, 2)),
            ("gaussian",),
        ],
    )
    def test_invalid_gaussian_tuple_raises_value_error(self, window: Tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="gaussian window"):
            get_window_dispatch(window, 8)  # type: ignore

    @pytest.mark.parametrize("beta", [14, np.float32(14.0), np.int64(14)])
    def test_whole_and_numpy_kaiser_beta_returns_ndarray(self, beta: float) -> None:
        with pytest.warns(UserWarning, match="Kaiser"):
            result = get_window_dispatch(beta, 8)
        np.testing.assert_allclose(result, get_window_dispatch(14.0, 8))

    @pytest.mark.parametrize("window", [True, float("nan"), float("inf"), None])
    def test_other_values_raise(self, window: object) -> None:
        with pytest.raises(Exception, match="only supports"):
            get_window_dispatch(window, 8)  # type: ignore


class TestCQT2010v2PadMode:
    def test_unsupported_pad_mode_raises(self) -> None:
        layer = CQT2010v2(sr=22050, hop_length=256, pad_mode="symmetric")
        with pytest.raises(ValueError, match="Unsupported pad_mode"):
            layer.build(input_shape=(None, 22050))


class TestCQT2010v2Options:
    @pytest.mark.parametrize(
        "options, message",
        [
            ({"n_bins": 0}, "n_bins must be a whole number of at least 1"),
            ({"n_bins": -5}, "n_bins must be a whole number of at least 1"),
            ({"n_bins": 36.5}, "n_bins must be a whole number of at least 1"),
            ({"bins_per_octave": 0}, "bins_per_octave must be a whole number of at least 1"),
            ({"bins_per_octave": -12}, "bins_per_octave must be a whole number of at least 1"),
            ({"hop_length": 0}, "hop_length must be a whole number of at least 1"),
            ({"hop_length": -256}, "hop_length must be a whole number of at least 1"),
            ({"hop_length": True}, "hop_length must be a whole number of at least 1"),
            ({"sr": 0}, "sr must be a finite number above 0"),
            ({"sr": -22050}, "sr must be a finite number above 0"),
            ({"fmin": 0}, "fmin must be a finite number above 0"),
            ({"fmin": -5.0}, "fmin must be a finite number above 0"),
            ({"fmin": float("nan")}, "fmin must be a finite number above 0"),
            ({"fmax": float("inf")}, "fmax must be a finite number above 0"),
            ({"filter_scale": 0}, "filter_scale must be a finite number above 0"),
        ],
    )
    def test_options_that_cannot_work_raise_a_value_error(self, options: Dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            CQT2010v2(**options)

    def test_numpy_numbers_and_no_fmax_are_accepted(self) -> None:
        layer = CQT2010v2(sr=np.float32(22050), hop_length=np.int64(256), n_bins=np.int32(36), fmax=None)
        assert layer.n_bins == 36


class TestCQT2010v2Config:
    @pytest.mark.parametrize("built", [False, True])
    def test_can_be_rebuilt_from_its_config_before_and_after_build(self, built: bool) -> None:
        layer = CQT2010v2(sr=22050, hop_length=256, fmin=32.7, n_bins=36, bins_per_octave=36, name="cqt_layer")
        audio = tf.random.normal((1, 44100))
        expected = layer(audio) if built else None
        rebuilt = CQT2010v2.from_config(layer.get_config())
        assert rebuilt.name == "cqt_layer"
        assert rebuilt.get_config() == layer.get_config()
        output = rebuilt(audio)
        if expected is not None:
            np.testing.assert_allclose(output.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6)


class TestCQT2010v2ShortInput:
    def test_input_shorter_than_the_kernel_is_padded_with_zeros_and_warns(self) -> None:
        layer = CQT2010v2(sr=22050, hop_length=256, fmin=32.7, n_bins=36, bins_per_octave=36)
        with pytest.warns(UserWarning, match="constant padding"):
            output = layer(tf.zeros((1, 22050)))
        assert output.shape[0] == 1
        assert output.shape[2] == 36
        assert np.all(output.numpy() == 0)


class TestPad1D:
    @pytest.mark.parametrize("layer_class", [ReflectionPad1D, ConstantPad1D])
    @pytest.mark.parametrize(
        "padding, left, right",
        [(3, 3, 3), (np.int64(2), 2, 2), ((4,), 4, 4), ((1, 2), 1, 2), ((0, 3), 0, 3)],
    )
    def test_pads_each_side_and_reports_the_padded_width(
        self, layer_class: Any, padding: Any, left: int, right: int
    ) -> None:
        x = np.arange(10, dtype="float32").reshape(1, 1, 10)
        layer = layer_class(padding)
        result = layer(x).numpy()
        assert result.shape == (1, 1, 10 + left + right)
        assert layer.compute_output_shape((1, 1, 10)) == (1, 1, 10 + left + right)
        np.testing.assert_array_equal(result[0, 0, left : left + 10], x[0, 0])

    @pytest.mark.parametrize("layer_class", [ReflectionPad1D, ConstantPad1D])
    @pytest.mark.parametrize("padding", [-1, (1, -2), (1, 2, 3), (), 1.5, True, "2"])
    def test_rejects_padding_that_cannot_be_used(self, layer_class: Any, padding: Any) -> None:
        with pytest.raises(ValueError, match="padding"):
            layer_class(padding)

    @pytest.mark.parametrize("layer_class", [ReflectionPad1D, ConstantPad1D])
    def test_can_be_rebuilt_from_its_config(self, layer_class: Any) -> None:
        layer = layer_class((1, 2))
        rebuilt = layer_class.from_config(layer.get_config())
        assert rebuilt.left_right_padding == (1, 2)

    def test_constant_padding_uses_its_value(self) -> None:
        result = ConstantPad1D((1, 2), value=5)(np.zeros((1, 1, 4), dtype="float32")).numpy()
        assert result[0, 0].tolist() == [5, 0, 0, 0, 0, 5, 5]
