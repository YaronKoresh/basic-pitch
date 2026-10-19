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

import numpy as np
import pytest

from basic_pitch.layers.nnaudio import get_window_dispatch


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
        with pytest.raises(Exception, match="only supports"):
            get_window_dispatch([1, 2, 3], 256)
