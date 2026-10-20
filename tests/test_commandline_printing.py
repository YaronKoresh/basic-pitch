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

import os

import pytest

from basic_pitch.commandline_printing import TF_LOG_LEVEL_KEY, no_tf_warnings


def test_no_tf_warnings_silences_inside_and_removes_variable_afterwards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TF_LOG_LEVEL_KEY, raising=False)
    with no_tf_warnings():
        assert os.environ[TF_LOG_LEVEL_KEY] == "3"
    assert TF_LOG_LEVEL_KEY not in os.environ


def test_no_tf_warnings_restores_existing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TF_LOG_LEVEL_KEY, "1")
    with no_tf_warnings():
        assert os.environ[TF_LOG_LEVEL_KEY] == "3"
    assert os.environ[TF_LOG_LEVEL_KEY] == "1"
