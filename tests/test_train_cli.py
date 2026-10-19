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

import sys
from typing import Any, List
from unittest import mock

import pytest

pytest.importorskip("tensorflow")

from basic_pitch import train  # noqa: E402

BASE_ARGS = ["train", "--source", "source", "--output", "output", "--guitarset"]


def run_train_cli(extra_args: List[str]) -> Any:
    with mock.patch.object(sys, "argv", BASE_ARGS + extra_args), mock.patch.object(train, "main") as main:
        train.console_entry_point()
    return main


def test_valid_options_reach_main() -> None:
    main = run_train_cli(["--training-shuffle-size", "0", "--positive-onset-weight", "1"])
    main.assert_called_once()


@pytest.mark.parametrize(
    "extra_args, message",
    [
        (["--epochs", "0"], "--epochs"),
        (["--batch-size", "0"], "--batch-size"),
        (["--batch-size", "-4"], "--batch-size"),
        (["--steps-per-epoch", "-1"], "--steps-per-epoch"),
        (["--validation-steps", "0"], "--validation-steps"),
        (["--validation-steps", "-3"], "--validation-steps"),
        (["--training-shuffle-size", "-5"], "--training-shuffle-size"),
        (["--size-evaluation-callback-datasets", "0"], "--size-evaluation-callback-datasets"),
        (["--learning-rate", "-1"], "--learning-rate"),
        (["--learning-rate", "0"], "--learning-rate"),
        (["--learning-rate", "nan"], "--learning-rate"),
        (["--positive-onset-weight", "2"], "--positive-onset-weight"),
        (["--positive-onset-weight", "nan"], "--positive-onset-weight"),
    ],
)
def test_invalid_options_exit_with_a_usage_error_before_training(
    extra_args: List[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with mock.patch.object(train, "main") as main:
        with mock.patch.object(sys, "argv", BASE_ARGS + extra_args):
            with pytest.raises(SystemExit) as exit_info:
                train.console_entry_point()
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    main.assert_not_called()
