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
import sys
from typing import Any, List
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from basic_pitch import train  # noqa: E402


def make_source(root: pathlib.Path, splits: List[str]) -> str:
    for split in splits:
        split_dir = root / "guitarset" / "splits" / split
        split_dir.mkdir(parents=True)
        (split_dir / "a.tfrecord").write_bytes(b"")
    return str(root)


def base_args(source: str) -> List[str]:
    return ["train", "--source", source, "--output", "output", "--guitarset"]


@pytest.fixture
def source(tmp_path: pathlib.Path) -> str:
    return make_source(tmp_path, ["train", "validation"])


def run_train_cli(source: str, extra_args: List[str]) -> Any:
    with mock.patch.object(sys, "argv", base_args(source) + extra_args), mock.patch.object(train, "main") as main:
        train.console_entry_point()
    return main


def test_valid_options_reach_main(source: str) -> None:
    main = run_train_cli(source, ["--training-shuffle-size", "0", "--positive-onset-weight", "1"])
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
    source: str, extra_args: List[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with mock.patch.object(train, "main") as main:
        with mock.patch.object(sys, "argv", base_args(source) + extra_args):
            with pytest.raises(SystemExit) as exit_info:
                train.console_entry_point()
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    main.assert_not_called()


@pytest.mark.parametrize("splits", [[], ["train"], ["validation"]])
def test_source_without_train_and_validation_files_exits_with_a_usage_error_before_training(
    tmp_path: pathlib.Path, splits: List[str], capsys: pytest.CaptureFixture[str]
) -> None:
    source = make_source(tmp_path, splits)
    with mock.patch.object(train, "main") as main:
        with mock.patch.object(sys, "argv", base_args(source)):
            with pytest.raises(SystemExit) as exit_info:
                train.console_entry_point()
    assert exit_info.value.code == 2
    assert "No tfrecord files found" in capsys.readouterr().err
    main.assert_not_called()


def test_main_rejects_a_source_without_dataset_files_before_building_the_model(tmp_path: pathlib.Path) -> None:
    with mock.patch("basic_pitch.models.model") as build_model:
        with pytest.raises(ValueError, match="No tfrecord files found"):
            train.main(
                str(tmp_path),
                "output",
                1,
                1,
                0.001,
                1,
                1,
                1,
                1,
                ["guitarset"],
                np.array([1.0]),
                False,
                False,
                False,
                0.5,
            )
    build_model.assert_not_called()


@pytest.mark.parametrize("requested", range(1, 51))
def test_evaluation_callback_batching_covers_exactly_the_requested_examples(requested: int) -> None:
    batch_size, steps = train.evaluation_callback_batching(requested)

    assert 1 <= batch_size <= train.MAX_EVAL_CBF_BATCH_SIZE
    assert batch_size * steps == requested
