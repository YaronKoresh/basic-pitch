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
from typing import List
from unittest import mock

import pytest

from basic_pitch.data import download


def run_download(arguments: List[str]) -> mock.MagicMock:
    dataset_main = mock.MagicMock()
    datasets = {name: dataset_main for name in download.DATASET_DICT}
    with mock.patch.object(sys, "argv", ["bp-download"] + arguments), mock.patch.object(
        download, "DATASET_DICT", datasets
    ):
        download.main()
    return dataset_main


@pytest.mark.parametrize("dataset", ["guitarset", "ikala", "medleydb_pitch", "maestro", "slakh"])
def test_default_options_reach_the_dataset_script(dataset: str) -> None:
    run_download([dataset]).assert_called_once()


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["guitarset", "--train-percent", "0.9", "--validation-percent", "0.5"], "over allocate"),
        (["guitarset", "--train-percent", "0"], "train percent must be above 0"),
        (["guitarset", "--validation-percent", "-0.1"], "validation percent must not be negative"),
        (["guitarset", "--train-percent", "nan"], "finite"),
        (["ikala", "--train-percent", "1.1"], "over allocate"),
        (["medleydb_pitch", "--train-percent", "-1"], "train percent must be above 0"),
    ],
)
def test_split_options_that_cannot_make_a_split_exit_with_a_usage_error_before_the_dataset_script(
    arguments: List[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_main = mock.MagicMock()
    datasets = {name: dataset_main for name in download.DATASET_DICT}
    with mock.patch.object(sys, "argv", ["bp-download"] + arguments), mock.patch.object(
        download, "DATASET_DICT", datasets
    ):
        with pytest.raises(SystemExit) as exit_info:
            download.main()
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err
    dataset_main.assert_not_called()


def test_ikala_ignores_the_validation_percent_it_does_not_use() -> None:
    run_download(["ikala", "--train-percent", "0.9", "--validation-percent", "0.5"]).assert_called_once()


def test_datasets_without_a_random_split_ignore_the_split_options() -> None:
    run_download(["maestro", "--train-percent", "5"]).assert_called_once()


def test_dataset_help_lists_the_options_of_the_dataset_script(capsys: pytest.CaptureFixture[str]) -> None:
    dataset_main = mock.MagicMock()
    datasets = {name: dataset_main for name in download.DATASET_DICT}
    with mock.patch.object(sys, "argv", ["bp-download", "guitarset", "--help"]), mock.patch.object(
        download, "DATASET_DICT", datasets
    ):
        with pytest.raises(SystemExit) as exit_info:
            download.main()
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for option in ("--source", "--destination", "--train-percent", "--validation-percent", "--split-seed"):
        assert option in output
    dataset_main.assert_not_called()


def test_help_without_a_dataset_lists_the_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    with mock.patch.object(sys, "argv", ["bp-download", "--help"]):
        with pytest.raises(SystemExit) as exit_info:
            download.main()
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for dataset in download.DATASET_DICT:
        assert dataset in output


def test_missing_dataset_exits_with_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with mock.patch.object(sys, "argv", ["bp-download"]):
        with pytest.raises(SystemExit) as exit_info:
            download.main()
    assert exit_info.value.code == 2
    assert "dataset" in capsys.readouterr().err
