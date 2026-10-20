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

import argparse

import pytest

from basic_pitch.data.commandline import add_default, verify_split_percents


@pytest.mark.parametrize("train, validation", [(0.8, 0.1), (0.5, 0.0), (0.33, 0.33), (0.99, 0.0)])
def test_verify_split_percents_accepts_a_split_with_train_and_test_parts(train: float, validation: float) -> None:
    verify_split_percents(train, validation)


@pytest.mark.parametrize(
    "train, validation, message",
    [
        (1.5, -0.7, "validation percent must not be negative"),
        (-0.2, 0.1, "train percent must be above 0"),
        (0.0, 0.1, "train percent must be above 0"),
        (0.9, 0.1, "over allocate"),
        (1.0, 0.0, "over allocate"),
        (float("nan"), 0.1, "finite"),
        (0.5, float("nan"), "finite"),
        (float("inf"), 0.0, "finite"),
    ],
)
def test_verify_split_percents_rejects_values_that_cannot_make_a_split(
    train: float, validation: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        verify_split_percents(train, validation)


def test_batch_size_option_defaults_to_five_and_accepts_positive_numbers() -> None:
    parser = argparse.ArgumentParser()
    add_default(parser, "x")
    assert parser.parse_args([]).batch_size == 5
    assert parser.parse_args(["--batch-size", "12"]).batch_size == 12


@pytest.mark.parametrize("value", ["0", "-3", "1.5", "abc", ""])
def test_batch_size_option_rejects_values_below_one_and_non_integers(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser()
    add_default(parser, "x")
    with pytest.raises(SystemExit):
        parser.parse_args(["--batch-size", value])
    assert "must be a whole number of at least 1" in capsys.readouterr().err
