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
import os
from typing import Any, Callable, List, Optional
from unittest import mock

import pytest

from basic_pitch.data.datasets import guitarset, ikala, maestro, medleydb_pitch, slakh


def fake_dataset() -> mock.MagicMock:
    dataset = mock.MagicMock()
    dataset.track_ids = ["a", "b", "c", "d"]
    track = mock.MagicMock(split="train", data_split="train")
    dataset.load_tracks.return_value = {"a": track, "b": track}
    return dataset


CASES = [
    ("guitarset", lambda source: guitarset.create_input_data(0.5, 0.25, None, source), lambda source: source),
    ("ikala", lambda source: ikala.create_input_data(0.5, None, source), lambda source: os.path.join(source, "iKala")),
    ("medleydb_pitch", lambda source: medleydb_pitch.create_input_data(0.5, None, source), lambda source: source),
    ("maestro", lambda source: maestro.create_input_data(source), lambda source: source),
    ("slakh", lambda source: slakh.create_input_data(source), lambda source: source),
]


@pytest.mark.parametrize("name, create, expected_home", CASES)
def test_create_input_data_reads_the_index_from_the_source_directory(
    name: str, create: Callable[[Optional[str]], List[Any]], expected_home: Callable[[str], str]
) -> None:
    with mock.patch("mirdata.initialize", return_value=fake_dataset()) as initialize:
        create("/custom/source")
    initialize.assert_called_once_with(name, data_home=expected_home("/custom/source"))


@pytest.mark.parametrize("name, create, expected_home", CASES)
def test_create_input_data_uses_the_mirdata_default_directory_without_a_source(
    name: str, create: Callable[[Optional[str]], List[Any]], expected_home: Callable[[str], str]
) -> None:
    with mock.patch("mirdata.initialize", return_value=fake_dataset()) as initialize:
        create(None)
    initialize.assert_called_once_with(name, data_home=None)


@pytest.mark.parametrize("module", [guitarset, ikala, medleydb_pitch, maestro, slakh])
def test_main_passes_the_source_option_to_create_input_data(module: Any) -> None:
    known_args = argparse.Namespace(
        source="/custom/source",
        destination="/custom/destination",
        timestamped=False,
        runner="DirectRunner",
        sdk_container_image="",
        job_endpoint="embed",
        batch_size=5,
        train_percent=0.5,
        validation_percent=0.25,
        split_seed=None,
    )
    with mock.patch.object(module, "create_input_data", return_value=[]) as create, mock.patch.object(
        module.pipeline, "run"
    ):
        module.main(known_args, [])
    assert create.call_args.args[-1] == "/custom/source"
