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
import argparse
import os
import pathlib
from typing import Any, Dict, List
from unittest import mock

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline

from basic_pitch.data import commandline, pipeline
from basic_pitch.data.datasets import maestro
from basic_pitch.data.datasets.maestro import (
    MaestroToTfExample,
    MaestroInvalidTracks,
    create_input_data,
)
from basic_pitch.data.pipeline import WriteBatchToTfRecord

from utils import create_mock_wav, create_mock_midi

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
MAESTRO_TEST_DATA_PATH = RESOURCES_PATH / "data" / "maestro"

TRAIN_TRACK_ID = "2004/MIDI-Unprocessed_SMF_05_R1_2004_01_ORIG_MID--AUDIO_05_R1_2004_03_Track03_wav"
VALID_TRACK_ID = "2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav"
TEST_TRACK_ID = "2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_08_Track08_wav"
GT_15M_TRACK_ID = "2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav"


def test_maestro_to_tf_example(tmp_path: pathlib.Path, mock_maestro_index: None) -> None:
    mock_maestro_home = tmp_path / "maestro"
    mock_maestro_ext = mock_maestro_home / "2004"
    mock_maestro_ext.mkdir(parents=True, exist_ok=True)

    create_mock_wav(mock_maestro_ext / f"{TRAIN_TRACK_ID.split('/')[1]}.wav", 3)
    create_mock_midi(mock_maestro_ext / f"{TRAIN_TRACK_ID.split('/')[1]}.midi")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_data: List[str] = [TRAIN_TRACK_ID]
    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([input_data])
            | "Create tf.Example" >> beam.ParDo(MaestroToTfExample(str(mock_maestro_home), download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(str(output_dir)))
        )

    listdir = os.listdir(str(output_dir))
    assert len(listdir) == 1
    assert os.path.splitext(listdir[0])[-1] == ".tfrecord"
    with open(os.path.join(str(output_dir), os.listdir(str(output_dir))[0]), "rb") as fp:
        data = fp.read()
        assert len(data) != 0


def test_maestro_invalid_tracks(tmp_path: pathlib.Path, mock_maestro_index: None) -> None:
    mock_maestro_home = tmp_path / "maestro"
    mock_maestro_ext = mock_maestro_home / "2004"
    mock_maestro_ext.mkdir(parents=True, exist_ok=True)

    input_data = [(TRAIN_TRACK_ID, "train"), (VALID_TRACK_ID, "validation"), (TEST_TRACK_ID, "test")]

    for track_id, _ in input_data:
        create_mock_wav(mock_maestro_ext / f"{track_id.split('/')[1]}.wav", 3)

    split_labels = set([e[1] for e in input_data])
    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(MaestroInvalidTracks(str(mock_maestro_home))).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(str(tmp_path / f"output_{split}.txt"), shard_name_template="")
            )

    for track_id, split in input_data:
        with open(tmp_path / f"output_{split}.txt", "r") as fp:
            assert fp.read().strip() == track_id


def test_maestro_invalid_tracks_over_15_min(tmp_path: pathlib.Path, mock_maestro_index: None) -> None:
    """
    The track id used here is a real track id in maestro, and it is part of the train split, but we mock the data so as
    not to store a large file in git, hence the variable name.
    """

    mock_maestro_home = tmp_path / "maestro"
    mock_maestro_ext = mock_maestro_home / "2004"
    mock_maestro_ext.mkdir(parents=True, exist_ok=True)

    create_mock_wav(mock_maestro_ext / f"{GT_15M_TRACK_ID.split('/')[1]}.wav", 16)

    input_data = [(GT_15M_TRACK_ID, "train")]
    split_labels = set([e[1] for e in input_data])
    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(MaestroInvalidTracks(str(mock_maestro_home))).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(str(tmp_path / f"output_{split}.txt"), shard_name_template="")
            )

    for _, split in input_data:
        with open(tmp_path / f"output_{split}.txt", "r") as fp:
            assert fp.read().strip() == ""


def test_maestro_create_input_data(mock_maestro_index: None) -> None:
    """
    A commuted metadata file is included in the repo for testing. mirdata references the metadata file to
    populate the tracklist with metadata. Since the file is commuted to only the filenames referenced here,
    we only consider these when testing the metadata.
    """
    data = create_input_data()
    assert len(data)

    test_fnames = {TRAIN_TRACK_ID, VALID_TRACK_ID, TEST_TRACK_ID, GT_15M_TRACK_ID}
    splits = {d[1] for d in data if d[0].split(".")[0] in test_fnames}
    assert splits == set(["train", "validation", "test"])


def _maestro_known_args(source: pathlib.Path, destination: pathlib.Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commandline.add_default(parser, "maestro")
    commandline.add_split(parser)
    known_args, _ = parser.parse_known_args(
        ["--source", str(source), "--destination", str(destination), "--batch-size", "1"]
    )
    return known_args


def test_maestro_main_does_not_ask_for_a_docker_environment(tmp_path: pathlib.Path, mock_maestro_index: None) -> None:
    """The default DirectRunner runs in this process, so the options must not ask Beam to start a Docker container."""
    captured: Dict[str, Any] = {}

    def fake_run(pipeline_options: Dict[str, Any], *args: Any) -> None:
        captured.update(pipeline_options)

    with mock.patch.object(pipeline, "run", side_effect=fake_run):
        maestro.main(_maestro_known_args(tmp_path / "source", tmp_path / "destination"), [])

    assert captured["runner"] == "DirectRunner"
    assert "environment_type" not in captured


def test_maestro_main_writes_the_splits(tmp_path: pathlib.Path, mock_maestro_index: None) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "2004").mkdir(parents=True)
    track_name = TRAIN_TRACK_ID.split("/")[1]
    create_mock_wav(source / "2004" / f"{track_name}.wav", 1)
    create_mock_midi(source / "2004" / f"{track_name}.midi")

    # the mock index lists more tracks than the one that has files here
    with mock.patch.object(maestro, "create_input_data", return_value=[(TRAIN_TRACK_ID, "train")]):
        maestro.main(_maestro_known_args(source, destination), [])

    splits = destination / "splits"
    for split in ("train", "test", "validation"):
        assert (splits / split / "index.csv").is_file()
    assert len(list((splits / "train").glob("*.tfrecord"))) == 1
    assert not list((splits / "test").glob("*.tfrecord"))
    assert not list((splits / "validation").glob("*.tfrecord"))
