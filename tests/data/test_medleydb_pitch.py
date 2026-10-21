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
import apache_beam as beam
import argparse
import io
import itertools
import numpy as np
import numpy.typing as npt
import os
import pathlib
import pytest
import soundfile as sf
import tensorflow as tf
from typing import Any
from unittest import mock

from apache_beam.testing.test_pipeline import TestPipeline

from basic_pitch.constants import AUDIO_SAMPLE_RATE
from basic_pitch.data import commandline
from basic_pitch.data.datasets import medleydb_pitch
from basic_pitch.data.datasets.medleydb_pitch import (
    MedleyDbPitchInvalidTracks,
    MedleyDbPitchToTfExample,
    create_input_data,
)
from basic_pitch.data.pipeline import WriteBatchToTfRecord

from utils import read_tfrecord_features

TRACK_ID = "AClassicEducation_NightOwl_STEM_08"


def _write_medleydb_pitch_track(source: pathlib.Path) -> None:
    for folder in ("audio", "pitch", "medleydb-pitch-pyin-notes"):
        (source / folder).mkdir(parents=True)

    sample_rate = 44100
    samples = np.arange(2 * sample_rate) / sample_rate
    sf.write(str(source / "audio" / f"{TRACK_ID}.wav"), 0.5 * np.sin(2 * np.pi * 440 * samples), sample_rate)

    # one pitch value in Hz every 11.6 ms between 0.4 s and 1.2 s, 0 where nothing is played
    frames = np.arange(0, 2, 256 / 22050)
    (source / "pitch" / f"{TRACK_ID}.csv").write_text(
        "".join(f"{t},{440.0 if 0.4 <= t < 1.2 else 0.0}\n" for t in frames)
    )
    (source / "medleydb-pitch-pyin-notes" / f"{TRACK_ID}_vamp_pyin_pyin_notes.csv").write_text("0.4,0.8,440.0\n")


def _decode(feature: tf.train.Feature, dtype: tf.DType) -> npt.NDArray[Any]:
    return tf.io.parse_tensor(feature.bytes_list.value[0], dtype).numpy()  # type: ignore[no-any-return]


def test_medleydb_pitch_to_tf_example(tmp_path: pathlib.Path, mock_medleydb_pitch_index: None) -> None:
    source = tmp_path / "source"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_medleydb_pitch_track(source)

    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([[TRACK_ID]])
            | "Create tf.Example" >> beam.ParDo(MedleyDbPitchToTfExample(str(source), download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(str(output_dir)))
        )

    (features,) = read_tfrecord_features(output_dir)
    assert features["file_id"].bytes_list.value[0] == TRACK_ID.encode()
    assert features["source"].bytes_list.value[0] == b"medleydb_pitch"

    audio, sample_rate = sf.read(io.BytesIO(features["audio_wav"].bytes_list.value[0]))
    assert sample_rate == AUDIO_SAMPLE_RATE
    assert audio.ndim == 1

    notes_indices = _decode(features["notes_indices"], tf.int64)
    assert len(notes_indices) > 0
    assert len(_decode(features["notes_values"], tf.float32)) == len(notes_indices)
    assert len(_decode(features["onsets_indices"], tf.int64)) > 0
    assert len(_decode(features["contours_indices"], tf.int64)) > 0
    n_times, n_notes = _decode(features["notes_onsets_shape"], tf.int64)
    assert n_times > 0 and n_notes > 0
    assert notes_indices[:, 0].max() < n_times
    assert notes_indices[:, 1].max() < n_notes


def test_medleydb_pitch_invalid_tracks(tmpdir: str) -> None:
    split_labels = ["train", "validation"]
    input_data = [(str(i), split) for i, split in enumerate(split_labels)]
    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(MedleyDbPitchInvalidTracks()).with_outputs(*split_labels)
        )

        for split in split_labels:
            (
                getattr(splits, split)
                | f"Write {split} to text"
                >> beam.io.WriteToText(os.path.join(tmpdir, f"output_{split}.txt"), shard_name_template="")
            )

    for i, split in enumerate(split_labels):
        with open(os.path.join(tmpdir, f"output_{split}.txt"), "r") as fp:
            assert fp.read().strip() == str(i)


def test_medleydb_create_input_data(mock_medleydb_pitch_index: None) -> None:
    data = create_input_data(train_percent=0.5)
    data.sort(key=lambda el: el[1])  # sort by split
    tolerance = 0.01
    for _, group in itertools.groupby(data, lambda el: el[1]):
        assert (0.5 - tolerance) * len(data) <= len(list(group)) <= (0.5 + tolerance) * len(data)


def test_create_input_data_overallocate() -> None:
    with pytest.raises(ValueError, match="over allocate"):
        create_input_data(train_percent=1.1)


def _known_args(source: pathlib.Path, destination: pathlib.Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commandline.add_default(parser, "medleydb_pitch")
    commandline.add_split(parser)
    known_args, _ = parser.parse_known_args(
        ["--source", str(source), "--destination", str(destination), "--batch-size", "1"]
    )
    return known_args


def test_medleydb_pitch_main_writes_the_splits(tmp_path: pathlib.Path, mock_medleydb_pitch_index: None) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_medleydb_pitch_track(source)

    with mock.patch.object(medleydb_pitch, "create_input_data", return_value=[(TRACK_ID, "validation")]):
        medleydb_pitch.main(_known_args(source, destination), [])

    splits = destination / "splits"
    for split in ("train", "test", "validation"):
        assert (splits / split / "index.csv").is_file()
    assert len(list((splits / "validation").glob("*.tfrecord"))) == 1
    assert not list((splits / "train").glob("*.tfrecord"))
