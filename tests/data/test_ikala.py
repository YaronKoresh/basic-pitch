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
import pathlib
import pytest
import os
import soundfile as sf
import tensorflow as tf
from typing import Any
from unittest import mock
from apache_beam.testing.test_pipeline import TestPipeline

from basic_pitch.constants import AUDIO_SAMPLE_RATE
from basic_pitch.data import commandline
from basic_pitch.data.datasets import ikala
from basic_pitch.data.datasets.ikala import (
    IkalaInvalidTracks,
    IkalaToTfExample,
    create_input_data,
)
from basic_pitch.data.pipeline import WriteBatchToTfRecord

from utils import read_tfrecord_features

TRACK_ID = "10161_chorus"


def _write_ikala_track(source: pathlib.Path) -> None:
    ikala_home = source / "iKala"
    for folder in ("Wavfile", "PitchLabel", "ikala-pyin-notes"):
        (ikala_home / folder).mkdir(parents=True)

    # iKala audio has the instrumental on the left channel and the vocal on the right channel
    sample_rate = 44100
    samples = np.arange(2 * sample_rate) / sample_rate
    vocal = 0.5 * np.sin(2 * np.pi * 440 * samples)
    sf.write(str(ikala_home / "Wavfile" / f"{TRACK_ID}.wav"), np.stack([0 * vocal, vocal], axis=1), sample_rate)

    # one pitch value in MIDI notes per 32 ms frame, 0 where nothing is sung
    pitch = [69.0 if 10 <= frame < 40 else 0.0 for frame in range(62)]
    (ikala_home / "PitchLabel" / f"{TRACK_ID}.pv").write_text("".join(f"{value}\n" for value in pitch))
    (ikala_home / "ikala-pyin-notes" / f"{TRACK_ID}_vamp_pyin_pyin_notes.csv").write_text("0.4,0.8,440.0\n")


def _decode(feature: tf.train.Feature, dtype: tf.DType) -> npt.NDArray[Any]:
    return tf.io.parse_tensor(feature.bytes_list.value[0], dtype).numpy()  # type: ignore[no-any-return]


def test_ikala_to_tf_example(tmp_path: pathlib.Path, mock_ikala_index: None) -> None:
    source = tmp_path / "source"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_ikala_track(source)

    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([[TRACK_ID]])
            | "Create tf.Example" >> beam.ParDo(IkalaToTfExample(str(source), download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(str(output_dir)))
        )

    (features,) = read_tfrecord_features(output_dir)
    assert features["file_id"].bytes_list.value[0] == TRACK_ID.encode()
    assert features["source"].bytes_list.value[0] == b"ikala"

    # only the vocal channel is kept, at full level
    audio, sample_rate = sf.read(io.BytesIO(features["audio_wav"].bytes_list.value[0]))
    assert sample_rate == AUDIO_SAMPLE_RATE
    assert audio.ndim == 1
    assert np.abs(audio).max() == pytest.approx(0.5, abs=0.02)

    notes_indices = _decode(features["notes_indices"], tf.int64)
    assert len(notes_indices) > 0
    assert len(_decode(features["notes_values"], tf.float32)) == len(notes_indices)
    assert len(_decode(features["onsets_indices"], tf.int64)) > 0
    assert len(_decode(features["contours_indices"], tf.int64)) > 0
    n_times, n_notes = _decode(features["notes_onsets_shape"], tf.int64)
    assert n_times > 0 and n_notes > 0
    assert notes_indices[:, 0].max() < n_times
    assert notes_indices[:, 1].max() < n_notes


def test_ikala_invalid_tracks(tmpdir: str) -> None:
    split_labels = ["train", "validation"]
    input_data = [(str(i), split) for i, split in enumerate(split_labels)]
    with TestPipeline() as p:
        splits = (
            p
            | "Create PCollection" >> beam.Create(input_data)
            | "Tag it" >> beam.ParDo(IkalaInvalidTracks()).with_outputs(*split_labels)
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


def test_ikala_create_input_data(mock_ikala_index: None) -> None:
    data = create_input_data(train_percent=0.5)
    data.sort(key=lambda el: el[1])  # sort by split
    tolerance = 0.1
    for _, group in itertools.groupby(data, lambda el: el[1]):
        assert (0.5 - tolerance) * len(data) <= len(list(group)) <= (0.5 + tolerance) * len(data)


def test_ikala_create_input_data_overallocate() -> None:
    with pytest.raises(ValueError, match="over allocate"):
        create_input_data(train_percent=1.1)


def _known_args(source: pathlib.Path, destination: pathlib.Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commandline.add_default(parser, "ikala")
    commandline.add_split(parser)
    known_args, _ = parser.parse_known_args(
        ["--source", str(source), "--destination", str(destination), "--batch-size", "1"]
    )
    return known_args


def test_ikala_main_writes_the_splits(tmp_path: pathlib.Path, mock_ikala_index: None) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _write_ikala_track(source)

    with mock.patch.object(ikala, "create_input_data", return_value=[(TRACK_ID, "train")]):
        ikala.main(_known_args(source, destination), [])

    splits = destination / "splits"
    for split in ("train", "test", "validation"):
        assert (splits / split / "index.csv").is_file()
    assert len(list((splits / "train").glob("*.tfrecord"))) == 1
    assert not list((splits / "validation").glob("*.tfrecord"))
