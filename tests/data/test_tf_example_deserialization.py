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
import atexit
import numpy as np
import os
import pathlib
import pytest
import shutil
import tensorflow as tf

from apache_beam.testing.test_pipeline import TestPipeline
from typing import Any, List

from basic_pitch.constants import Split
from basic_pitch.data.datasets.guitarset import GuitarSetToTfExample
from basic_pitch.data.pipeline import WriteBatchToTfRecord
from basic_pitch.data.tf_example_deserialization import (
    prepare_datasets,
    prepare_visualization_datasets,
    sample_datasets,
    transcription_file_generator,
    validation_cache_path,
    verify_dataset_files,
)

from utils import create_mock_wav

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
TRAIN_TRACK_ID = "00_BN1-129-Eb_comp"
VALID_TRACK_ID = "00_BN1-147-Gb_comp"


def create_empty_tfrecord(filepath: pathlib.Path) -> None:
    assert filepath.suffix == ".tfrecord"
    with tf.io.TFRecordWriter(str(filepath)) as writer:
        writer.write("")


def create_tfrecord(input_data: List[str], dataset_home: str, output_dir: str) -> None:
    with TestPipeline() as p:
        (
            p
            | "Create PCollection of track IDs" >> beam.Create([input_data])
            | "Create tf.Example" >> beam.ParDo(GuitarSetToTfExample(dataset_home, download=False))
            | "Write to tfrecord" >> beam.ParDo(WriteBatchToTfRecord(output_dir))
        )


def setup_test_resources(tmp_path: pathlib.Path) -> pathlib.Path:
    mock_guitarset_home = tmp_path / "guitarset"
    mock_guitarset_audio = mock_guitarset_home / "audio_mono-mic"
    mock_guitarset_annotations = mock_guitarset_home / "annotation"

    mock_guitarset_audio.mkdir(parents=True)
    mock_guitarset_annotations.mkdir(parents=True)

    output_home = tmp_path / "data" / "basic_pitch"
    output_splits_dir = output_home / "guitarset" / "splits"

    def mock_and_process(split: str, track_id: str) -> None:
        create_mock_wav(mock_guitarset_audio / f"{track_id}_mic.wav", duration_min=1)
        shutil.copy(
            RESOURCES_PATH / "data" / "guitarset" / "annotation" / f"{track_id}.jams",
            mock_guitarset_annotations / f"{track_id}.jams",
        )

        output_dir = output_splits_dir / split
        output_dir.mkdir(parents=True)

        create_tfrecord(input_data=[track_id], dataset_home=str(mock_guitarset_home), output_dir=str(output_dir))

    mock_and_process("train", TRAIN_TRACK_ID)
    mock_and_process("validation", VALID_TRACK_ID)

    return output_home


def test_prepare_datasets(tmp_path: pathlib.Path, mock_guitarset_index: None) -> None:
    datasets_home = setup_test_resources(tmp_path)

    ds_train, ds_valid = prepare_datasets(
        datasets_base_path=str(datasets_home),
        training_shuffle_buffer_size=1,
        batch_size=1,
        validation_steps=1,
        datasets_to_use=["guitarset"],
        dataset_sampling_frequency=np.array([1]),
    )

    assert ds_train is not None and isinstance(ds_train, tf.data.Dataset)
    assert ds_valid is not None and isinstance(ds_valid, tf.data.Dataset)


def test_prepare_visualization_dataset(tmp_path: pathlib.Path, mock_guitarset_index: None) -> None:
    datasets_home = setup_test_resources(tmp_path)

    ds_train, ds_valid = prepare_visualization_datasets(
        datasets_base_path=str(datasets_home),
        batch_size=1,
        validation_steps=1,
        datasets_to_use=["guitarset"],
        dataset_sampling_frequency=np.array([1]),
    )

    assert ds_train is not None and isinstance(ds_train, tf.data.Dataset)
    assert ds_valid is not None and isinstance(ds_valid, tf.data.Dataset)


def test_sample_datasets(tmp_path: pathlib.Path, mock_guitarset_index: None) -> None:
    """touches the following methods:
    - transcription_dataset
        - parse_transcription_tfexample
        - is_not_bad_shape
        - sparse2dense
        - reduce_transcription_inputs
        - get_sample_weights
            - _infer_time_size
            - _infer_time_size
        - get_transcription_chunks
            - extract_random_window
                - extract_window
                    - trim_time
        - is_not_all_silent_annotations
        - to_transcription_training_input
    """
    datasets_home = setup_test_resources(tmp_path)

    ds = sample_datasets(
        split=Split.train,
        datasets_base_path=str(datasets_home),
        datasets=["guitarset"],
        dataset_sampling_frequency=np.array([1]),
        n_shuffle=1,
        n_samples_per_track=1,
        pairs=True,
    )

    assert ds is not None and isinstance(ds, tf.data.Dataset)


def test_transcription_file_generator_train(tmp_path: pathlib.Path) -> None:
    dataset_path = tmp_path / "test_ds" / "splits" / Split.train.name
    dataset_path.mkdir(parents=True)
    create_empty_tfrecord(dataset_path / "test.tfrecord")

    file_gen, random_seed = transcription_file_generator(
        Split.train, ["test_ds"], datasets_base_path=str(tmp_path), sample_weights=np.array([1])
    )

    assert random_seed is False

    generator = file_gen()
    assert next(generator).numpy().decode("utf-8") == str(dataset_path / "test.tfrecord")
    try:
        next(generator)
    except Exception as e:
        assert isinstance(e, StopIteration)


def test_transcription_file_generator_valid(tmp_path: pathlib.Path) -> None:
    dataset_path = tmp_path / "test_ds" / "splits" / Split.validation.name
    dataset_path.mkdir(parents=True)
    create_empty_tfrecord(dataset_path / "test.tfrecord")

    file_gen, random_seed = transcription_file_generator(
        Split.validation, ["test_ds"], datasets_base_path=str(tmp_path), sample_weights=np.array([1])
    )

    assert random_seed is True

    generator = file_gen()
    assert next(generator).numpy().decode("utf-8") == str(dataset_path / "test.tfrecord")
    try:
        next(generator)
    except Exception as e:
        assert isinstance(e, StopIteration)


def test_validation_sets_leave_no_cache_files_in_the_working_directory(
    tmp_path: pathlib.Path, mock_guitarset_index: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets_home = setup_test_resources(tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    for ds_valid in (
        prepare_datasets(str(datasets_home), 1, 1, 1, ["guitarset"], np.array([1]))[1],
        prepare_visualization_datasets(str(datasets_home), 1, 1, ["guitarset"], np.array([1]))[1],
    ):
        iterator = iter(ds_valid)
        first = next(iterator)
        second = next(iterator)  # the second pass reads the cached batch
        np.testing.assert_array_equal(first[0].numpy(), second[0].numpy())

    assert list(work_dir.iterdir()) == []


def test_validation_cache_paths_are_unique_and_removed_at_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: List[Any] = []
    monkeypatch.setattr(
        atexit, "register", lambda function, *args, **kwargs: registered.append((function, args, kwargs))
    )

    first, second = validation_cache_path(), validation_cache_path()

    assert first != second
    assert os.path.isdir(os.path.dirname(first))
    for function, args, kwargs in registered:
        function(*args, **kwargs)
    assert not os.path.exists(os.path.dirname(first))
    assert not os.path.exists(os.path.dirname(second))


@pytest.mark.parametrize(
    "batch_size, validation_steps, shuffle_size, name",
    [
        (0, 1, 1, "batch_size"),
        (-2, 1, 1, "batch_size"),
        (1.5, 1, 1, "batch_size"),
        (1, 0, 1, "validation_steps"),
        (1, None, 1, "validation_steps"),
        (1, 1, None, "training_shuffle_buffer_size"),
        (1, 1, -1, "training_shuffle_buffer_size"),
    ],
)
def test_prepare_datasets_rejects_options_it_cannot_use_with_a_value_error(
    tmp_path: pathlib.Path, batch_size: Any, validation_steps: Any, shuffle_size: Any, name: str
) -> None:
    with pytest.raises(ValueError, match=name):
        prepare_datasets(str(tmp_path), shuffle_size, batch_size, validation_steps, ["guitarset"], np.array([1.0]))


@pytest.mark.parametrize("batch_size, validation_steps, name", [(0, 1, "batch_size"), (1, 0, "validation_steps")])
def test_prepare_visualization_datasets_rejects_options_it_cannot_use_with_a_value_error(
    tmp_path: pathlib.Path, batch_size: int, validation_steps: int, name: str
) -> None:
    with pytest.raises(ValueError, match=name):
        prepare_visualization_datasets(str(tmp_path), batch_size, validation_steps, ["guitarset"], np.array([1.0]))


def test_verify_dataset_files_names_the_missing_split(tmp_path: pathlib.Path) -> None:
    train_dir = tmp_path / "guitarset" / "splits" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "a.tfrecord").write_bytes(b"")
    with pytest.raises(ValueError, match="validation split of guitarset"):
        verify_dataset_files(str(tmp_path), ["guitarset"])
    validation_dir = tmp_path / "guitarset" / "splits" / "validation"
    validation_dir.mkdir(parents=True)
    (validation_dir / "b.tfrecord").write_bytes(b"")
    verify_dataset_files(str(tmp_path), ["guitarset"])
    with pytest.raises(ValueError, match="train split of maestro"):
        verify_dataset_files(str(tmp_path), ["guitarset", "maestro"])
