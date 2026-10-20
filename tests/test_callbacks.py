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

import logging

import numpy as np
import pytest
import tensorflow as tf

from typing import Dict

from basic_pitch.callbacks import VisualizeCallback
from basic_pitch.constants import (
    AUDIO_N_SAMPLES,
    ANNOT_N_FRAMES,
    N_FREQ_BINS_CONTOURS,
    N_FREQ_BINS_NOTES,
)


class MockModel(tf.keras.Model):
    def __init__(self) -> None:
        super(MockModel, self).__init__()

    def call(self, inputs: tf.Tensor) -> Dict[str, tf.Tensor]:
        return {
            "onset": tf.random.normal((1, ANNOT_N_FRAMES, N_FREQ_BINS_NOTES)),
            "note": tf.random.normal((1, ANNOT_N_FRAMES, N_FREQ_BINS_NOTES)),
            "contour": tf.random.normal((1, ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS)),
        }


def create_mock_dataset() -> tf.data.Dataset:
    batch_size = 1
    inputs = tf.random.normal((batch_size, AUDIO_N_SAMPLES, 1))
    targets = {
        "onset": tf.random.normal((batch_size, ANNOT_N_FRAMES, N_FREQ_BINS_NOTES)),
        "note": tf.random.normal((batch_size, ANNOT_N_FRAMES, N_FREQ_BINS_NOTES)),
        "contour": tf.random.normal((batch_size, ANNOT_N_FRAMES, N_FREQ_BINS_CONTOURS)),
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.batch(batch_size)
    return dataset


def test_visualize_callback_on_epoch_end(tmpdir: str) -> None:

    vc = VisualizeCallback(
        train_ds=create_mock_dataset(),
        validation_ds=create_mock_dataset(),
        tensorboard_dir=str(tmpdir),
        sonify=True,
        contours=True,
    )

    vc.model = MockModel()

    vc.on_epoch_end(1, {"loss": np.random.random(), "val_loss": np.random.random()})


def make_callback(tmpdir: str, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset) -> VisualizeCallback:
    vc = VisualizeCallback(
        train_ds=train_ds,
        validation_ds=validation_ds,
        tensorboard_dir=str(tmpdir),
        sonify=False,
        contours=True,
    )
    vc.model = MockModel()
    return vc


def test_visualize_callback_starts_a_dataset_that_ran_out_again(tmpdir: str) -> None:
    # a dataset with one batch and no repeat is empty after the first epoch
    vc = make_callback(tmpdir, create_mock_dataset(), create_mock_dataset())
    logs = {"loss": 0.5, "val_loss": 0.6}

    vc.on_epoch_end(1, logs)
    vc.on_epoch_end(2, logs)
    vc.on_epoch_end(3, logs)


def test_visualize_callback_warns_instead_of_failing_when_a_dataset_has_no_examples(
    tmpdir: str, caplog: pytest.LogCaptureFixture
) -> None:
    empty_ds = create_mock_dataset().take(0)
    vc = make_callback(tmpdir, empty_ds, create_mock_dataset())

    with caplog.at_level(logging.WARNING):
        vc.on_epoch_end(1, {"loss": 0.5, "val_loss": 0.6})

    assert "no visualization is made" in caplog.text
