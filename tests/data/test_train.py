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

import pathlib

import numpy as np
import pytest

from basic_pitch import train

from test_tf_example_deserialization import setup_test_resources


@pytest.mark.parametrize("no_contours", [False, True])
def test_main_trains_and_saves_the_model(tmp_path: pathlib.Path, mock_guitarset_index: None, no_contours: bool) -> None:
    source = setup_test_resources(tmp_path)
    output = tmp_path / "output"

    train.main(
        str(source),
        str(output),
        batch_size=2,
        shuffle_size=2,
        learning_rate=0.001,
        epochs=1,
        steps_per_epoch=1,
        validation_steps=1,
        size_evaluation_callback_datasets=2,
        datasets_to_use=["guitarset"],
        dataset_sampling_frequency=np.array([1.0]),
        no_sonify=True,
        no_contours=no_contours,
        weighted_onset_loss=no_contours,
        positive_onset_weight=0.5,
    )

    (run_dir,) = output.iterdir()
    assert (run_dir / "model.best").is_dir()
    assert (run_dir / "checkpoints" / "model.01").is_dir()
    for stage in ("train", "validation"):
        assert any((run_dir / "tensorboard" / stage).iterdir())
