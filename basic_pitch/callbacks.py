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
from typing import Any, Dict, Iterator, Optional, Tuple

import tensorflow as tf

from basic_pitch import visualize


class VisualizeCallback(tf.keras.callbacks.Callback):
    # TODO RACHEL make this WAY faster
    """
    Callback to run during training to create tensorboard visualizations per epoch.

        Attributes:
            train_ds: training dataset to use for prediction / visualization / sonification / summarization
            valid_ds: validation dataset to use for "" ""
            tensorboard_dir: directory to output "" ""
            sonify: whether to include sonifications in tensorboard
            contours: whether to plot note contours in tensorboard
    """

    def __init__(
        self,
        train_ds: tf.data.Dataset,
        validation_ds: tf.data.Dataset,
        tensorboard_dir: str,
        sonify: bool,
        contours: bool,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.validation_ds = validation_ds
        self.train_iter = iter(train_ds)
        self.validation_iter = iter(validation_ds)
        self.file_writer = tf.summary.create_file_writer(tensorboard_dir)
        self.sonify = sonify
        self.contours = contours

    @staticmethod
    def _next_batch(dataset: tf.data.Dataset, iterator: Iterator[Any]) -> Tuple[Optional[Any], Iterator[Any]]:
        """Return the next batch and the iterator that follows it.

        A dataset that runs out (for example when every sampled chunk of a small dataset was silent and filtered
        out) is started again once. The batch is None when it has nothing to give even then.
        """
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(dataset)
        try:
            return next(iterator), iterator
        except StopIteration:
            return None, iterator

    def on_epoch_end(self, epoch: int, logs: Dict[Any, Any]) -> None:
        train_batch, self.train_iter = self._next_batch(self.train_ds, self.train_iter)
        validation_batch, self.validation_iter = self._next_batch(self.validation_ds, self.validation_iter)
        if train_batch is None or validation_batch is None:
            logging.warning("The visualization datasets gave no examples, so no visualization is made for this epoch.")
            return
        # the first two outputs of generator needs to be the input and the targets
        train_inputs, train_targets = train_batch[:2]
        validation_inputs, validation_targets = validation_batch[:2]
        for stage, inputs, targets, loss in [
            ("train", train_inputs, train_targets, logs["loss"]),
            ("validation", validation_inputs, validation_targets, logs["val_loss"]),
        ]:
            outputs = self.model.predict(inputs)
            visualize.visualize_transcription(
                self.file_writer,
                stage,
                inputs,
                targets,
                outputs,
                loss,
                epoch,
                sonify=self.sonify,
                contours=self.contours,
            )
