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

from typing import Any, Iterator, List, Tuple, cast
import unittest

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from basic_pitch import models, nn
from basic_pitch.constants import (
    ANNOT_N_FRAMES,
    ANNOTATIONS_N_SEMITONES,
    AUDIO_N_SAMPLES,
    N_FREQ_BINS_CONTOURS,
)

BATCH_SIZE = 3

tfkl = tf.keras.layers


class TestHarmonicStacking(unittest.TestCase):
    def _audio_data_gen(self) -> Iterator[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
        while True:
            audio = np.random.uniform(size=(BATCH_SIZE, AUDIO_N_SAMPLES, 1)).astype(np.float32)
            output = np.random.uniform(size=(BATCH_SIZE, ANNOT_N_FRAMES, ANNOTATIONS_N_SEMITONES * 3, 1)).astype(
                np.float32
            )
            yield (audio, output)

    def _dummy_dataset(self) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_generator(
            self._audio_data_gen,
            (tf.float32, tf.float32),
            (
                tf.TensorShape([BATCH_SIZE, AUDIO_N_SAMPLES, 1]),
                tf.TensorShape([BATCH_SIZE, ANNOT_N_FRAMES, ANNOTATIONS_N_SEMITONES * 3, 1]),
            ),
        )
        return ds

    def test_defaults(self) -> None:
        inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))
        x = models.get_cqt(inputs, 5, True)
        conv = nn.HarmonicStacking(3, [1, 2, 3, 4, 5], N_FREQ_BINS_CONTOURS)(x)
        outputs = tfkl.Conv2D(1, (3, 3), padding="same")(conv)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        train_ds = self._dummy_dataset()
        valid_ds = self._dummy_dataset()

        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.1),
        )

        model.fit(
            train_ds,
            epochs=2,
            steps_per_epoch=2,
            validation_data=valid_ds,
            validation_steps=1,
        )

        for element in train_ds:
            in_vals = element[0]
            out_vals = model(in_vals)
            assert out_vals.shape == (
                BATCH_SIZE,
                ANNOT_N_FRAMES,
                ANNOTATIONS_N_SEMITONES * 3,
                1,
            )
            break

    def test_fractions(self) -> None:
        inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))
        x = models.get_cqt(inputs, 2, True)
        conv = nn.HarmonicStacking(3, [0.5, 1, 2], N_FREQ_BINS_CONTOURS)(x)
        outputs = tfkl.Conv2D(1, (3, 3), padding="same")(conv)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        train_ds = self._dummy_dataset()
        valid_ds = self._dummy_dataset()

        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.1),
        )

        model.fit(
            train_ds,
            epochs=2,
            steps_per_epoch=2,
            validation_data=valid_ds,
            validation_steps=1,
        )

        for element in train_ds:
            inputs = element[0]
            output = model(inputs)
            assert output.shape == (
                BATCH_SIZE,
                ANNOT_N_FRAMES,
                ANNOTATIONS_N_SEMITONES * 3,
                1,
            )
            break


class TestHarmonicStackingEdgeCases(unittest.TestCase):
    def test_shifts_keep_the_values_that_stay_inside_the_frequency_axis(self) -> None:
        x = np.arange(1, 13, dtype=np.float32).reshape(1, 1, 12, 1)
        output = nn.HarmonicStacking(1, [0.5, 1, 2], 12)(x).numpy()[0, 0]  # shifts are -12, 0 and 12
        np.testing.assert_array_equal(output[:, 1], np.arange(1, 13))
        np.testing.assert_array_equal(output[:, 0], np.zeros(12))
        np.testing.assert_array_equal(output[:, 2], np.zeros(12))

    def test_partial_shift_moves_the_values(self) -> None:
        x = np.arange(1, 13, dtype=np.float32).reshape(1, 1, 12, 1)
        output = nn.HarmonicStacking(1, [1, 2], 12)(x).numpy()[0, 0]  # the second shift is +12
        self.assertEqual(output.shape, (12, 2))
        output = nn.HarmonicStacking(1, [2 ** (3 / 12)], 12)(x).numpy()[0, 0, :, 0]  # shift +3
        np.testing.assert_array_equal(output, np.concatenate([np.arange(4, 13), np.zeros(3)]))
        output = nn.HarmonicStacking(1, [2 ** (-3 / 12)], 12)(x).numpy()[0, 0, :, 0]  # shift -3
        np.testing.assert_array_equal(output, np.concatenate([np.zeros(3), np.arange(1, 10)]))

    def test_shift_longer_than_the_frequency_axis_gives_zeros(self) -> None:
        x = tf.ones((1, 4, 12, 1))
        output = nn.HarmonicStacking(3, [0.5, 1, 2], 10)(x)
        self.assertEqual(output.shape, (1, 4, 10, 3))
        np.testing.assert_array_equal(output.numpy()[..., 0], np.zeros((1, 4, 10)))
        np.testing.assert_array_equal(output.numpy()[..., 1], np.ones((1, 4, 10)))

    def test_invalid_harmonics_raise_value_error(self) -> None:
        cases: List[List[float]] = [[0], [-1.0], [float("nan")], [float("inf")], [1, 0]]
        for harmonics in cases:
            with self.subTest(harmonics=harmonics):
                with self.assertRaisesRegex(ValueError, "harmonics must be finite numbers above 0"):
                    nn.HarmonicStacking(3, harmonics, 10)

    def test_invalid_harmonic_stacking_options_raise_value_error(self) -> None:
        cases: List[Tuple[Any, List[Any], Any, str]] = [
            (3, [], 10, "At least one harmonic"),
            (3, [True], 10, "harmonics must be finite numbers above 0"),
            (3, ["a"], 10, "harmonics must be finite numbers above 0"),
            (3, [1, 2], 0, "n_output_freqs must be a whole number of at least 1"),
            (3, [1, 2], -3, "n_output_freqs must be a whole number of at least 1"),
            (3, [1, 2], 2.5, "n_output_freqs must be a whole number of at least 1"),
            (0, [1, 2], 10, "bins_per_semitone must be a whole number of at least 1"),
            (-3, [1, 2], 10, "bins_per_semitone must be a whole number of at least 1"),
            (2.5, [1, 2], 10, "bins_per_semitone must be a whole number of at least 1"),
            (True, [1, 2], 10, "bins_per_semitone must be a whole number of at least 1"),
        ]
        for bins_per_semitone, harmonics, n_output_freqs, message in cases:
            with self.subTest(bins=bins_per_semitone, harmonics=harmonics, outputs=n_output_freqs):
                with self.assertRaisesRegex(ValueError, message):
                    nn.HarmonicStacking(bins_per_semitone, harmonics, n_output_freqs)

    def test_numpy_integers_are_accepted_as_stacking_sizes(self) -> None:
        layer = nn.HarmonicStacking(*cast(Any, (np.int64(3), [0.5, 1, np.float32(2)], np.int32(10))))
        self.assertEqual(layer.n_output_freqs, 10)


class TestLossFunctions(unittest.TestCase):
    def test_weighted_loss_is_finite_when_a_batch_has_only_one_kind_of_label(self) -> None:
        rng = np.random.default_rng(0)
        y_pred = tf.constant(rng.uniform(0.05, 0.95, (2, 5, 8)).astype(np.float32))
        for y_true in (np.zeros((2, 5, 8), np.float32), np.ones((2, 5, 8), np.float32)):
            with tf.GradientTape() as tape:
                tape.watch(y_pred)
                loss = models.weighted_transcription_loss(tf.constant(y_true), y_pred, 0.2, 0.5)
            self.assertTrue(np.all(np.isfinite(loss.numpy())))
            self.assertTrue(np.all(np.isfinite(tape.gradient(loss, y_pred).numpy())))

    def test_weighted_loss_with_both_kinds_of_label_is_the_weighted_mean_of_each_kind(self) -> None:
        y_true = tf.constant([[1.0, 0.0, 0.0, 1.0]])
        y_pred = tf.constant([[0.9, 0.2, 0.4, 0.6]])
        expected = 0.25 * np.mean(-np.log(1 - np.array([0.2, 0.4]))) + 0.75 * np.mean(-np.log(np.array([0.9, 0.6])))
        loss = models.weighted_transcription_loss(y_true, y_pred, 0.0, 0.75)
        self.assertAlmostEqual(float(loss[0]), float(expected), places=5)

    def test_weighted_loss_gives_one_value_for_each_example(self) -> None:
        rng = np.random.default_rng(1)
        y_true = tf.constant((rng.random((3, 5, 8)) > 0.7).astype(np.float32))
        y_pred = tf.constant(rng.uniform(0.05, 0.95, (3, 5, 8)).astype(np.float32))
        loss = models.weighted_transcription_loss(y_true, y_pred, 0.1, 0.6)
        self.assertEqual(loss.shape, (3,))
        for i in range(3):
            single = models.weighted_transcription_loss(y_true[i : i + 1], y_pred[i : i + 1], 0.1, 0.6)
            self.assertAlmostEqual(float(loss[i]), float(single[0]), places=5)

    def test_model_trains_with_the_weighted_loss_and_sample_weights_on_a_batch(self) -> None:
        inputs = tf.keras.Input(shape=(3,))
        outputs = {"onset": tf.keras.layers.Reshape((5, 8))(tf.keras.layers.Dense(40, activation="sigmoid")(inputs))}
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss={"onset": models.onset_loss(True, 0.1, 0.7)}, optimizer="adam")
        rng = np.random.default_rng(2)
        x = rng.random((4, 3)).astype(np.float32)
        y = {"onset": (rng.random((4, 5, 8)) > 0.7).astype(np.float32)}
        weights = {"onset": np.array([1.0, 0.0, 1.0, 1.0], np.float32)}
        history = model.fit(x, y, sample_weight=weights, batch_size=4, epochs=1, verbose=0)
        self.assertTrue(np.isfinite(history.history["loss"][0]))

    def test_loss_has_all_keys(self) -> None:
        result = models.loss()
        assert set(result.keys()) == {"contour", "note", "onset"}

    def test_loss_no_contour_excludes_contour(self) -> None:
        result = models.loss_no_contour()
        assert set(result.keys()) == {"note", "onset"}
        assert "contour" not in result

    def test_loss_no_contour_weighted(self) -> None:
        result = models.loss_no_contour(weighted=True, positive_weight=0.7)
        assert set(result.keys()) == {"note", "onset"}


class TestModelNoContours(unittest.TestCase):
    def test_model_no_contours_compiles_with_loss_no_contour(self) -> None:
        m = models.model(no_contours=True)
        loss = models.loss_no_contour()
        m.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(0.001))


class TestModelOptionValidation(unittest.TestCase):
    def test_invalid_n_harmonics_raise_value_error(self) -> None:
        for n_harmonics in (0, -2, 2.5, True, float("nan")):
            with self.subTest(n_harmonics=n_harmonics):
                with self.assertRaisesRegex(ValueError, "n_harmonics must be a whole number of at least 1"):
                    models.model(n_harmonics=n_harmonics)  # type: ignore
                with self.assertRaisesRegex(ValueError, "n_harmonics must be a whole number of at least 1"):
                    inputs = tf.keras.Input(shape=(AUDIO_N_SAMPLES, 1))
                    models.get_cqt(inputs, n_harmonics, False)  # type: ignore

    def test_invalid_filter_counts_raise_value_error(self) -> None:
        for name in ("n_filters_contour", "n_filters_onsets", "n_filters_notes"):
            for count in (0, -4, 2.5, True, float("nan")):
                with self.subTest(name=name, count=count):
                    with self.assertRaisesRegex(ValueError, f"{name} must be a whole number of at least 1"):
                        models.model(**{name: count})  # type: ignore

    def test_invalid_loss_options_raise_value_error(self) -> None:
        for factory in (models.loss, models.loss_no_contour):
            for options in (
                {"label_smoothing": 2.0},
                {"label_smoothing": -0.1},
                {"label_smoothing": float("nan")},
                {"weighted": True, "positive_weight": 2.0},
                {"weighted": True, "positive_weight": float("inf")},
            ):
                with self.subTest(factory=factory.__name__, options=options):
                    with self.assertRaisesRegex(ValueError, "must be a number between 0 and 1"):
                        factory(**options)  # type: ignore

    def test_default_and_edge_loss_options_are_accepted(self) -> None:
        self.assertIn("onset", models.loss())
        self.assertIn("onset", models.loss_no_contour(label_smoothing=0.0, weighted=True, positive_weight=1.0))
