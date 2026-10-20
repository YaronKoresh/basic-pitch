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

import csv
import faulthandler
import json
import os
import pathlib
import tempfile
from typing import Dict, List
from unittest import mock

import librosa
import pretty_midi
import numpy as np
import numpy.typing as npt
import pytest

from basic_pitch import ICASSP_2022_MODEL_PATH, inference, note_creation
from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    ANNOTATIONS_N_SEMITONES,
    FFT_HOP,
    ANNOTATION_HOP,
)

RESOURCES_PATH = pathlib.Path(__file__).parent / "resources"

faulthandler.enable()


def test_predict() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    model_output, midi_data, note_events = inference.predict(
        test_audio_path,
        inference.Model(ICASSP_2022_MODEL_PATH),
    )
    assert set(model_output.keys()) == set(["note", "onset", "contour"])
    assert model_output["note"].shape == model_output["onset"].shape
    assert isinstance(midi_data, pretty_midi.PrettyMIDI)
    lowest_supported_midi = 21
    note_pitch_min = [n[2] >= lowest_supported_midi for n in note_events]
    note_pitch_max = [n[2] <= lowest_supported_midi + ANNOTATIONS_N_SEMITONES for n in note_events]
    assert all(note_pitch_min)
    assert all(note_pitch_max)
    assert isinstance(note_events, list)

    # Check that model output has the expected length according to the last frame second computed downstream
    # (via model_frames_to_time) with to a few frames of tolerance
    audio_samples, audio_sr = librosa.load(str(test_audio_path), sr=None, mono=True)
    audio_length_s = len(audio_samples) / audio_sr
    n_model_output_frames = model_output["note"].shape[0]
    last_frame_s = note_creation.model_frames_to_time(n_model_output_frames)[-1]
    np.testing.assert_allclose(last_frame_s, audio_length_s, atol=2 * ANNOTATION_HOP)

    expected_model_output = np.load(RESOURCES_PATH / "vocadito_10" / "model_output.npz", allow_pickle=True)[
        "arr_0"
    ].item()
    for k in expected_model_output.keys():
        np.testing.assert_allclose(expected_model_output[k], model_output[k], atol=1e-4, rtol=0)

    expected_note_events = np.load(RESOURCES_PATH / "vocadito_10" / "note_events.npz", allow_pickle=True)["arr_0"]
    assert len(expected_note_events) == len(note_events)
    # the stored events come from before note events were sorted by start time
    expected_note_events = sorted(expected_note_events, key=lambda e: (e[0], e[2], e[1]))
    for expected, calculated in zip(expected_note_events, note_events):
        for i in range(len(expected)):
            np.testing.assert_allclose(expected[i], calculated[i], atol=1e-4, rtol=0)


def test_predict_with_saves() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        inference.predict_and_save(
            [test_audio_path],
            tmpdir,
            True,
            True,
            True,
            True,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
        expected_midi_path = tmpdir / pathlib.Path("vocadito_10_basic_pitch.mid")
        expected_csv_path = tmpdir / pathlib.Path("vocadito_10_basic_pitch.csv")
        expected_npz_path = tmpdir / pathlib.Path("vocadito_10_basic_pitch.npz")
        expected_sonif_path = tmpdir / pathlib.Path("vocadito_10_basic_pitch.wav")

        for output_path in [
            expected_midi_path,
            expected_csv_path,
            expected_npz_path,
            expected_sonif_path,
        ]:
            assert os.path.exists(output_path)


def test_predict_and_save_skips_audio_with_existing_output_and_continues() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        first = pathlib.Path(tmpdir) / "first.wav"
        second = pathlib.Path(tmpdir) / "second.wav"
        output_dir = pathlib.Path(tmpdir) / "out"
        output_dir.mkdir()
        for audio in (first, second):
            audio.write_bytes(test_audio_path.read_bytes())
        existing_midi = output_dir / "first_basic_pitch.mid"
        existing_midi.write_bytes(b"keep")

        inference.predict_and_save(
            [first, second],
            output_dir,
            True,
            False,
            True,
            True,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )

        assert existing_midi.read_bytes() == b"keep"
        # nothing else is written for the skipped audio
        assert not (output_dir / "first_basic_pitch.npz").exists()
        assert not (output_dir / "first_basic_pitch.csv").exists()
        # the following audio is still processed
        for suffix in ("mid", "npz", "csv"):
            assert (output_dir / f"second_basic_pitch.{suffix}").exists()


def test_predict_and_save_continues_after_unreadable_audio_and_reports_it() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bad = pathlib.Path(tmpdir) / "bad.wav"
        good = pathlib.Path(tmpdir) / "good.wav"
        output_dir = pathlib.Path(tmpdir) / "out"
        output_dir.mkdir()
        bad.write_bytes(b"not audio")
        good.write_bytes((RESOURCES_PATH / "vocadito_10.wav").read_bytes())

        with pytest.raises(IOError, match="1 of 2 audio files could not be processed.*bad.wav"):
            inference.predict_and_save(
                [bad, good],
                output_dir,
                True,
                False,
                False,
                False,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
            )

        assert not (output_dir / "bad_basic_pitch.mid").exists()
        assert (output_dir / "good_basic_pitch.mid").exists()


def test_predict_and_save_names_the_error_when_the_reason_for_skipping_audio_is_blank(
    capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with mock.patch.object(inference, "predict", side_effect=EOFError()):
        with pytest.raises(IOError):
            inference.predict_and_save(
                [bad], tmp_path, True, False, False, False, model_or_model_path=ICASSP_2022_MODEL_PATH
            )
    output = capsys.readouterr().out
    assert f"Could not process {bad}: EOFError. Skipping it." in output


@pytest.mark.parametrize("as_path", [True, False])
def test_predict_and_save_accepts_a_single_audio_path(tmp_path: pathlib.Path, as_path: bool) -> None:
    audio = tmp_path / "song.wav"
    audio.write_bytes((RESOURCES_PATH / "vocadito_10.wav").read_bytes())
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    inference.predict_and_save(
        audio if as_path else str(audio),
        output_dir,
        True,
        False,
        False,
        False,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )
    assert [p.name for p in output_dir.iterdir()] == ["song_basic_pitch.mid"]


def test_predict_and_save_rejects_invalid_note_options_before_processing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        with pytest.raises(ValueError, match="onset threshold"):
            inference.predict_and_save(
                [output_dir / "missing.wav"],
                output_dir,
                True,
                False,
                False,
                False,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=2,
            )
        assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("samplerate", [0, -5, 1.5, True])
def test_predict_and_save_rejects_invalid_sonification_samplerate_before_writing(samplerate) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = pathlib.Path(tmpdir)
        with pytest.raises(ValueError, match="sonification sample rate"):
            inference.predict_and_save(
                [RESOURCES_PATH / "vocadito_10.wav"],
                output_dir,
                True,
                True,
                True,
                True,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                sonification_samplerate=samplerate,
            )
        assert list(output_dir.iterdir()) == []


def test_predict_onset_threshold() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for onset_threshold in [0, 0.3, 0.8, 1]:
        inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
        )


def test_predict_frame_threshold() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for frame_threshold in [0, 0.3, 0.8, 1]:
        inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            frame_threshold=frame_threshold,
        )


def test_predict_min_note_length() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for minimum_note_length in [10, 100, 1000]:
        _, _, note_events = inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            minimum_note_length=minimum_note_length,
        )
        min_len_s = minimum_note_length / 1000.0
        note_lengths = [n[1] - n[0] >= min_len_s for n in note_events]
        assert all(note_lengths)


def test_predict_min_freq() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for minimum_frequency in [40, 80, 200, 2000]:
        _, _, note_events = inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            minimum_frequency=minimum_frequency,
        )
        min_freq_midi = np.round(librosa.hz_to_midi(minimum_frequency))
        note_pitch = [n[2] >= min_freq_midi for n in note_events]
        assert all(note_pitch)


def test_predict_max_freq() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    for maximum_frequency in [40, 80, 200, 2000]:
        _, _, note_events = inference.predict(
            test_audio_path,
            ICASSP_2022_MODEL_PATH,
            maximum_frequency=maximum_frequency,
        )
        max_freq_midi = np.round(librosa.hz_to_midi(maximum_frequency))
        note_pitch = [n[2] <= max_freq_midi for n in note_events]
        assert all(note_pitch)


def test_window_audio_file() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    audio, _ = librosa.load(str(test_audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    audio_windowed, window_times = zip(*inference.window_audio_file(audio, AUDIO_N_SAMPLES - 30 * FFT_HOP))
    assert len(audio_windowed) == 6
    assert len(window_times) == 6
    for time in window_times:
        assert time["start"] <= time["end"]
    np.testing.assert_equal(audio[:AUDIO_N_SAMPLES], np.squeeze(audio_windowed[0]))


def test_get_audio_input() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    audio, _ = librosa.load(str(test_audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    overlap_len = 30 * FFT_HOP
    audio = np.concatenate([np.zeros((overlap_len // 2,), dtype=np.float32), audio])
    audio_windowed: List[npt.NDArray[np.float32]] = []
    window_times: List[Dict[str, float]] = []
    for audio_window, window_time, original_length in inference.get_audio_input(
        test_audio_path, overlap_len, AUDIO_N_SAMPLES - overlap_len
    ):
        audio_windowed.append(audio_window)
        window_times.append(window_time)
    audio_windowed = np.array(audio_windowed)
    assert len(audio_windowed) == 6
    assert len(window_times) == 6
    for time in window_times:
        assert time["start"] <= time["end"]
    np.testing.assert_equal(audio[:AUDIO_N_SAMPLES], np.squeeze(audio_windowed[0]))

    assert original_length == 200607


def test_predict_writes_debug_file() -> None:
    test_audio_path = RESOURCES_PATH / "vocadito_10.wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        debug_file = pathlib.Path(tmpdir) / "debug.json"
        _, _, note_events = inference.predict(test_audio_path, ICASSP_2022_MODEL_PATH, debug_file=debug_file)
        with open(debug_file) as f:
            debug_data = json.load(f)
    assert np.array(debug_data["audio_windowed"]).shape == (1, AUDIO_N_SAMPLES, 1)
    assert set(debug_data["unwrapped_output"].keys()) == {"note", "onset", "contour"}
    assert len(debug_data["estimated_notes"]) == len(note_events)


def test_save_note_events_keeps_velocity_in_midi_range(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "events.csv"
    inference.save_note_events([(0.0, 1.0, 60, 1.2, None), (1.0, 2.0, 62, -0.1, None)], path)
    rows = path.read_text().strip().splitlines()[1:]
    assert [int(row.split(",")[3]) for row in rows] == [127, 1]


def test_save_note_events_writes_pitch_bends_in_a_single_column(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "events.csv"
    inference.save_note_events(
        [(0.0, 1.0, 60, 0.5, [1, -2, 3]), (1.0, 2.0, 62, 0.5, None), (2.0, 3.0, 64, 0.5, [])], path
    )
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["start_time_s", "end_time_s", "pitch_midi", "velocity", "pitch_bend"]
    assert all(len(row) == 5 for row in rows)
    assert rows[1][4] == "1 -2 3"
    assert rows[2][4] == ""
    assert rows[3][4] == ""


def test_predict_and_save_rejects_audio_files_that_share_an_output_name() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        first = pathlib.Path(tmpdir) / "a" / "song.wav"
        second = pathlib.Path(tmpdir) / "b" / "song.wav"
        output_dir = pathlib.Path(tmpdir) / "out"
        for directory in (first.parent, second.parent, output_dir):
            directory.mkdir()
        for audio in (first, second):
            audio.write_bytes((RESOURCES_PATH / "vocadito_10.wav").read_bytes())

        with pytest.raises(ValueError, match="song_basic_pitch"):
            inference.predict_and_save(
                [first, second],
                output_dir,
                True,
                False,
                False,
                False,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
            )
        assert list(output_dir.iterdir()) == []


def test_verify_unique_output_names_allows_distinct_names_and_repeated_paths(tmp_path: pathlib.Path) -> None:
    inference.verify_unique_output_names([tmp_path / "one.wav", tmp_path / "two.mp3", tmp_path / "one.wav"])
    with pytest.raises(ValueError, match="would both write"):
        inference.verify_unique_output_names([tmp_path / "song.wav", tmp_path / "song.mp3"])


def test_predict_and_save_rejects_a_call_that_saves_nothing(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="nothing would be saved"):
        inference.predict_and_save(
            [RESOURCES_PATH / "vocadito_10.wav"],
            tmp_path,
            False,
            False,
            False,
            False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
    assert list(tmp_path.iterdir()) == []


def test_predict_and_save_rejects_a_missing_output_directory_before_processing(tmp_path: pathlib.Path) -> None:
    missing_dir = tmp_path / "missing"
    with pytest.raises(ValueError, match="does not exist"):
        inference.predict_and_save(
            [RESOURCES_PATH / "vocadito_10.wav"],
            missing_dir,
            True,
            False,
            False,
            False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
    assert not missing_dir.exists()


def test_predict_and_save_loads_the_model_once_for_all_audio_files(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads = []
    original_init = inference.Model.__init__

    def counting_init(self: inference.Model, model_path: object) -> None:
        loads.append(model_path)
        original_init(self, model_path)  # type: ignore

    monkeypatch.setattr(inference.Model, "__init__", counting_init)
    audio_paths = []
    for name in ("first.wav", "second.wav"):
        audio = tmp_path / name
        audio.write_bytes((RESOURCES_PATH / "vocadito_10.wav").read_bytes())
        audio_paths.append(audio)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    inference.predict_and_save(audio_paths, output_dir, True, False, False, False, ICASSP_2022_MODEL_PATH)

    assert len(loads) == 1
    for name in ("first", "second"):
        assert (output_dir / f"{name}_basic_pitch.mid").exists()


def test_predict_and_save_rejects_an_unloadable_model_before_processing(tmp_path: pathlib.Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes((RESOURCES_PATH / "vocadito_10.wav").read_bytes())
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="cannot be loaded"):
        inference.predict_and_save([audio], output_dir, True, False, False, False, tmp_path / "missing_model")

    assert list(output_dir.iterdir()) == []


def test_verify_debug_file_accepts_no_file_and_a_file_in_an_existing_directory(tmp_path: pathlib.Path) -> None:
    inference.verify_debug_file(None)
    inference.verify_debug_file(tmp_path / "debug.json")


def test_verify_debug_file_rejects_a_missing_directory_and_a_directory(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="debug file does not exist"):
        inference.verify_debug_file(tmp_path / "missing" / "debug.json")
    with pytest.raises(ValueError, match="is a directory"):
        inference.verify_debug_file(tmp_path)


def test_predict_and_save_rejects_an_unwritable_debug_file_before_processing(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="debug file does not exist"):
        inference.predict_and_save(
            [RESOURCES_PATH / "vocadito_10.wav"],
            tmp_path,
            True,
            False,
            False,
            False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            debug_file=tmp_path / "missing" / "debug.json",
        )
    assert list(tmp_path.iterdir()) == []


def test_verify_input_path_non_existent(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        inference.verify_input_path(tmp_path / "missing.wav")


def test_verify_input_path_directory(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="is not a file path"):
        inference.verify_input_path(tmp_path)


def test_verify_input_path_valid_file(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "audio.wav"
    f.write_bytes(b"data")
    inference.verify_input_path(f)


def test_verify_output_dir_non_existent(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        inference.verify_output_dir(tmp_path / "missing")


def test_verify_output_dir_file(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "file.txt"
    f.write_bytes(b"data")
    with pytest.raises(ValueError, match="is not a directory"):
        inference.verify_output_dir(f)


def test_verify_output_dir_valid(tmp_path: pathlib.Path) -> None:
    inference.verify_output_dir(tmp_path)


def test_unwrap_output_bad_shape() -> None:
    with pytest.raises(ValueError, match="three-dimensional"):
        inference.unwrap_output(np.ones((100, 88)), 44100, 30, 7424)


def test_unwrap_output_valid_shape() -> None:
    result = inference.unwrap_output(np.ones((2, 172, 88)), 44100, 30, 7424)
    assert result is not None
    assert len(result.shape) == 2


@pytest.mark.parametrize("function", [inference.predict, inference.predict_and_save])
def test_documented_arguments_of_the_public_functions_exist_in_their_signature(function) -> None:  # type: ignore
    import inspect
    import re

    docstring = inspect.cleandoc(function.__doc__)
    arguments_section = re.search(r"Args:\n(.*?)(?:\n\s*\n|\nReturns|$)", docstring, re.DOTALL)
    assert arguments_section is not None
    documented = re.findall(r"^ {4}(\w+):", arguments_section.group(1), re.MULTILINE)
    assert documented
    assert set(documented) <= set(inspect.signature(function).parameters)
