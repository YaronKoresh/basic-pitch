import pathlib
import sys
import tempfile

import pytest

from basic_pitch import predict

RESOURCES_PATH = pathlib.Path(__file__).parent / "resources"


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["basic-pitch", *args])
    predict.main()


def test_missing_output_dir_exits_with_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = pathlib.Path(tmpdir) / "missing"
        with pytest.raises(SystemExit) as exit_info:
            run_main(monkeypatch, str(missing), str(RESOURCES_PATH / "vocadito_10.wav"))
    assert exit_info.value.code == 2
    assert "is not a directory" in capsys.readouterr().err


def test_missing_input_file_exits_with_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exit_info:
            run_main(monkeypatch, tmpdir, str(pathlib.Path(tmpdir) / "missing.wav"))
    assert exit_info.value.code == 2
    assert "is not a file path" in capsys.readouterr().err


def test_unreadable_audio_exits_with_failure_status(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_audio = pathlib.Path(tmpdir) / "bad.wav"
        bad_audio.write_text("not audio")
        with pytest.raises(SystemExit) as exit_info:
            run_main(monkeypatch, tmpdir, str(bad_audio))
    assert exit_info.value.code == 1


@pytest.mark.parametrize("samplerate", ["0", "-5"])
def test_invalid_sonification_samplerate_exits_with_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, samplerate: str
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exit_info:
            run_main(
                monkeypatch,
                tmpdir,
                str(RESOURCES_PATH / "vocadito_10.wav"),
                "--sonify-midi",
                "--sonification-samplerate",
                samplerate,
            )
        assert list(pathlib.Path(tmpdir).iterdir()) == []
    assert exit_info.value.code == 2
    assert "sonification sample rate" in capsys.readouterr().err


def test_unloadable_model_path_exits_with_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(SystemExit) as exit_info:
            run_main(
                monkeypatch,
                tmpdir,
                str(RESOURCES_PATH / "vocadito_10.wav"),
                "--model-path",
                str(pathlib.Path(tmpdir) / "missing-model"),
            )
        assert list(pathlib.Path(tmpdir).iterdir()) == []
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert "cannot be loaded" in captured.err
    assert "Traceback" not in captured.err


def capture_save_midi(monkeypatch: pytest.MonkeyPatch, *args: str) -> bool:
    from basic_pitch import inference

    captured = {}

    def fake_predict_and_save(audio_path_list, output_dir, save_midi, *rest, **kwargs):  # type: ignore
        captured["save_midi"] = save_midi

    monkeypatch.setattr(inference, "Model", lambda path: object())
    monkeypatch.setattr(predict, "Model", lambda path: object())
    monkeypatch.setattr(inference, "predict_and_save", fake_predict_and_save)
    with tempfile.TemporaryDirectory() as tmpdir:
        run_main(monkeypatch, tmpdir, str(RESOURCES_PATH / "vocadito_10.wav"), *args)
    return captured["save_midi"]


def test_midi_is_saved_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert capture_save_midi(monkeypatch) is True


def test_no_save_midi_flag_skips_the_midi_file(monkeypatch: pytest.MonkeyPatch) -> None:
    assert capture_save_midi(monkeypatch, "--save-note-events", "--no-save-midi") is False


def test_save_midi_flag_is_still_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    assert capture_save_midi(monkeypatch, "--save-midi") is True


def test_audio_files_with_the_same_output_name_exit_with_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        first = pathlib.Path(tmpdir) / "a" / "song.wav"
        second = pathlib.Path(tmpdir) / "b" / "song.wav"
        for audio in (first, second):
            audio.parent.mkdir()
            audio.write_bytes((RESOURCES_PATH / "vocadito_10.wav").read_bytes())
        with pytest.raises(SystemExit) as exit_info:
            run_main(monkeypatch, tmpdir, str(first), str(second))
    assert exit_info.value.code == 2
    assert "would both write" in capsys.readouterr().err
