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
