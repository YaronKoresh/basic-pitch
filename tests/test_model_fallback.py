import pathlib
from typing import List

import pytest

from basic_pitch import inference


def _model_dir(tmp_path: pathlib.Path, *siblings: str) -> pathlib.Path:
    model = tmp_path / "nmp"
    model.mkdir()
    for suffix in siblings:
        (tmp_path / ("nmp" + suffix)).write_text("x")
    return model


def _patch_load(monkeypatch: pytest.MonkeyPatch, loadable: List[str]) -> List[str]:
    tried: List[str] = []

    def fake_load(self: inference.Model, model_path: object) -> None:
        tried.append(pathlib.Path(str(model_path)).name)
        if pathlib.Path(str(model_path)).name not in loadable:
            raise ValueError("cannot load")
        self.loaded = str(model_path)

    monkeypatch.setattr(inference.Model, "_load", fake_load)
    return tried


def test_model_uses_onnx_file_when_saved_model_cannot_be_loaded(tmp_path, monkeypatch) -> None:
    model = _model_dir(tmp_path, ".onnx", ".tflite")
    tried = _patch_load(monkeypatch, ["nmp.onnx"])
    loaded = inference.Model(model)
    assert tried == ["nmp", "nmp.onnx"]
    assert loaded.loaded.endswith("nmp.onnx")


def test_model_tries_next_sibling_when_first_cannot_be_loaded(tmp_path, monkeypatch) -> None:
    model = _model_dir(tmp_path, ".onnx", ".tflite")
    tried = _patch_load(monkeypatch, ["nmp.tflite"])
    inference.Model(model)
    assert tried == ["nmp", "nmp.onnx", "nmp.tflite"]


def test_model_keeps_original_error_when_nothing_loads(tmp_path, monkeypatch) -> None:
    model = _model_dir(tmp_path, ".onnx")
    _patch_load(monkeypatch, [])
    with pytest.raises(ValueError, match="cannot load"):
        inference.Model(model)


def test_model_does_not_look_for_siblings_when_path_loads(tmp_path, monkeypatch) -> None:
    model = _model_dir(tmp_path, ".onnx")
    tried = _patch_load(monkeypatch, ["nmp"])
    inference.Model(model)
    assert tried == ["nmp"]
