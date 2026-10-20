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

import pathlib
import re
from typing import Any, Dict, List

import numpy as np
import pytest

README_PATH = pathlib.Path(__file__).parent.parent / "README.md"


def readme_python_examples() -> List[str]:
    blocks = re.findall(r"```python\n(.*?)```", README_PATH.read_text(encoding="utf-8"), re.DOTALL)
    # the examples mark the values to fill in as <placeholder-name>
    return [re.sub(r"<[a-z-]+>", "placeholder", block) for block in blocks]


def test_readme_has_python_examples() -> None:
    assert len(readme_python_examples()) >= 3


def test_readme_python_examples_are_valid_python() -> None:
    for index, example in enumerate(readme_python_examples()):
        try:
            compile(example, f"README.md python example {index}", "exec")
        except SyntaxError as error:
            raise AssertionError(f"README.md python example {index} is not valid Python: {error}") from error


def test_readme_python_examples_do_not_call_range_without_arguments() -> None:
    for example in readme_python_examples():
        assert "range()" not in example


def test_contributing_names_the_line_length_that_tox_uses_for_black() -> None:
    root = README_PATH.parent
    match = re.search(r'LINE_LENGTH = "?(\d+)"?', (root / "tox.ini").read_text())
    assert match is not None
    line_length = match.group(1)
    contributing = (root / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert f"--line-length {line_length}" in contributing
    assert "black` with defaults" not in contributing


def test_source_comments_and_docstrings_do_not_contain_known_misspellings() -> None:
    misspellings = ["Everytime", "higest", "minium", "matricies"]
    for path in (README_PATH.parent / "basic_pitch").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for word in misspellings:
            assert word not in text, f"{path} contains the misspelling {word}"


def test_readme_describes_the_audio_input_of_predict_and_save_as_files_not_directories() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    assert "`<input-audio-path-list>` & `<output-directory>`" not in readme
    assert "paths of the audio files" in readme


def test_readme_shows_how_to_load_the_saved_model_output(tmp_path: pathlib.Path) -> None:
    example = next(block for block in readme_python_examples() if "basic_pitch_model_output" in block)
    npz_path = tmp_path / "model_output.npz"
    model_output = {"note": np.zeros((4, 88)), "onset": np.zeros((4, 88)), "contour": np.zeros((4, 264))}
    np.savez(npz_path, basic_pitch_model_output=model_output)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allow_pickle"):
        np.load(npz_path)["basic_pitch_model_output"]

    namespace: Dict[str, Any] = {}
    exec(example.replace("placeholder", repr(str(npz_path))), namespace)

    assert set(namespace["model_output"]) == {"note", "onset", "contour"}
