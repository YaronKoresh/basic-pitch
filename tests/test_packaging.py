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

import configparser
import pathlib
import re
import subprocess
import sys
from typing import Any, Dict, List

import pytest
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PYPROJECT = pathlib.Path(__file__).parent.parent / "pyproject.toml"

RUNTIMES = {"tensorflow", "tensorflow-macos", "tflite-runtime", "onnxruntime", "coremltools"}


def _active_requirements(requirements: List[str], system: str, python_version: str) -> Dict[str, Requirement]:
    environment = {"platform_system": system, "python_version": python_version}
    parsed = (Requirement(r) for r in requirements)
    return {r.name: r for r in parsed if r.marker is None or r.marker.evaluate(environment)}


def _project() -> Dict[str, Any]:
    with open(PYPROJECT, "rb") as f:
        project: Dict[str, Any] = tomllib.load(f)["project"]
    return project


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
@pytest.mark.parametrize("python_version", ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"])
def test_a_model_runtime_is_installed_by_default(system: str, python_version: str) -> None:
    active = _active_requirements(_project()["dependencies"], system, python_version)
    assert RUNTIMES & set(active), f"no model runtime for {system} python {python_version}"


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
@pytest.mark.parametrize("python_version", ["3.12", "3.13"])
def test_tensorflow_is_not_a_default_dependency_on_new_python(system: str, python_version: str) -> None:
    active = _active_requirements(_project()["dependencies"], system, python_version)
    assert not {"tensorflow", "tensorflow-macos"} & set(active)


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
@pytest.mark.parametrize("python_version", ["3.12", "3.13"])
def test_tf_extra_installs_keras_2_compatible_tensorflow_on_new_python(system: str, python_version: str) -> None:
    active = _active_requirements(_project()["optional-dependencies"]["tf"], system, python_version)
    assert "tensorflow-macos" not in active
    assert active["tensorflow"].specifier.contains("2.21.0")
    assert not active["tensorflow"].specifier.contains("2.15.0")
    assert {"tf_keras", "tensorboard"} <= set(active)


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
@pytest.mark.parametrize("python_version", ["3.8", "3.9", "3.10", "3.11"])
def test_tf_extra_keeps_the_pinned_tensorflow_on_older_python(system: str, python_version: str) -> None:
    active = _active_requirements(_project()["optional-dependencies"]["tf"], system, python_version)
    name = "tensorflow-macos" if system == "Darwin" else "tensorflow"
    if python_version == "3.8" and system == "Darwin":
        return
    assert not active[name].specifier.contains("2.16.0")
    assert "tf_keras" not in active


def test_classifiers_cover_new_python_versions() -> None:
    classifiers = _project()["classifiers"]
    for version in ["3.12", "3.13"]:
        assert f"Programming Language :: Python :: {version}" in classifiers


def test_package_imports_without_any_model_runtime() -> None:
    code = (
        "import sys\n"
        "for name in ('tensorflow', 'coremltools', 'tflite_runtime', 'tflite_runtime.interpreter', 'onnxruntime'):\n"
        "    sys.modules[name] = None\n"
        "import basic_pitch\n"
        "assert not (basic_pitch.TF_PRESENT or basic_pitch.CT_PRESENT or basic_pitch.TFLITE_PRESENT"
        " or basic_pitch.ONNX_PRESENT)\n"
        "print(basic_pitch.ICASSP_2022_MODEL_PATH.name)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "nmp"


def test_tox_mypy_command_can_check_both_conftest_files() -> None:
    pytest.importorskip("mypy")
    tox = configparser.ConfigParser(interpolation=None)
    tox.read(PYPROJECT.parent / "tox.ini")
    command = tox["testenv:mypy"]["commands"].split()
    assert command[0] == "mypy"
    flags = [argument for argument in command[1:] if argument.startswith("--")]
    assert "--explicit-package-bases" in flags

    result = subprocess.run(
        [sys.executable, "-m", "mypy", "tests/conftest.py", "tests/data/conftest.py", *flags],
        cwd=PYPROJECT.parent,
        capture_output=True,
        text=True,
    )
    assert "Duplicate module" not in result.stdout + result.stderr


def test_dockerfile_and_tox_install_only_extras_that_the_project_defines() -> None:
    root = PYPROJECT.parent
    texts = [(root / "Dockerfile").read_text(), (root / "tox.ini").read_text()]
    requested = {
        name.strip() for text in texts for match in re.findall(r"\.\[([^\]]+)\]", text) for name in match.split(",")
    }
    assert requested
    assert requested <= set(_project()["optional-dependencies"])


def test_tox_mypy_environment_installs_pytest_so_test_decorators_are_typed() -> None:
    parser = configparser.ConfigParser()
    parser.read(PYPROJECT.parent / "tox.ini")
    deps = {line.strip() for line in parser["testenv:mypy"]["deps"].splitlines() if line.strip()}
    assert {"mypy", "pytest", "pytest-mock"} <= deps


@pytest.mark.parametrize("python_version", ["3.12", "3.13"])
def test_resampy_without_pkg_resources_is_allowed_on_new_python(python_version: str) -> None:
    # resampy before 0.4.3 imports pkg_resources, which a new virtual environment of Python 3.12+ does not have
    active = _active_requirements(_project()["dependencies"], "Linux", python_version)
    assert active["resampy"].specifier.contains("0.4.3")
    assert not active["resampy"].specifier.contains("0.4.2")
