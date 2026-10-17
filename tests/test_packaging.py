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
        return tomllib.load(f)["project"]


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
@pytest.mark.parametrize("python_version", ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"])
def test_a_model_runtime_is_installed_by_default(system: str, python_version: str) -> None:
    active = _active_requirements(_project()["dependencies"], system, python_version)
    assert RUNTIMES & set(active), f"no model runtime for {system} python {python_version}"


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
@pytest.mark.parametrize("python_version", ["3.12", "3.13"])
def test_tensorflow_pin_is_not_required_on_python_without_matching_wheels(system: str, python_version: str) -> None:
    project = _project()
    requirements = project["dependencies"] + project["optional-dependencies"]["tf"]
    active = _active_requirements(requirements, system, python_version)
    assert not {"tensorflow", "tensorflow-macos"} & set(active)


def test_classifiers_cover_new_python_versions() -> None:
    classifiers = _project()["classifiers"]
    for version in ["3.12", "3.13"]:
        assert f"Programming Language :: Python :: {version}" in classifiers
