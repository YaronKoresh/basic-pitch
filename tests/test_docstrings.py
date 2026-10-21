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
import inspect
import re
from typing import Any, Callable, List

import pytest

from basic_pitch import note_creation

FUNCTIONS_WITH_ARGS_SECTION: List[Callable[..., Any]] = [note_creation.model_output_to_notes]

try:
    from basic_pitch.data import tf_example_deserialization as deserialization

    FUNCTIONS_WITH_ARGS_SECTION += [
        deserialization.sample_datasets,
        deserialization.transcription_file_generator,
        deserialization.reduce_transcription_inputs,
        deserialization.get_sample_weights,
        deserialization.extract_window,
        deserialization.extract_random_window,
        deserialization.get_transcription_chunks,
    ]
except ImportError:  # the data extras are not installed
    pass


def documented_arguments(function: Callable[..., Any]) -> List[str]:
    docstring = inspect.cleandoc(function.__doc__ or "")
    section = docstring.split("Args:")[1]
    section = re.split(r"\n\s*\n|\n(?:Returns|Raises|Yields)", section)[0]
    indent = min(len(line) - len(line.lstrip()) for line in section.splitlines() if line.strip() and ":" in line)
    return re.findall(rf"^ {{{indent}}}(\w+)\s*:", section, re.MULTILINE)


@pytest.mark.parametrize("function", FUNCTIONS_WITH_ARGS_SECTION, ids=lambda function: function.__name__)
def test_docstring_names_exactly_the_parameters_of_the_signature(function: Callable[..., Any]) -> None:
    assert sorted(documented_arguments(function)) == sorted(inspect.signature(function).parameters)


def test_model_output_to_notes_docstring_names_the_note_array_and_five_field_note_events() -> None:
    docstring = note_creation.model_output_to_notes.__doc__ or ""
    assert "'note'" in docstring
    assert "'frame'" not in docstring
    assert "pitch_bends" in docstring.split("Returns:")[1]
