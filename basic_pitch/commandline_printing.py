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

import os
import pathlib
import threading
from contextlib import contextmanager
from typing import Iterator, Union

TF_LOG_LEVEL_KEY = "TF_CPP_MIN_LOG_LEVEL"
TF_LOG_LEVEL_NO_WARNINGS_VALUE = "3"
DEFAULT_PRINT_INDENT = "  "

s_print_lock = threading.Lock()
OUTPUT_EMOJIS = {
    "MIDI": "💅",
    "MODEL_OUTPUT_NPZ": "💁‍♀️",
    "MIDI_SONIFICATION": "🎧",
    "NOTE_EVENTS": "🌸",
}


def generating_file_message(output_type: str) -> None:
    """Print a message that a file is being generated

    Args:
        output_type: string indicating which kind of file is being generated

    """
    print(f"\n\n{DEFAULT_PRINT_INDENT}Creating {output_type.replace('_', ' ').lower()}...")


def file_saved_confirmation(output_type: str, save_path: Union[pathlib.Path, str]) -> None:
    """Print a confirmation that the file was saved succesfully

    Args:
        output_type: The kind of file that is being generated.
        save_path: The path to output file.

    """
    emoji = OUTPUT_EMOJIS.get(output_type, "")
    print(f"{DEFAULT_PRINT_INDENT}{emoji} Saved to {save_path}")


def failed_to_save(output_type: str, save_path: Union[pathlib.Path, str]) -> None:
    """Print a failure to save message

    Args:
        output_type: The kind of file that is being generated.
        save_path: The path to output file.

    """
    print(f"\n🚨 Failed to save {output_type.replace('_', ' ').lower()} to {save_path} \n")


@contextmanager
def no_tf_warnings() -> Iterator[None]:
    """
    Supress tensorflow warnings in this context
    """
    tf_logging_level = os.environ.get(TF_LOG_LEVEL_KEY)
    os.environ[TF_LOG_LEVEL_KEY] = TF_LOG_LEVEL_NO_WARNINGS_VALUE
    try:
        yield
    finally:
        if tf_logging_level is None:
            os.environ.pop(TF_LOG_LEVEL_KEY, None)
        else:
            os.environ[TF_LOG_LEVEL_KEY] = tf_logging_level
