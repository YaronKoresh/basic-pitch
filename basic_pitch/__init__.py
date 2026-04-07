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

import enum
import importlib.util
import pathlib


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


CT_PRESENT = _module_available("coremltools")
TF_PRESENT = _module_available("tensorflow")
TFLITE_PRESENT = TF_PRESENT or _module_available("tflite_runtime.interpreter")
ONNX_PRESENT = _module_available("onnxruntime")


class FilenameSuffix(enum.Enum):
    tf = ""
    coreml = ".mlpackage"
    tflite = ".tflite"
    onnx = ".onnx"


def build_icassp_2022_model_path(filename_suffix: FilenameSuffix = FilenameSuffix.tf) -> pathlib.Path:
    return pathlib.Path(
        f"{pathlib.Path(__file__).parent / 'saved_models' / 'icassp_2022' / 'nmp'}{filename_suffix.value}"
    )


ICASSP_2022_MODEL_PATH = build_icassp_2022_model_path()
