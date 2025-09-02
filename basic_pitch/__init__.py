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
import logging
import pathlib


try:
    import coremltools

    CT_PRESENT = True
except ImportError:
    CT_PRESENT = False
    logging.warning(
        "Coremltools is not installed. "
        "If you plan to use a CoreML Saved Model, "
        "reinstall basic-pitch with `pip install 'basic-pitch[coreml]'`"
    )

try:
    import tensorflow.lite

    TFLITE_PRESENT = True
except ImportError:
    try:
        import tflite_runtime.interpreter
        TFLITE_PRESENT = True
    except ImportError:
        TFLITE_PRESENT = False
        logging.warning(
            "tflite-runtime is not installed. " +
            "If you plan to use a TFLite Model, " +
            "reinstall basic-pitch with `pip install 'basic-pitch tflite-runtime'` or " +
            "`pip install 'basic-pitch[tf]'"
        )

try:
    import onnx

    ONNX_PRESENT = True
except ImportError:
    ONNX_PRESENT = False
    logging.warning(
        "onnxruntime is not installed. "
        "If you plan to use an ONNX Model, "
        "reinstall basic-pitch with `pip install 'basic-pitch[onnx]'`"
    )


try:
    import tensorflow

    TF_PRESENT = True
except ImportError:
    TF_PRESENT = False
    logging.warning(
        "Tensorflow is not installed. "
        "If you plan to use a TF Saved Model, "
        "reinstall basic-pitch with `pip install 'basic-pitch[tf]'`"
    )

def build_icassp_2022_model_path() -> pathlib.Path:
    return f"{pathlib.Path(__file__).parent}/saved_models/icassp_2022/nmp"


ICASSP_2022_MODEL_PATH = build_icassp_2022_model_path()
