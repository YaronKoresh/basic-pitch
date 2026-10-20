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

import ast
import pathlib
from typing import Iterator, List

import pytest

PACKAGE = pathlib.Path(__file__).parent.parent / "basic_pitch"
NOT_TYPES = {"np.array": "a function, not a type", "np.ndarray": "a generic type that needs its arguments"}


def _dotted_name(node: ast.Attribute) -> str:
    if isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return node.attr


def _annotations(tree: ast.AST) -> Iterator[ast.expr]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            for argument in (
                arguments.posonlyargs + arguments.args + arguments.kwonlyargs + [arguments.vararg, arguments.kwarg]
            ):
                if argument is not None and argument.annotation is not None:
                    yield argument.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


@pytest.mark.parametrize("path", sorted(PACKAGE.rglob("*.py")), ids=lambda path: str(path.relative_to(PACKAGE)))
def test_annotations_use_numpy_typing_instead_of_np_array_or_bare_np_ndarray(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: List[str] = []
    for annotation in _annotations(tree):
        for node in ast.walk(annotation):
            if isinstance(node, ast.Attribute) and _dotted_name(node) in NOT_TYPES:
                inside_subscript = any(
                    isinstance(parent, ast.Subscript) and parent.value is node for parent in ast.walk(annotation)
                )
                if not inside_subscript:
                    found.append(f"line {node.lineno}: {_dotted_name(node)} is {NOT_TYPES[_dotted_name(node)]}")
    assert not found, "; ".join(found)
