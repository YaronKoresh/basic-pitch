import pytest
import json
import pathlib
from typing import Iterator
from unittest import mock

RESOURCES_PATH = pathlib.Path(__file__).parent.parent / "resources"
GUITAR_SET_TEST_INDEX = json.loads((RESOURCES_PATH / "data" / "guitarset" / "dummy_index.json").read_text())
IKALA_TEST_INDEX = json.loads((RESOURCES_PATH / "data" / "ikala" / "dummy_index.json").read_text())
MAESTRO_TEST_INDEX = json.loads((RESOURCES_PATH / "data" / "maestro" / "dummy_index.json").read_text())
METADATA_TEST_INDEX = json.loads((RESOURCES_PATH / "data" / "maestro" / "maestro-v2.0.0.json").read_text())
MEDLEYDB_PITCH_TEST_INDEX = json.loads((RESOURCES_PATH / "data" / "medleydb_pitch" / "dummy_index.json").read_text())
SLAKH_TEST_INDEX = json.loads((RESOURCES_PATH / "data" / "slakh" / "dummy_index.json").read_text())


@pytest.fixture
def mock_slakh_index() -> Iterator[None]:
    with mock.patch("mirdata.datasets.slakh.Dataset.download"):
        with mock.patch("mirdata.datasets.slakh.Dataset._index", new=SLAKH_TEST_INDEX):
            yield


@pytest.fixture
def mock_medleydb_pitch_index() -> Iterator[None]:
    with mock.patch("mirdata.datasets.medleydb_pitch.Dataset.download"):
        with mock.patch("mirdata.datasets.medleydb_pitch.Dataset._index", new=MEDLEYDB_PITCH_TEST_INDEX):
            yield


@pytest.fixture
def mock_maestro_index() -> Iterator[None]:
    index_with_metadata = MAESTRO_TEST_INDEX
    metadata = {mdata["midi_filename"].split(".")[0]: mdata for mdata in METADATA_TEST_INDEX}
    with mock.patch("mirdata.datasets.maestro.Dataset.download"):
        with mock.patch("mirdata.datasets.maestro.Dataset._metadata", new=metadata):
            with mock.patch("mirdata.datasets.maestro.Dataset._index", new=index_with_metadata):
                yield


@pytest.fixture
def mock_guitarset_index() -> Iterator[None]:
    with mock.patch("mirdata.datasets.guitarset.Dataset.download"):
        with mock.patch("mirdata.datasets.guitarset.Dataset._index", new=GUITAR_SET_TEST_INDEX):
            yield


@pytest.fixture
def mock_ikala_index() -> Iterator[None]:
    with mock.patch("mirdata.datasets.ikala.Dataset.download"):
        with mock.patch("mirdata.datasets.ikala.Dataset._index", new=IKALA_TEST_INDEX):
            yield
