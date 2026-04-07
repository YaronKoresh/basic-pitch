import io
import json
import pathlib
from typing import Dict, List

from basic_pitch.data.datasets.mirdata_compat import (
    copy_remote_file,
    get_mirdata_track_ids,
    get_mirdata_track_split,
    load_maestro_input_data,
    resolve_maestro_metadata_path,
)


class FakeFilesystem:
    def __init__(self, files: Dict[str, bytes]) -> None:
        self.files = files

    def exists(self, path: str) -> bool:
        return path in self.files

    def open(self, path: str) -> io.BytesIO:
        return io.BytesIO(self.files[path])


class MissingIndexDataset:
    def __init__(self) -> None:
        self.download_calls: List[List[str]] = []

    @property
    def track_ids(self) -> List[str]:
        if not self.download_calls:
            raise FileNotFoundError("missing index")
        return ["track-1", "track-2"]

    def download(self, partial_download: List[str]) -> None:
        self.download_calls.append(partial_download)


def test_copy_remote_file(tmp_path: pathlib.Path) -> None:
    source_path = "remote/path/file.txt"
    destination_path = tmp_path / "nested" / "file.txt"
    filesystem = FakeFilesystem({source_path: b"basic-pitch"})

    copy_remote_file(filesystem, source_path, str(destination_path))

    assert destination_path.read_bytes() == b"basic-pitch"


def test_get_mirdata_track_ids_downloads_index() -> None:
    dataset = MissingIndexDataset()

    track_ids = get_mirdata_track_ids(dataset)

    assert track_ids == ["track-1", "track-2"]
    assert dataset.download_calls == [["index"]]


def test_get_mirdata_track_split_uses_supported_attributes() -> None:
    split_track = type("SplitTrack", (), {"split": "train"})()
    data_split_track = type("DataSplitTrack", (), {"data_split": "validation"})()

    assert get_mirdata_track_split(split_track) == "train"
    assert get_mirdata_track_split(data_split_track) == "validation"


def test_load_maestro_input_data_from_metadata_file() -> None:
    metadata_path = "/tmp/maestro-v2.0.0.json"
    filesystem = FakeFilesystem(
        {
            metadata_path: json.dumps(
                [
                    {
                        "midi_filename": "2004/example_track.midi",
                        "split": "train",
                    }
                ]
            ).encode("utf-8")
        }
    )

    data = load_maestro_input_data(filesystem, metadata_path)

    assert data == [("2004/example_track", "train")]


def test_resolve_maestro_metadata_path_from_index_file() -> None:
    root_path = "/tmp/maestro"
    index_path = f"{root_path}/maestro_index_2.0.0.json"
    metadata_path = f"{root_path}/maestro-v2.0.0.json"
    filesystem = FakeFilesystem(
        {
            index_path: json.dumps(
                {
                    "metadata": {"maestro-v2.0.0": ["maestro-v2.0.0.json", "checksum"]},
                    "tracks": {},
                    "version": "2.0.0",
                }
            ).encode("utf-8"),
            metadata_path: json.dumps(
                [
                    {
                        "midi_filename": "2004/example_track.midi",
                        "split": "validation",
                    }
                ]
            ).encode("utf-8"),
        }
    )

    resolved_path = resolve_maestro_metadata_path(filesystem, root_path)
    data = load_maestro_input_data(filesystem, root_path)

    assert resolved_path == metadata_path
    assert data == [("2004/example_track", "validation")]