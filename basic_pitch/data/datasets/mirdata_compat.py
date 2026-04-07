import json
import os
import shutil
from typing import Any, List, Tuple


COPY_BUFFER_SIZE = 1024 * 1024
MAESTRO_METADATA_FILENAME = "maestro-v2.0.0.json"
MAESTRO_INDEX_FILENAMES = ("maestro_index_2.0.0.json", "maestro_index_2.0.0_sample.json")


def _join_resource_path(root_path: str, child_path: str) -> str:
    if "://" in root_path or ("/" in root_path and "\\" not in root_path):
        return f"{root_path.rstrip('/')}/{child_path}"
    return os.path.join(root_path, child_path)


def copy_remote_file(filesystem: Any, source_path: str, destination_path: str) -> None:
    destination_directory = os.path.dirname(destination_path)
    if destination_directory:
        os.makedirs(destination_directory, exist_ok=True)

    with filesystem.open(source_path) as source_handle, open(destination_path, "wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=COPY_BUFFER_SIZE)


def ensure_mirdata_index(dataset: Any) -> None:
    try:
        _ = dataset.track_ids
    except FileNotFoundError:
        dataset.download(partial_download=["index"])


def get_mirdata_track_ids(dataset: Any) -> List[str]:
    ensure_mirdata_index(dataset)
    return list(dataset.track_ids)


def get_mirdata_track_split(track: Any) -> str:
    for attribute_name in ("split", "data_split"):
        split = getattr(track, attribute_name, None)
        if split is not None:
            return split

    raise AttributeError(f"Track {track} does not expose a split attribute.")


def _load_json_resource(filesystem: Any, resource_path: str) -> Any:
    with filesystem.open(resource_path) as resource_handle:
        raw_resource = resource_handle.read()

    if isinstance(raw_resource, bytes):
        raw_resource = raw_resource.decode("utf-8")

    return json.loads(raw_resource)


def _resolve_maestro_metadata_from_index(filesystem: Any, index_path: str) -> str:
    index_data = _load_json_resource(filesystem, index_path)
    metadata_entries = index_data.get("metadata", {})

    for metadata_entry in metadata_entries.values():
        metadata_filename = metadata_entry[0] if isinstance(metadata_entry, (list, tuple)) else metadata_entry
        if isinstance(metadata_filename, str) and metadata_filename.endswith(".json"):
            metadata_path = _join_resource_path(os.path.dirname(index_path), metadata_filename)
            if filesystem.exists(metadata_path):
                return metadata_path
            raise FileNotFoundError(f"Could not find MAESTRO metadata file at {metadata_path}.")

    raise FileNotFoundError(f"Could not resolve a MAESTRO metadata file from {index_path}.")


def resolve_maestro_metadata_path(filesystem: Any, source: str) -> str:
    source_name = os.path.basename(source)

    if source_name == MAESTRO_METADATA_FILENAME:
        return source

    if source_name in MAESTRO_INDEX_FILENAMES:
        return _resolve_maestro_metadata_from_index(filesystem, source)

    metadata_path = _join_resource_path(source, MAESTRO_METADATA_FILENAME)
    if filesystem.exists(metadata_path):
        return metadata_path

    for index_filename in MAESTRO_INDEX_FILENAMES:
        index_path = _join_resource_path(source, index_filename)
        if filesystem.exists(index_path):
            return _resolve_maestro_metadata_from_index(filesystem, index_path)

    raise FileNotFoundError(f"Could not find a MAESTRO metadata file in {source}.")


def load_maestro_input_data(filesystem: Any, source: str) -> List[Tuple[str, str]]:
    metadata = _load_json_resource(filesystem, resolve_maestro_metadata_path(filesystem, source))
    return [(os.path.splitext(entry["midi_filename"])[0], entry["split"]) for entry in metadata]