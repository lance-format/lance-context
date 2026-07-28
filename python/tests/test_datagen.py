from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "python"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lance_context.api import DatagenStore, DatagenStreamWriter  # noqa: E402

_CONTEXT = {"run_id": "run-1", "writer_epoch": "writer-1"}


def _leaf_position(name: str, index: int) -> dict[str, object]:
    return {
        "step_name": name,
        "step_kind": "leaf",
        "index": index,
        "enclosing": None,
        "selector": None,
    }


def _set_field(name: str, value: object, field_type: str = "str") -> dict[str, object]:
    return {
        "name": name,
        "field_type": field_type,
        "codec_version": 1,
        "op": "set",
        "value": {"kind": field_type, "value": value},
    }


def test_open_stream_writer_appends_and_folds(tmp_path: Path) -> None:
    store = DatagenStore.open(str(tmp_path / "log"))

    writer = store.open_stream(
        "5", run_id="run-1", writer_epoch="writer-1", query_tags={"lang": "en"}
    )
    assert isinstance(writer, DatagenStreamWriter)
    assert writer.item_id == "5"
    assert writer.attempt == 0

    checkpoint = writer.step_completed(
        _leaf_position("gen", 0), [_set_field("draft", "v1")]
    )
    store.append_checkpoint(checkpoint)
    store.append([writer.item_terminal("completed")])

    folded = store.fold_item("5")
    assert folded is not None
    assert folded["status"] == "completed"
    assert folded["fields"]["draft"] == {
        "mode": "set",
        "value": {"kind": "str", "value": "v1"},
    }
    assert folded["query_tags"] == {"lang": "en"}


def test_resume_stream_bumps_attempt(tmp_path: Path) -> None:
    store = DatagenStore.open(str(tmp_path / "log"))
    writer = store.open_stream("5", run_id="run-1", writer_epoch="writer-1")
    store.append_checkpoint(
        writer.step_completed(_leaf_position("gen", 0), [_set_field("draft", "v1")])
    )

    resumed = store.resume_stream("5", run_id="run-1", writer_epoch="writer-2")
    assert resumed is not None
    assert resumed.attempt == 1

    store.append_checkpoint(
        resumed.step_completed(_leaf_position("gen", 0), [_set_field("draft", "v2")])
    )
    store.append([resumed.item_terminal("completed")])

    folded = store.fold_item("5")
    assert folded is not None
    assert folded["last_attempt"] == 1
    assert folded["fields"]["draft"] == {
        "mode": "set",
        "value": {"kind": "str", "value": "v2"},
    }


def test_resume_stream_none_when_never_started(tmp_path: Path) -> None:
    store = DatagenStore.open(str(tmp_path / "log"))
    assert store.resume_stream("9", run_id="run-1", writer_epoch="writer-1") is None


def test_item_tree_links_parent_and_child(tmp_path: Path) -> None:
    store = DatagenStore.open(str(tmp_path / "log"))

    root = store.open_stream("9", run_id="run-1", writer_epoch="writer-1")
    store.append([root.item_terminal("completed")])

    child = store.open_stream(
        "9/expand:0", run_id="run-1", writer_epoch="writer-1", parent_item_id="9"
    )
    store.append([child.item_terminal("completed")])

    tree = store.item_tree("9")
    assert tree["roots"] == ["9"]
    root_node = tree["nodes"]["9"]
    assert root_node["item"]["status"] == "completed"
    assert root_node["children"] == ["9/expand:0"]
    child_node = tree["nodes"]["9/expand:0"]
    assert child_node["item"]["parent_item_id"] == "9"
    assert child_node["children"] == []


def test_load_blob_by_field_name(tmp_path: Path) -> None:
    store = DatagenStore.open(str(tmp_path / "log"))
    writer = store.open_stream("5", run_id="run-1", writer_epoch="writer-1")

    blob_field = {
        "name": "screenshot",
        "field_type": "blob",
        "codec_version": 1,
        "op": "set",
        "value": {"kind": "blob", "bytes": b"payload", "size": 7},
    }
    checkpoint = writer.step_completed(_leaf_position("shot", 0), [blob_field])
    store.append_checkpoint(checkpoint)
    store.append([writer.item_terminal("completed")])

    folded = store.fold_item("5")
    assert folded is not None
    assert store.load_blob(folded, "screenshot") == b"payload"
    # A non-blob / absent field resolves to None.
    assert store.load_blob(folded, "missing") is None


def test_item_failed_is_failure_lens_not_terminal(tmp_path: Path) -> None:
    store = DatagenStore.open(str(tmp_path / "log"))
    writer = store.open_stream("5", run_id="run-1", writer_epoch="writer-1")
    store.append(
        [writer.item_failed(_leaf_position("gen", 0), "ValueError", error_dump="boom")]
    )

    failures = store.item_failures("5")
    assert len(failures) == 1
    assert failures[0]["error_type"] == "ValueError"

    # FAILED does not terminate the item.
    folded = store.fold_item("5")
    assert folded is not None
    assert folded["status"] == "running"
