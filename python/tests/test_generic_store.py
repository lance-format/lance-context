"""Stores over a user-declared schema."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from lance_context import GenericStore

SCHEMA = {
    "id": {"type": "string", "nullable": False},
    "user": "string",
    "score": "float32",
    "tags": {"type": "list", "item": {"type": "string"}},
    "embedding": {"type": "vector", "dim": 3, "metric": "cosine"},
    "payload": {"type": "binary", "blob": True},
}


def open_store(tmp_path: Path, name: str = "s", **kwargs) -> GenericStore:
    # seal_on_add so each test reads its own writes without an explicit flush;
    # the deferred default is exercised separately below.
    kwargs.setdefault("seal_on_add", True)
    return GenericStore.open(str(tmp_path / f"{name}.lance"), schema=SCHEMA, **kwargs)


def test_declared_schema_round_trips(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.add(
        [
            {
                "id": "r1",
                "user": "u1",
                "score": 0.5,
                "tags": ["a", "b"],
                "embedding": [1.0, 2.0, 3.0],
            },
            # Every nullable column may be omitted.
            {"id": "r2"},
        ]
    )

    rows = store.list()
    assert len(rows) == 2
    first = store.get("r1")
    assert first is not None
    assert first["user"] == "u1"
    assert first["tags"] == ["a", "b"]
    assert first["embedding"] == [1.0, 2.0, 3.0]

    second = store.get("r2")
    assert second is not None
    # Nulls are omitted rather than returned as None.
    assert set(second) == {"id"}

    assert store.get("missing") is None


def test_schema_reports_declared_columns(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    names = [name for name, _ in store.schema()["columns"]]
    assert names == ["id", "user", "score", "tags", "embedding", "payload"]


def test_id_column_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="id"):
        GenericStore.open(str(tmp_path / "bad.lance"), schema={"name": "string"})


def test_undeclared_column_is_rejected_not_dropped(tmp_path: Path) -> None:
    # A field-name typo must fail the write rather than becoming missing data.
    store = open_store(tmp_path)
    with pytest.raises(Exception, match="not declared"):
        store.add([{"id": "r1", "usr": "typo"}])


def test_type_mismatch_is_rejected(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    with pytest.raises(Exception):
        store.add([{"id": "r1", "score": "not-a-number"}])


def test_vector_length_must_match(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    with pytest.raises(Exception, match="vector of 3"):
        store.add([{"id": "r1", "embedding": [1.0, 2.0]}])


def test_blob_excluded_from_list_but_fetchable_per_row(tmp_path: Path) -> None:
    # The cost model for large payloads: bulk reads skip them, point reads ask.
    store = open_store(tmp_path)
    store.add([{"id": "r1", "payload": [1, 2, 3]}])

    listed = store.list()
    assert "payload" not in listed[0]

    fetched = store.get("r1", columns=["payload"])
    assert fetched is not None
    assert base64.b64decode(fetched["payload"]) == bytes([1, 2, 3])


def test_large_blob_round_trips(tmp_path: Path) -> None:
    # Multi-megabyte inline payloads are the point of these stores.
    store = open_store(tmp_path)
    payload = bytes((i % 251) for i in range(2 * 1024 * 1024))
    store.add([{"id": "big", "payload": list(payload)}])

    fetched = store.get("big", columns=["payload"])
    assert fetched is not None
    assert base64.b64decode(fetched["payload"]) == payload


def test_filter_and_paging(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.add(
        [
            {"id": f"r{i}", "user": "even" if i % 2 == 0 else "odd", "score": float(i)}
            for i in range(6)
        ]
    )

    evens = store.list(filter="user = 'even'")
    assert len(evens) == 3
    assert all(row["user"] == "even" for row in evens)

    page = store.list(limit=2, offset=1)
    assert len(page) == 2


def test_duplicate_id_keeps_the_newest_write(tmp_path: Path) -> None:
    # `id` is the merge key, so re-adding one supersedes the older row.
    store = open_store(tmp_path)
    store.add([{"id": "r1", "user": "first"}])
    store.add([{"id": "r1", "user": "second"}])

    rows = store.list()
    assert len(rows) == 1
    assert rows[0]["user"] == "second"


def test_deferred_seal_needs_a_flush(tmp_path: Path) -> None:
    store = GenericStore.open(
        str(tmp_path / "deferred.lance"), schema=SCHEMA, seal_on_add=False
    )
    store.add([{"id": "r1"}])
    assert store.list() == []

    store.flush()
    assert len(store.list()) == 1


def test_wal_generations_merge_into_the_base_table(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    for i in range(3):
        store.add([{"id": f"r{i}"}])

    assert store.cleanup_wal() > 0
    assert len(store.list()) == 3


def test_reopen_reads_the_persisted_schema(tmp_path: Path) -> None:
    uri = str(tmp_path / "persist.lance")
    store = GenericStore.open(uri, schema=SCHEMA, seal_on_add=True)
    store.add([{"id": "r1", "user": "u1"}])
    del store

    reopened = GenericStore.open_existing(uri, seal_on_add=True)
    assert [name for name, _ in reopened.schema()["columns"]] == list(SCHEMA)
    assert len(reopened.list()) == 1


def test_reopening_with_a_conflicting_schema_is_rejected(tmp_path: Path) -> None:
    # Reinterpreting existing data under a new schema is the failure to prevent.
    uri = str(tmp_path / "conflict.lance")
    GenericStore.open(uri, schema=SCHEMA, seal_on_add=True)

    with pytest.raises(Exception, match="different schema"):
        GenericStore.open(uri, schema={**SCHEMA, "extra": "int64"}, seal_on_add=True)


def test_shorthand_and_full_type_forms_agree(tmp_path: Path) -> None:
    shorthand = GenericStore.open(
        str(tmp_path / "short.lance"),
        schema={"id": {"type": "string", "nullable": False}, "n": "int64"},
        seal_on_add=True,
    )
    shorthand.add([{"id": "a", "n": 7}])
    assert shorthand.list()[0]["n"] == 7


def test_seal_mode_survives_a_reopen(tmp_path: Path) -> None:
    # `seal_on_add` is a property of the store, not of whoever opens it. It used
    # to live only in the open options, so a reopened store silently reverted to
    # the caller's default and lost read-your-write with no error.
    uri = str(tmp_path / "sealed.lance")
    store = GenericStore.open(uri, schema=SCHEMA, seal_on_add=True)
    store.add([{"id": "r1"}])
    del store

    # Reopen with the opposite default: the persisted value must win.
    reopened = GenericStore.open_existing(uri, seal_on_add=False)
    reopened.add([{"id": "r2"}])
    assert len(reopened.list()) == 2, (
        "a store created with seal_on_add must keep it across a reopen"
    )


def test_deferred_seal_mode_survives_a_reopen(tmp_path: Path) -> None:
    uri = str(tmp_path / "deferred.lance")
    GenericStore.open(uri, schema=SCHEMA, seal_on_add=False)

    reopened = GenericStore.open_existing(uri, seal_on_add=True)
    reopened.add([{"id": "r1"}])
    assert reopened.list() == [], "a deferred-seal store must stay deferred"

    reopened.flush()
    assert len(reopened.list()) == 1
