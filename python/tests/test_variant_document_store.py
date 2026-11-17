"""The standard Variant backend remains distinct from the baseline."""

import pytest
from lance_context import DocumentStore, VariantDocumentStore


def test_variant_layout_migration_reopen_snapshots_and_search(tmp_path):
    uri = tmp_path / "variant.lance"
    store = VariantDocumentStore.create(uri, policy={"max_columns": 2})
    original = {"a": 1, "b": 2}
    old_id = store.add([original])[0]
    version = store.version()
    document = {"body": "lateotter " * 10000, "a": "text", "tags": ["residualword"]}
    new_id = store.add([document])[0]
    assert ["body"] in [field["path"] for field in store.layout()["fields"]]
    assert store.get(old_id) == original
    assert store.get(new_id) == document
    assert store.get_path(new_id, ["tags"]) == ["residualword"]
    assert store.search("lateotter")[0]["id"] == new_id
    assert store.search("residualword")[0]["id"] == new_id
    store.checkout(version)
    assert store.get(old_id) == original
    with pytest.raises(KeyError):
        store.get(new_id)
    store.refresh()
    assert VariantDocumentStore.open(uri).get(new_id) == document
    with pytest.raises(Exception, match="format"):
        DocumentStore.open(uri)


def test_variant_preserves_json_boundaries_and_literal_keys(tmp_path):
    store = VariantDocumentStore.create(tmp_path / "boundaries.lance")
    documents = [{"a.b": {"": "kept"}, "integer": 2**64 - 1}, {"a.b": None}, {}, None]
    ids = store.add(documents)
    for record_id, document in zip(ids, documents):
        assert store.get(record_id) == document
    assert store.get_path(ids[0], ["a.b", ""]) == "kept"
    assert store.get_path(ids[1], ["a.b"]) is None
    with pytest.raises(KeyError):
        store.get_path(ids[2], ["a.b"])
    assert store.filter_eq(["a.b"], None) == [ids[1]]
