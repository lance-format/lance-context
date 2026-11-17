import pytest
from lance_context import DocumentStore


def test_automatic_selection_reconstruction_and_reopen(tmp_path):
    uri = tmp_path / "documents.lance"
    store = DocumentStore.create(uri, policy={"max_columns": 2})
    body = "large searchable otter document " * 100_000
    document = {"body": body, "metadata": {"source": "wiki"}, "null": None}
    record_id = store.add([document])[0]
    assert ["body"] in [field["path"] for field in store.layout()["fields"]]
    assert store.get(record_id) == document
    reopened = DocumentStore.open(uri)
    assert reopened.get_path(record_id, ["body"]) == body
    assert reopened.get_path(record_id, ["null"]) is None
    with pytest.raises(KeyError):
        reopened.get_path(record_id, ["missing"])
    with pytest.raises(KeyError):
        reopened.get("missing")
    assert reopened.get(reopened.add([None])[0]) is None


def test_evolving_fields_filter_and_text_search(tmp_path):
    store = DocumentStore.create(
        tmp_path / "evolution.lance", policy={"min_frequency": 1.0}
    )
    older = store.add([{"body": "otter swims"}, {}])
    assert not store.layout()["fields"]
    old_version = store.version()
    newer = store.add([{"body": "otter dives"}])
    assert store.filter_eq(["body"], "otter swims") == [older[0]]
    assert {hit["id"] for hit in store.search("otter")} == {older[0], newer[0]}
    assert store.get(older[0]) == {"body": "otter swims"}
    store.checkout(old_version)
    assert not store.layout()["fields"]
    with pytest.raises(KeyError):
        store.get(newer[0])
    with pytest.raises(RuntimeError, match="pinned"):
        store.add([{}])
    store.refresh()
    assert store.get(newer[0]) == {"body": "otter dives"}


def test_literal_keys_numeric_precision_and_input_errors(tmp_path):
    store = DocumentStore.create(tmp_path / "types.lance")
    document = {
        "a.b": {"/~": ""},
        "integer": 2**64 - 1,
        "escaped": 'quote " newline\n tab\t backslash \\ 中文 🦦',
        "residual": [51.248178375505404, 2.0030397744267762e-253],
    }
    record_id = store.add([document])[0]
    assert store.get(record_id) == document
    assert store.get_path(record_id, ["a.b", "/~"]) == ""
    with pytest.raises(TypeError):
        store.get_path(record_id, "a.b")
    with pytest.raises(TypeError):
        store.add({"body": "not a batch"})
    with pytest.raises(ValueError):
        DocumentStore.create(tmp_path / "bad.lance", policy={"unknown": True})
    with pytest.raises(ValueError):
        DocumentStore.create(tmp_path / "bad-limit.lance", policy={"max_columns": 0})


@pytest.mark.parametrize("value", [2**64, -(2**63) - 1, float("nan"), float("inf")])
def test_reject_lossy_numbers_before_mutation(tmp_path, value):
    store = DocumentStore.create(tmp_path / "numbers.lance")
    version = store.version()
    with pytest.raises(ValueError):
        store.add([{"body": "valid"}, {"number": value}])
    assert store.version() == version
    assert not store.layout()["fields"]


def test_reject_non_string_keys_and_cycles(tmp_path):
    store = DocumentStore.create(tmp_path / "invalid.lance")
    with pytest.raises(TypeError, match="keys must be strings"):
        store.add([{1: "number key", "1": "string key"}])
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="nesting"):
        store.add([cycle])


def test_search_covers_late_text_and_type_changes_after_budget_exhaustion(tmp_path):
    store = DocumentStore.create(tmp_path / "search-coverage.lance")
    store.add([{f"field{i}": i for i in range(32)}])
    document = {"body": "lateotter " * 8192, "field0": "typedrift"}
    record_id = store.add([document])[0]
    for query in ["LATEOTTER", "typedrift", "missing lateotter"]:
        assert store.search(query) == [{"id": record_id, "score": 0.0}]
    assert store.get(record_id) == document
    reopened = DocumentStore.open(tmp_path / "search-coverage.lance")
    assert reopened.search("lateotter")[0]["id"] == record_id


def test_residual_search_only_matches_string_values(tmp_path):
    store = DocumentStore.create(
        tmp_path / "residual-search.lance", policy={"max_columns": 1}
    )
    store.add([{"slot": 1}])
    ids = store.add(
        [{"keyonly": 987654, "body": "mixed\nquoted"}, ["arrayotter"], "roototter"]
    )
    assert store.search("MIXED")[0]["id"] == ids[0]
    assert store.search("arrayotter")[0]["id"] == ids[1]
    assert store.search("roototter")[0]["id"] == ids[2]
    for query in ["keyonly", "987654", "nquoted", "!!!"]:
        assert store.search(query) == []
