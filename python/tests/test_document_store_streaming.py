import pytest
from lance_context import DocumentStore


def test_add_iter_is_lazy_and_commits_each_yielded_batch(tmp_path):
    store = DocumentStore.create(tmp_path / "stream.lance")
    consumed = []

    def documents():
        for i in range(5):
            consumed.append(i)
            yield {"sequence": i}
        raise RuntimeError("source failed")

    iterator = store.add_iter(documents(), max_batch_rows=2)
    assert consumed == []
    first = next(iterator)
    assert consumed == [0, 1]
    second = next(iterator)
    assert consumed == [0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="source failed"):
        next(iterator)
    assert [store.get(i)["sequence"] for i in first + second] == [0, 1, 2, 3]
    assert store.filter_eq(["sequence"], 4) == []


def test_add_iter_byte_targets_oversized_values_and_validation(tmp_path):
    store = DocumentStore.create(tmp_path / "stream-bytes.lance")
    docs = [{"body": "otter" * 100}, {"body": "large" * 10000}, {"body": "last"}]
    batches = list(store.add_iter(iter(docs), max_batch_bytes=1024))
    assert [len(batch) for batch in batches] == [1, 1, 1]
    assert [store.get(batch[0]) for batch in batches] == docs
    assert list(store.add_iter([])) == []
    for limit in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            list(store.add_iter([], max_batch_bytes=limit))
    version = store.version()
    with pytest.raises(ValueError, match="finite"):
        list(store.add_iter([{"body": "pending"}, float("nan")]))
    assert store.version() == version
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="nesting"):
        list(store.add_iter([cycle]))
