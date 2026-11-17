import pytest
from lance_context import VariantDocumentStore


def test_variant_streaming_uses_variant_backend_and_preserves_partial_commits(tmp_path):
    store = VariantDocumentStore.create(tmp_path / "stream.lance")

    def documents():
        yield {"body": "otter zero"}
        yield {"body": "otter one"}
        raise RuntimeError("source stopped")

    batches = store.add_iter(documents(), max_batch_rows=1)
    first = next(batches)
    assert store.get(first[0]) == {"body": "otter zero"}
    # The next source failure must not undo the batch already committed.
    with pytest.raises(RuntimeError, match="source stopped"):
        list(batches)
    reopened = VariantDocumentStore.open(tmp_path / "stream.lance")
    assert reopened.get(first[0]) == {"body": "otter zero"}
