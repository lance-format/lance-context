from lance_context import DocumentStore


def test_prepare_point_reads_is_repeatable_and_snapshot_safe(tmp_path):
    store = DocumentStore.create(tmp_path / "point-index.lance")
    ids = store.add([{"body": "otter", "metadata": {"counter": 1}}])
    version = store.version()
    store.prepare_point_reads()
    assert store.version() == version
    store.checkout(version)
    store.prepare_point_reads()
    assert store.get_path(ids[0], ["metadata"]) == {"counter": 1}
