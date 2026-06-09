import lance_context as lc


def test_context_create_and_add():
    ctx = lc.Context.create("memory://test")
    assert ctx.uri() == "memory://test"
    assert ctx.branch() == "main"
    assert ctx.entries() == 0

    ctx.add("user", "hello")
    ctx.add("assistant", 123, data_type="text/plain")
    assert ctx.entries() == 2
    version = ctx.version()
    assert isinstance(version, int)
    ctx.checkout(version)


def test_context_snapshot_and_fork():
    ctx = lc.Context.create("memory://test")
    ctx.add("user", "hello")

    snap = ctx.snapshot()
    assert snap == "snapshot-1"

    labeled = ctx.snapshot(label="pre_run")
    assert labeled == "pre_run"

    fork = ctx.fork("branch-a")
    assert fork.branch() == "branch-a"
    assert fork.entries() == ctx.entries()
    assert fork.version() == ctx.version()


def test_context_delete_and_forget_by_external_id(tmp_path):
    ctx = lc.Context.create(str(tmp_path / "context.lance"))
    ctx.add("user", "stale", external_id="doc-1#chunk-1")

    entry = ctx.get(external_id="doc-1#chunk-1")
    assert entry is not None

    assert ctx.delete(external_id="doc-1#chunk-1") is True
    assert ctx.get(external_id="doc-1#chunk-1") is None
    assert ctx.get(id=entry["id"]) is None
    assert ctx.forget(external_id="doc-1#chunk-1") is False
