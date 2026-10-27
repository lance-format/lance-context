# Lance 13 scoped CPU task joining

This directory starts from the published `lance-core 13.0.0` crate. Its archive
SHA256 is `d3e377ee5bc5118efe5c321ecff0a07ac8f56db6bed16e31304e7d876ddcd96c`
and upstream commit is `0923349baa19fd9af776b7d4626dc758dc610c80`.
The original workspace manifest and Cargo VCS provenance are retained.
`LICENSE` is the upstream Apache 2.0 license, as in the vendored lance-io crate.

The local source delta is confined to `src/utils/tokio.rs`. The opt-in
`join_cpu_tasks` future waits for actual `spawn_cpu` closures submitted while
polling its work, even if a writer drops their handles on an encoding or IO
error. A guard enters the group before submission and exits only when the
blocking closure exits or unwinds. Blocking-thread context propagates the same
group into nested CPU submissions. Independent groups do not join each other's
children. The original work's output or panic is returned after joining; the
existing `spawn_cpu` panic propagation and unscoped behavior are preserved.

The join future must itself be driven to completion. Rocketkeep runs it inside
an owned preparation child that maintenance scope drain joins. The memory hold
lives outside the inner writer future, so an error that drops that writer's
reservation cannot release its final charge before CPU children finish. These
holds retain the existing accounting; they do not model every encoder or IO
allocation or establish a hard RSS bound. Arbitrary async tasks do not inherit
Tokio task locals and must be joined separately. No global waiting, timeout-based
memory release, or permanent memory quarantine is introduced.

Other changes: `Cargo.toml` excludes Cargo's reserved provenance filename when
repackaging; this document and `LICENSE` are added. No other Lance source,
format, version, CPU pool sizing, or runtime configuration is changed. The root
workspace patch selects this exact package for transitive users; core's direct
path dependency includes it in source distributions, as with lance-io.

The context integration tests use the actual Lance CPU runtime. They cover a
discarded CPU handle followed by writer error/panic, caller cancellation,
nested CPU submission, independent scopes, original CPU panic propagation, and
reservation growth/shrinkage/oversized final accounting. Run with:

```sh
cargo test -p lance-context-core -p lance-context-master --lib staged_wal -- --test-threads=1
cargo test -p lance-context-core -p lance-context-master --lib merge_budget -- --test-threads=1
```

This is a local patch pending upstream review. Remove it when a compatible
published Lance release supplies equivalent owned CPU lifetime semantics and
the same regressions pass against that release.

## Async children and shared IO requests

`scope_task` captures membership before an async child is submitted, restores
the task-local group while it runs, and destroys its future before releasing
membership even if cancelled before its first poll. `spawn_scoped` uses it with
Tokio; existing third-party abort-on-drop wrappers use the future directly.
`join_cpu_tasks` now joins both kinds of children on success, error, or panic.

`retain_task_scope` supports shared services that queue/batch callers from
independent groups. The blob reader attaches one token to each logical request;
the standard physical IO scheduler retains one per batch through all physical
reads and result cleanup; the lite scheduler retains one per task through its
actual state destruction. Neither a shared scheduler's lifetime nor all future
requests are attributed to its first caller.

The corresponding exact-version call-site patches are in `../lance` and
`../lance-encoding`; the existing `../lance-io` patch adds request tokens. They
are required together. An unmodified arbitrary `tokio::spawn` elsewhere still
does not inherit task locals. This does not account for every allocator or
transport buffer and is not a hard RSS bound.

`spawn_blocking_scoped` retains the same ownership through actual local IO
closures on Tokio's ordinary blocking pool. It preserves that pool and its
JoinHandle semantics, propagates nested CPU context, and destroys queued
closure captures before releasing membership. Existing CPU work remains on
Lance's dedicated CPU pool. Neither wrapper releases ownership merely because
its receiving handle was dropped.
