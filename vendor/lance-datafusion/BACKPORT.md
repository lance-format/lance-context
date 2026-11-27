# Caller-owned execution runtime

This directory starts from the published `lance-datafusion 14.0.2` crate.
Archive SHA256: `2df1f26d0a01b1103d587bd46226979b5a03b1117f63d6fb0503a496708a6e38`.
Upstream commit: `40f15ed1b71b66950961b175829fdf17a763f108`.

`LanceExecutionOptions::runtime_env` accepts the caller's existing DataFusion
runtime. Sessions using it bypass the scalar-configuration cache, retain the
supplied memory pool and disk manager, and keep the existing execution plan,
partition coalescing, ordering, statistics, and spill configuration. The
`mem_pool_size` option still configures sort spill headroom; it does not resize
a supplied pool. Native decoder/index allocations and Lance's own spill store
remain separate.

`provider_to_stream_with_context` carries the same caller resources through
provider adaptation. The existing helper retains its default context.

The Cargo manifest uses the sibling patched `lance-core` and excludes Cargo's
reserved provenance filenames during sdist repackaging. Original provenance
and manifest files are retained.
