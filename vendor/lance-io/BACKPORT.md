# Lance 13 multipart completion retry backport

This directory starts from the published `lance-io 13.0.0` crate, whose archive
SHA256 is `224730479e781d24b81a0afecfe230b5b21a86a4f767ba7655b974402d0c67c5`.
Its upstream source commit is `0923349baa19fd9af776b7d4626dc758dc610c80`.

The root Cargo patch selects this package for all transitive users of
`lance-io 13.0.0`. Other Lance packages continue to resolve from crates.io at
their existing versions. Do not upgrade the whole Lance dependency graph to a
beta release just to obtain this fix.

## Correction

Backport [lance-format/lance#9913](https://github.com/lance-format/lance/pull/9913),
commit `3e8360c77f578516a770799f31d72d51c90746b7`:

- Call `MultipartUpload::complete` once. Azure and S3 consume the recorded part
  IDs before sending the completion request, so calling it again can replace a
  transient storage error with `Missing part`.
- Retry the same serialized completion request in the HTTP connector, preserving
  the part list and options. Retry limits and AIMD throttling still apply.
- Parse XML error documents instead of treating a successful S3 response whose
  object key contains a throttle word as a failure.

The backport preserves Lance 13's connector names and configuration. It does not
include later changes to delete throttling. `quick-xml` is enabled only with the
existing cloud storage features.

## Local changes to the published crate

- `src/object_store/throttle.rs`: the backport and an Azure SDK regression with
  36 parts, covering successful completion, one 503 followed by success, and
  exhausted 503 retries. It checks byte-identical block lists, retained headers
  and URI, no part re-upload, and preservation of the original storage error.
- `Cargo.toml`: optional `quick-xml` dependency, cloud feature wiring, and
  exclusion of Cargo's reserved provenance filenames when repackaging.
- `Cargo.lock`: standalone regression-test dependency resolution.
- `LICENSE`: upstream Apache 2.0 license, copied from the Lance repository.
- This provenance document. `Cargo.toml.orig` is retained as an unmodified
  record of the published crate's original workspace manifest.

The cloud regressions use real object_store SDKs with a fault-injecting HTTP
transport; they do not contact Azure or S3. Run them with:

```sh
cargo test --locked --manifest-path vendor/lance-io/Cargo.toml --lib object_store::throttle::tests
cargo +1.98.1 clippy --locked --manifest-path vendor/lance-io/Cargo.toml --all-targets -- -D warnings
```

The tests also run on the context repository's current stable Rust. Linting the
unchanged upstream source uses Lance 13's pinned Rust 1.98.1; newer Clippy reports
`double_must_use` inside its existing `async_trait` macro expansions. The Python
sdist includes this patch directory through core's explicit path dependency so
wheel rebuilds use the same source as repository builds. CI also checks that the
archive contains the byte-identical patched throttle source.

Remove this patch after a compatible published Lance release includes the fix
and the same regressions pass against that release.
