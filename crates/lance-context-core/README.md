# lance-context-core

Pure Rust engine for the lance-context project. This crate is free of Python
dependencies and is re-exported by the `lance-context` wrapper crate.

For a generic store, `store.get_many(&ids, Some(&["id".into()])).await?`
checks up to 1,024 IDs without fetching payloads or issuing a disk lookup per
ID. Found rows are returned once in first-requested order, including flushed
WAL updates. See [batch lookup](../../docs/benchmarks/batch-get.md) for the HTTP
and Rust client APIs, visibility rules, and a reproducible benchmark.
