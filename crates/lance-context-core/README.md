# lance-context-core

Pure Rust engine for the lance-context project. This crate is free of Python
dependencies and is re-exported by the `lance-context` wrapper crate.

For a generic store, `store.get_many(&ids, Some(&["id".into()])).await?`
checks up to 1,024 IDs without fetching payloads or issuing a disk lookup per
ID. Found rows are returned once in first-requested order, including flushed
WAL updates. See [batch lookup](../../docs/benchmarks/batch-get.md) for the HTTP
and Rust client APIs, visibility rules, and a reproducible benchmark.

## Application-owned schemas

Use `StorageBase` when an application owns its Arrow schema and needs the shared
WAL, consistent-read, merge, and indexing machinery. It accepts a custom merge
key instead of requiring a built-in record type. Domain events, their ordering
and validation, and application HTTP APIs remain in the consuming crate.

Start with `StorageBaseOptions::new(schema, "entry_key")`, set a distinct, stable
`shard_id` for each live writer, and call `StorageBase::open` (or `from_dataset`
for an already-open dataset). Share a bounded `Session` through the options when
opening many stores. The options default to `seal_on_put = true`: a successful
nonempty `put` durably writes and publishes its rows before returning. Setting
this to `false` defers visibility until an explicit or background flush.

Read through `read_consistent(ListSource::All, ...)`, building the scan with
`lsm_scanner_for_dataset` from the dataset and shard snapshots passed into the
callback. Complete the read inside the callback: await `try_collect` or fully
consume the stream before returning. Do not return a lazy scanner, stream, or
future, or retain the captured dataset or shard snapshots for later reads.
Read protection and retries do not cover reads performed after the callback
returns. The callback may run more than once, so keep it free of business side
effects.

This includes published WAL and refreshes the base/WAL view together;
scanning `dataset()` alone sees only base fragments. Use `key_column()` when
building the scanner. The dataset, shard identity and credentials cannot be
replaced through public fields after opening a store.

`flush()` preserves the existing publication API. `flush_with_outcome()` also
reports `Sealed`, `Noop`, or `RecoveredFence` for callers that need to distinguish
maintenance results. These outcomes are not per-request receipts or publication
watermarks, and do not resolve an unknown append result by themselves. Application
retries and writer handoff still need their own protocol; a writer identity does
not provide distributed ownership. Close writers gracefully with `close()`.

The [external-consumer test](tests/storage_base_api.rs) demonstrates a custom
key and binary payloads, a reader opened before writes, identical retries,
filtered projections, base/WAL reads across merge, reopening, and deferred
publication without using any built-in application schema.

Enable the optional `lance-io-metrics` feature to record Lance 13 object-store
request counts, bytes, durations, errors, in-flight requests and cloud retryable
responses through the application's existing `metrics` recorder. Embedded users
can call `metrics::describe_lance_io_metrics()` after installing their recorder.
The REST server forwards this feature and registers descriptions automatically:

```sh
cargo build -p lance-context-server --features lance-io-metrics
```

Storage metrics then appear on the server's existing `/metrics` endpoint. Names
start with `lance_object_store_`; request metrics have an `operation` label and
use only the storage scheme as the default `base` label. Keep
`LANCE_OBJECT_STORE_METRICS_LABEL=scheme` for bounded cardinality; Lance's `full`
and `dataset` modes include storage identities and can create many more series.
This feature is off by default. Cargo features are additive, so enabling it in
one dependency enables Lance instrumentation throughout that binary.

Request durations can overlap across uploads and include retries; their sum is
neither elapsed ingestion time nor encoder CPU time. Local optimized writes
measure the writer lifetime, including encoding between writes. Retryable HTTP
response metrics cover native cloud stores, not every custom backend. These
signals complement application phase timings; they do not alone establish
network saturation or isolate encoding cost. See the [Lance metric catalog](https://docs.rs/lance/13.0.0/lance/metrics/index.html).
