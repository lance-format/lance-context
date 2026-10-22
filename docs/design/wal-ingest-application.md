# Durable ingestion application

The optional `service` feature builds `lance-context-ingest`. It composes the
production TLS WAL client, HTTP gateway, table initializer, etcd materializer
worker, and managed Lance reader. Its deployment-controlled JSON registry is a
fixed set of authorized table generations. It does not yet discover namespace
changes dynamically or migrate legacy tables.

Build with `cargo build -p lance-context-ingest --features service --bin
lance-context-ingest`. Run one role per process:

- `bootstrap CONFIG`: initialize new tables using the persisted bootstrap IDs.
  Retry the same configuration after an unknown outcome; do not generate new IDs.
- `ingress CONFIG`: serve HTTP with a shared WAL connection pool, bounded gateway
  admission, and shared ingress memory. Every request requires the workload bearer
  token, including health checks.
- `worker CONFIG`: claim tables through etcd, fence publishers with durable WAL
  epochs, materialize into the configured storage, and drain on SIGTERM/SIGINT.
  Multiple workers may use the same table set and claim prefix.
- `inspect CONFIG TABLE`: read the catalog-selected version and count rows without
  acquiring a publisher epoch. This is a diagnostic command, not a scan benchmark.

`Config` and its nested types in `application/config.rs` define the JSON schema.
Unknown fields are rejected. Durations use Serde's `{ "secs": N, "nanos": N }`
representation. RPC and worker options have bounded defaults. TLS roots, client
certificate/key, HTTP bearer token, and receipt key files resolve relative to the
configuration file. Receipt keys have the form `{ "active": 1, "keys": { "1":
[...32 bytes...] } }`; keep them stable across ingress replicas and restarts.
Storage URIs/options use Lance's normal object-store and workload identity support.
The same immutable layout and complete WAL limits must be used by replicas and
clients. Explicit memory budgets account for the logical data path; they are not
an operating-system RSS cap.

WAL RPCs use mutual TLS, root verification and per-endpoint certificate pins.
The HTTP listener expects TLS termination and connection limits at the edge proxy;
the workload token authorizes the complete configured table set. Configure the
proxy's narrower end-user access policy and protect the internal listener. No
upload can select a destination URI or bootstrap identity. The etcd connection
optionally uses a CA and a client identity. Configuration and secret contents are
not printed by the program.

Startup validates every configured generation through the durable catalog and
managed reader. It does not acquire publication ownership or silently initialize
missing tables. HTTP shutdown stops gateway admission and joins admitted handlers;
worker shutdown joins submitted file work before releasing its own claims. Budget
termination grace for bounded RPCs and object-store retries. SIGKILL recovery uses
the original durable identities and exact producer bytes/sequence, not graceful
shutdown assumptions.

The qualification test runs the built executable for bootstrap, ingress and two
workers against real TLS WAL nodes and etcd. It kills/restarts ingress, checks exact
retry identity, waits for publication, drains workers, and compares every Arrow
column through a managed reader. It is explicitly ignored in ordinary tests and
requires `INGEST_TEST_BIN`, `INGEST_TEST_ARTIFACTS`, and `ETCD_TEST_ENDPOINTS`.
Run the executable test with `cargo test -p lance-context-ingest --features
service,http-client --lib tests::application -- --ignored` after building the
application binary. The CI workflow runs this test against etcd and retains
application logs on failure. Worker observations include a bounded last-failure
diagnostic, and process timeouts identify retained logs. This test does not
establish sustained throughput or Kubernetes behavior; all-worker failure,
dynamic namespace discovery and sustained scaling need separate qualification.
