# Independent WAL node service

`context-wal-node` starts the durable runtime, authenticated peer replication,
separate snapshot transfer, and separate client data/control listeners. It also
runs automatic checkpoint, image collection, disk measurement and native log
maintenance. Build the service without Arrow or Lance dependencies:

```sh
cargo build --release -p lance-context-wal --features service --bin context-wal-node
context-wal-node --config /etc/wal/node.json --io-threads 2 --blocking-threads 4
```

The service supplies the storage side of ingestion. HTTP ingestion and Lance
materialization are separate application components; this executable does not
publish rows into Lance by itself.

## Configuration and identity

The JSON configuration is `service::Config`; unknown top-level and transport
option fields are rejected. `tests/node_service/fixture.rs` builds a complete
configuration for six real processes in two independent three-voter groups.

Required fields are the node ID, data directory, complete immutable `layout` and
replication `limits`, advertised `endpoints`, local `listen` addresses, TLS file
paths, trusted application certificate pins, per-group signing file paths and
node disk budget. `storage`, `runtime`, `peer`, `snapshot`, `client` and
`maintenance` have bounded defaults and accept explicit overrides. Durations in
transport/runtime JSON are integer milliseconds; memory/disk fields are bytes.
All paths relative to a configuration file resolve against its containing
directory. The executable reads at most 1 MiB of configuration.

Each advertised node has distinct peer, snapshot, data and control addresses, a
TLS server name and a SHA-256 leaf-certificate pin. Addresses are stable service
names; changing the durable layout is a migration. Listeners may bind wildcard
addresses, but must have distinct, nonzero ports. Only voters sharing a hosted
group have peer queues and certificate authorization on a physical node; adding
unrelated groups does not create a full mesh of peer tasks.

TLS certificate chains and root bundles use PEM, as does the local private key.
Catalog/source attestation files contain Ed25519 PKCS#8 DER. Existing persistent
keys must match the public keys in the immutable replication policy. The local
TLS certificate is checked against its configured pin, DNS name, roots and
private key before opening storage. Startup never generates replacement keys,
producer policy or a new layout. Catalog keys are supplied only to catalog
replicas; source private keys cover exactly the groups hosted by each node.

Provision files through per-node mounted configuration and secrets. A node
requires its original persistent directory across process/Pod replacement.
For Kubernetes, use stable DNS, persistent volumes, and a termination grace
period sufficient for the largest native maintenance operation. Startup must
permit all voters to resolve each other before readiness; a headless service
normally needs `publishNotReadyAddresses: true`. Configure one process per
physical engine, not one process per virtual bucket.

## Automatic maintenance and shutdown

Snapshot transport is attached before maintenance starts. Each node has one
maintenance scheduler and at most one of its blocking jobs outstanding.
Checkpointing captures applied state on the owner thread, builds a complete
application image outside it, and durably installs that image before logically
compacting the covered log. Retained acknowledged payloads remain in the image.
The scheduler uses both an entry threshold and a maximum interval, and acts on
native-engine checkpoint suggestions without treating them as deletion permits.

Image collection retains a scan cursor across bounded batches and protects
installed images and live transfers/readers. Physical native log maintenance
runs once per node, with explicit rewrite headroom. It can pause persistence
while holding the engine's maintenance gate; it has no hard execution deadline.
Counters expose its duration separately from checkpoint work. Admission errors
or a busy image permit are observed and retried; a failed replica/transport task
causes the service to drain and exit.

SIGTERM/SIGINT stops client admission, joins maintenance and snapshot file work,
then stops and joins peer/runtime owners. Cancelling a shutdown waiter does not
abort the owning supervisor or free the engine while filesystem work is still
running. Unanswered writes remain unknown, and clients must retry their exact
persisted producer/sequence/body tuple.

Disk admission is an application control window, not a filesystem quota or a
promise of unlimited ingest. Replication can consume reserved headroom. Without
publication/source retirement, acknowledged data must accumulate and admission
must eventually stop; checkpoints cannot discard that data to improve a score.

## Observation and qualification

The diagnostic HTTP listener serves `/health/live`, `/health/ready` and `/status`.
It allows at most 16 connections, bounds header buffers, disables keepalive and
uses a two-second connection deadline, independently of WAL RPC capacity.
Readiness reports active local workers and fresh local observations; it is not
a quorum lease or proof of Lance visibility. Status includes each group's log
and checkpoint positions, disk accounting and maintenance/transfer counters.

The process test starts the actual executable on six independent directories,
stops a follower until automatic checkpointing has compacted its missing log,
requires a snapshot to catch it up, kills a leader, then stops/restarts every
process with the same configuration. Exact last-batch retries and complete
metadata/payload reads are checked on every replica. These are correctness and
lifecycle checks, not sustained throughput results.
