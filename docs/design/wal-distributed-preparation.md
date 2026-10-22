# Distributed preparation for one managed table

Refs #378. Experimental; disabled when `preparation` is absent from the application
configuration. This uses Lance 13's public uncommitted-write API and requires no
Lance OSS change or new WAL wire command.

One existing table owner selects and durably pins the complete publication plan.
It sends each configured preparer a small reference containing cluster/layout,
table incarnation, transaction, plan index, read version and partition position.
The preparer resolves storage from its deployment-authorized registry, observes
the exact pending plan and source pins through quorum reads, and selects whole
records by their stable ordinal modulo the partition count. It reads payloads
directly from WAL replicas and moves decoded Arrow handles into Lance writers.
No data payload passes through the publisher or the preparation HTTP protocol.

The publisher joins all requests, validates response identities, complete results,
unassigned fragment IDs and unique relative file paths, then records the combined
descriptor set through the existing durable Prepare command. Only that publisher
can subsequently select the fixed-target manifest and advance/retire sources.
The preparer API has no acquire, commit, publication or retirement operation.
Missing or failed partitions never produce a partial Prepare or manifest commit.

Each preparer owns a shared source-memory budget and bounded job/request slots.
Admitted file jobs outlive disconnected callers and shutdown joins them. A bounded
volatile cache coalesces concurrent identical references and reuses completed
results. Failed entries are removed so the same durable plan can recover. Losing
or evicting a completed cache entry can redo that partition's immutable files;
the complete aggregate becomes durable only at the existing WAL Prepare boundary.
Such abandoned files remain orphans under the existing cleanup contract: this
feature does not implement their deletion. It also does not provide a hard RSS
bound on Lance encoder workspace. No per-part blob metadata writes are added.

The partition count is explicit in each request and remains fixed within one
attempt. After takeover before Prepare, another complete attempt can use a
different count; results from different counts cannot be mixed. After Prepare,
recovery uses the durable aggregate without any preparer. Source data remains
pinned until the ordinary durable publication and closure protocol completes.

Run `lance-context-ingest preparer CONFIG` on separate worker processes. The new
top-level `preparation` block provides a `listen` address for that role, a workload
`bearer_token_file`, and bounded `active`/`retained_results` settings. Publishers
use its `endpoints` list (1–8 distinct HTTP(S) origins) and total RPC `timeout`.
Preparers use the existing authorized table registry, source cap, fetch/write
parallelism, streaming option and materializer memory budget. An empty endpoint
list is valid on preparation-only processes, but not on publishing workers.
As with ingress, deploy the HTTP listener behind TLS; the route requires the
workload token. The TLS WAL connections retain their existing deployment trust.

An integration test runs two separate authenticated HTTP services over a real
TLS WAL and Lance files. One missing service prevents Prepare and visibility;
restoring it reuses the healthy partition, and counters prove each preparer reads
only its assigned half. After both caches disappear and all WAL directories
reopen, the durable aggregate publishes once with every column and payload
verified. These services run in one test process.

A separate release-process experiment uses two producers, two publishing workers,
two preparer processes and nine WAL processes. It acknowledges 192 MiB before
publication, kills all WAL processes and reopens their original directories. With
one preparer missing, the other completes its files but the table remains empty.
Killing that completed preparer loses its volatile cache. Starting both preparers
then publishes all 64 original records in one version, with complete payload,
aggregate-count, read-only and negative-verifier checks and no producer replay.
Each recovered preparer fetches and decodes exactly half the records; publishers
perform zero source fetch, decode or file-write operations. This verifies local
process recovery, not storage-volume loss or throughput. Kubernetes scaling and
sustained performance qualification remain required.
