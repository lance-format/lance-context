# Rollout materialization through the public WAL transport

The optional `lance-context-ingest/materializer` feature supplies a publisher for
the fixed rollout schema. It reads source records over the same authenticated,
bounded WAL client used by ingestion, writes real Lance files, publishes through
the managed manifest catalog, and retires source payloads after durable source
closure. No Lance OSS changes are required.

The application supplies an authorized `Ingress`, resolved storage URI/object
path/options, a shared Lance session, a shared materialization memory budget and
a persisted publisher-acquisition identity. The storage namespace and dataset
incarnation must already be bound and bootstrapped. Catalog epochs fence stale
owners; a scheduling lease alone does not authorize a publication.

## Publication and recovery

`prepare_next` selects a bounded prefix across source groups, records one fixed
transaction/read version, obtains catalog-issued source permits, and commits
each source pin before fetching payloads. It matches each fetched record against
the pin, verifies its digest off the async executor, and validates the actual
rollout IPC/schema/session bucket. Existing input buffers remain shared through
Arrow decoding; alignment repair and Lance encoding can allocate.

The publisher limits source bytes, record count, fetch concurrency, file writers
and per-group scan pages. A bounded number of writers stage immutable files;
their descriptors are then recorded in the replicated catalog. An uncertain
plan/prepare RPC retains the same in-memory attempt. After a new authority takes
over, a durable prepared result is reused instead of writing replacement files.
Files produced before a prepare is committed can become orphans.

`commit_pending` uses the recorded fragments and transaction UUID with the
managed commit handler. The target remains exactly read-version plus one;
automatic rebase and cleanup are disabled. A selected manifest is resolved by a
quorum catalog read and checked against the transaction, plan, preparation and
descriptor digest before confirmation. An uncertain commit never authorizes
source deletion or moving the transaction to a new version.

`finish_pending` obtains closure permits from the confirmed catalog, commits each
source closure and obtains its quorum proof. The catalog verifies the proofs
before advancing its independent source positions. Ingress receipts become
published at that advancement. A crash after manifest selection or after some
source closures resumes the same pending publication; exact last-batch retries
retain their original ACK positions even after the payload is retired.

`run_once` combines these phases. An error is reconciled using the durable state
on the next call. A return of `None` means there is currently no pending work;
it is not proof that an earlier uncertain operation was rejected. The application
still supplies bounded scheduling, backoff, shutdown and table discovery.

## Resource and retention boundaries

One materialization reserves space for source buffers, decoding and row metadata
before payload fetch. Admitted file-writing work retains the reservation if its
caller is cancelled, and joins all its file writers even when one fails. These
are logical application bounds, not an allocator or encoder/cgroup hard limit.
The application must also bound its Lance encoder and metadata caches.

Pages containing only other tables can advance an in-memory scan hint; they do
not advance the durable table checkpoint. A selected record is never skipped by
that hint. Each scan is bounded, and publication rotates through source groups.
This is not a global uniqueness index: globally unique immutable row IDs remain
the caller's explicit contract, as on the ingress adapter.

Catalog manifest history is retired to a bounded suffix while retaining the
bootstrap record and active dependencies. Retirement does not delete any blob
object. Old-version reads can fail once catalog history is retired. Physical
WAL log/image collection belongs to the WAL service; coordinated blob orphan
collection and legacy-table migration remain separate integration work.

## Qualification scope

The materializer regression uses real TLS-connected replicas and a real Lance
dataset with the complete rollout schema. It restarts the original replica
directories after file preparation and again after manifest publication, takes
over publisher authority, verifies complete row values, retires the source
payloads and retries the original appends. Additional publications exercise
bounded manifest history, followed by another full-directory restart.

The regression and feature Clippy checks pass, including the existing ingress
and TLS catalog-reader regressions. This is not a sustained throughput result,
a Kubernetes deployment or an HTTP ingestion integration.

Tracking: #378.
