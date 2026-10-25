# Single-shard annotation application

The annotation service separates durable WAL acceptance from application success.
A background driver processes accepted records in order, retains the latest
result per bounded producer slot, and signals waiters only after both the complete
annotation state and the result are installed. Missing turns reject the whole
batch and advance progress; temporary staging pressure retains the decoded batch.
Storage or protocol errors stop the driver and make reads fail.

The first binding has exactly one virtual shard and an explicit `TurnAnnotations`
policy. Its Arrow schema contains `id`, label additions/removals, and separate
status/score operation columns: 0 means keep, 1 means set, and 2 means clear. Keep
and clear require a null value; set requires a non-null value. Empty status and
zero score remain valid values. Repeated turn IDs in a batch apply in row order.
Rejection-affecting limits are fingerprinted in the schema so replay and separate
ingresses cannot silently choose different outcomes.

The upload is validated before submission. The original IPC bytes are retained
for exact retries and replicated opaquely. A background applier decodes each fetched
batch once and retains its narrow patches through pressure; replicas do not decode
Arrow. This is not end-to-end zero copy: turning patches into current owned values
still allocates narrow strings/sets. No annotation operation copies a turn body.
Upload reservations and apply workspace use independent budgets: clients waiting
for application cannot occupy the memory needed to complete their own writes.

An exact retry under the latest producer identity returns the original result.
Changed bytes or old sequences are rejected by the existing WAL producer fence.
There is at most one unresolved request per producer. A deadline is ambiguous and
requires retaining the original identity and bytes; it does not authorize moving
to a new sequence. Results contain application status and a source position, not a
historical annotation value.

Each independently connected ingress has a bounded view and background driver.
Point reads obtain an O(1) scoped quorum barrier, wait for the background applied
position, then resolve active/frozen/Lance from a compatible view. The point-read
path does not enumerate or fetch the WAL suffix. A lagging ingress waits or fails
its deadline instead of returning its old base. This arrangement qualifies read
correctness; it does not establish the final multi-shard placement or throughput.

## Remaining integration gates

This service is not enabled in the application registry. It currently starts only
from an unmaterialized source with its original Lance base. The source WAL is the
durable recovery evidence for the bounded in-memory results as well as values.
It refuses an existing publication cut or source closure, instead of pretending
that a values-only snapshot can restore request outcomes.

Next, freeze the result snapshot with the same ordered cut as annotation values,
persist both through managed publication (including result-only batches), restore
them before suffix replay, and wire scheduling/confirmation/source closure. Only
then can this service reclaim WAL and demonstrate sustained draining. Existing
staging/publication component tests do not substitute for that service test.
Process SIGKILL, sustained mixed load, and publication/GC fault qualification remain
outstanding. The RF3 restart test here uses orderly shutdown/reopen.
