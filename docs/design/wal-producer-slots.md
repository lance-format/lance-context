# Bounded producer slots

The optional replicated WAL can use a fixed producer slot table instead of
retaining a UUID for every historical writer. Set `Limits::producer_slots` when
creating fresh groups. Zero keeps the existing UUID-based behavior. Capacity is
at most 16,384 cells per group and cannot exceed `max_producers`; this is a count
bound, not a measured memory budget. Session keys still determine bucket routing.
Producer lanes are reusable sequencing identities, not business sessions or TCP
connections. This is part of [the ingestion work in #378](https://github.com/lance-format/lance-context/issues/378).

`Node::producer_command` submits an allocation, close or administrative revocation
through the ordinary durable Raft path. A command carries cluster, layout, group,
table and slot scope. An allocation additionally supplies the expected generation,
bucket and a saved operation UUID. Its exact retry returns the same generation;
competing or stale allocations cannot silently pick another slot. A caller must
persist this intent before sending it. `Query::ProducerSlot` provides a bounded
quorum view for choosing a candidate, but the eventual allocation still needs the
generation comparison in the committed state machine.
An allocation retry can resolve an already-used lane. `Opened` does not reset its
sequence or prove it is unused; recovering clients must retain their original
sequence/pending-body journal and reconcile the quorum tail before sending data.

A close supplies the current generation and exact latest sequence/index. If an
append committed but its reply was lost, a close using an older tail returns the
actual tail without closing the lane. Once closed, new appends are fenced, while
the exact retained latest append can still return its original receipt. A trusted
administrative revoke also closes a generation but preserves its unresolved tail.
There is no timer-based eviction.

Reusing a slot requires both closure and retirement of its exact latest record by
the existing catalog-authorized source closure. Publication alone, a local timeout
or a closed flag does not establish retirement. Reuse removes the old receipt and
retirement marker and increments the cell's durable generation. Old append, open,
close and revoke requests are then fenced. Generation exhaustion never wraps.
An explicit fencing result does not mean the old write never committed and must
not cause automatic replay under a new identity. Historical receipt lookup beyond
slot reuse requires a separate contract and index.

Checkpoint metadata stores a flat, bounded cell table including unused and closed
cells. The reader checks the immutable policy's count before allocating cells and
validates each bounded frame, lane binding and retained receipt. Snapshot transfer
uses the same self-contained image. Zero-slot groups retain policy identity v9/v10
and image format 9; enabled groups use policy identity v11 and image format 10.
Opening an existing UUID-only directory with slots enabled is rejected. Mixed
versions and in-place migration are not implemented.

Scope checks do not authenticate clients or authorize revocation. Production
transport, administrative authorization, SDK journal integration and migration
remain separate work. This change adds no Arrow decode on a replica and does not
claim a throughput improvement or qualify sustained producer churn.

Validation includes state-machine bounds and stale-tail rejection, checkpoint
round trips, an interrupted snapshot transfer between real Raft processes, and a
real Lance publication/retirement fixture. The latter closes a producer before
publication, rejects premature reuse, then reuses it after signed source closure
and verifies old-request fencing, new-generation leader failover and full restart.
An asynchronous-checkpoint regression holds native persistence during the drain,
checks that a preproposal Open refusal changes neither the log nor its slot, and
resolves the same saved Open after the drain. It also checks competing operation
IDs and a duplicate Open after revocation, then reopens all three native stores
to verify the acknowledged records and closed generation.
