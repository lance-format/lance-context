# Quorum application queries

Refs #378.

Cross-group publication needs a committed view of catalog plans and source state.
A local `publisher`, `accepted` or `status` observation cannot establish that the
responding node still participates in a majority. The optional replication Node
now queues bounded `Query` requests using raft-rs safe read-index processing.

Only the group's current leader accepts a query. It must obtain a read-index
confirmation and apply through that index before capturing the query result.
Admission rejects a newly elected leader until an entry from its current term
commits: raft-rs otherwise silently drops the request. This rejection reserves
neither capacity nor the token; callers can retry after driving replication.
`Events::reads` includes the request token, group, cluster/layout scope, current
term, confirmed index and captured applied position. Accepted-page contents and
their cursor come from that same state. Producer queries return only the latest
metadata/index; they do not fetch, deserialize or clone payload bodies.

Reads and proposals share each group's `max_pending` capacity and the owner's
request-token namespace. Accepted pages are limited to 128 records; queries contain
fixed identifiers and bounds. Publisher answers use the existing bounded table
representation. An ahead-of-state cursor returns `CursorAhead` after the quorum
read, without fencing the node. Source placement and catalog-group routing are
validated before admission.

Each group has at most one unconfirmed read-index round. Requests arriving while
it is outstanding join a separate, unsent batch within the same `max_pending`
limit. Its fresh barrier is issued as soon as the preceding round confirms, in
that same poll; no timer or later request is needed to start it. The first round's
quorum never confirms the later arrivals. Confirmed queries still wait for apply,
and leadership loss retires both the sent round and its queued successor.
This reduces overlapping read rounds, but sequential reads still require fresh
quorums. It adds no lease, batching delay or persistent format.

Each group's reader uses a new incarnation UUID on reopen and a never-reused
monotonic context counter. Caller token reuse cannot match an earlier context.
Confirmed reads wait for apply; term or leadership loss yields `Unknown` for
unresolved queries. The owner must keep ticking and polling to observe these
transitions. No follower forwarding, cached lease or cancellation shortcut is
provided. Storage failure fences the physical node, and a failing poll releases
none of its messages, append completions or query results.

A returned value describes state observed during its request. It is not a lease,
authentication proof, producer ACK or external commit fence. Publication commands
must still validate immutable plans and dependencies at deterministic apply;
source retirement needs a durable publication outcome. A later takeover may occur
after a successful read. Exact append retries still use the existing proposal
path; using these reads to optimize retries is subsequent integration work.

Tests cover new-leader readiness, shared admission, request-token reuse, stale
contexts, ahead-of-state cursors, apply barriers, physical sync failure and isolated leaders. Real TCP
child processes verify majority reads, independent appends during isolation,
unchanged log positions for reads, retained publisher snapshots and full process
restart with exact producer tails and payloads.
