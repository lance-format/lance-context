# Bounded work for merge task admission

Dedicated catch-up processes previously scanned all running tasks during both
startup and exact-target admission. Each scan sent a conditional recovery
transaction for every running task, even when its claim was healthy. They also
read global task history on startup and completion. A busy fleet could therefore
make a free target miss its admission deadline and enter external retry backoff.

Dedicated processes now skip global recovery/history maintenance and reconcile
only the task referenced by their target's merge dedupe key. Ordinary masters
retain global maintenance. Their running-task scan uses 64-record pages and one
batched claim read per page; only tasks without observed claims reach the existing
recovery CAS. The CAS still checks the canonical task, absent claim and exact
running record. A stale observation cannot requeue a newly claimed task.

Candidate admission groups at most four locally eligible queued tasks, capped at
64 exact keys across the batch. It reads dependencies together, then reads
ownership metadata together only for dependency-ready candidates. This takes at
most two read-only transactions; dependency-blocked queues do not incur extra
ownership-key reads. Kind and routing filters run before these reads. Very long
dependency chains retain the existing per-key path rather than producing oversized requests.
No WAL payload, base-table data, or global history is read by this preflight.

The execution, target owner, native guard, merge claim, preparation/commit-turn
markers, task claim and live queue key are inspected before allocating a lease.
Dependencies still determine whether a candidate waits, fails, or proceeds to
admission. Clearly blocked candidates are skipped without a lease grant/revoke
pair. Advisory results stay within one page/slice. Lookahead for other candidates
is discarded before any ownership mutation or dependency-failure publication.
Task claims remain serial,
and every claim still passes the original current-state atomic comparisons and
fences; batching does not authorize ownership or cache it across claim calls.

`dependencies_batch` and `admission_batch` are admission diagnostic phases.
Completed etcd responses count actual requests, while queue candidates count inspected task rows; the two
counts are intentionally different. A smaller response count is not itself
proof of shorter production latency or improved WAL throughput.

Lease renewal starts before the claim transaction. A transport error after that
transaction triggers a read of the unique claim token; an accepted claim is
adopted only when the token matches this attempt. Otherwise the attempt stops
renewing and requests lease revocation. If cleanup cannot reach etcd, lease
expiry and ordinary/target-local recovery still apply. No storage fence or
persistent execution ownership is cleared by this recovery.

The dedicated 30-second admission limit bounds retry admission, not an already
in-flight claim attempt. Individual etcd RPCs still have ten-second timeouts.
A successful attempt may finish after the retry deadline and must be delivered
to its executor, rather than dropped by an outer timeout. This does not raise
merge execution or no-progress deadlines.

## Validation and observation

Regression tests cover unrelated malformed records, exact native Job ownership,
live-claim exclusion, recovery of the target's own expired claim, recovery across
64-record page boundaries, token verification after a simulated lost response,
and delivery of a successful attempt across its retry deadline. Existing etcd
tests continue to exercise HA exclusion, dependencies and worker execution locks.

The ignored `benchmark_target_admission_with_eighty_healthy_tasks` test creates an
isolated etcd prefix and compares the old sequential recovery transactions with
batched healthy-claim checking. It also times a complete exact-target claim.
It asserts correctness, not a timing ratio, and removes its fixture prefix.
Run it with `ETCD_TEST_ENDPOINTS` pointing to local or staging etcd.

`catch-up state ready` and `catch-up task claimed` logs expose startup and
admission seconds separately. Bounded-cardinality counters record inspected and
requeued recovery tasks, preflight-blocked candidates, and accepted claims whose
transaction response was lost. Production acceptance requires shorter admission
times and actual subsequent WAL publication; Pod readiness alone is insufficient.

This change does not modify merge slots, staging concurrency, append commits,
retry budgets, native ownership guards, or master/worker deployment settings.

A claim call lazily creates one private lease and establishes renewal before its
first ownership transaction. A definitively rejected claim CAS has published no
keys on that lease, so later candidates in the same call reuse it. The first
accepted (or successfully reconciled) claim takes exclusive ownership of the
lease and renewal task. Separate calls, pools and accepted tasks never share a
lease. An empty scan or error stops renewal and revokes any untransferred lease;
cancellation drops renewal, leaving the existing TTL-based recovery behavior.

`master_scheduler_claim_lease_attempts_total` counts actual grant attempts;
`master_scheduler_claim_lease_reuses_total` counts reuse for later candidates.
`master_scheduler_claim_results_total{outcome}` continues to count individual CAS
results. Grant attempts therefore need not equal accepted plus contended results.
This optimization avoids lease churn after losing claims; it neither removes
contention nor attributes all cluster lease traffic to the scheduler.
