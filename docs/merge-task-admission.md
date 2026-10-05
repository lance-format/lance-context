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

Candidate admission reads the execution, target owner, native guard, merge claim
and task claim together before allocating a lease. Clearly blocked candidates
are skipped without a lease grant/revoke pair. This preflight is an optimization:
all final atomic ownership comparisons remain. Dependency handling, queue
pagination and kind filtering remain in their original order.

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
