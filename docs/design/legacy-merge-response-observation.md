# Yield scheduler capacity while retaining a legacy merge

A legacy worker can accept a merge POST and take an unbounded time to return.
The master must retain that original request and its ownership: dropping it on
an HTTP timeout can lose the acknowledgement needed for safe recovery.

After `LEGACY_MERGE_FOREGROUND_SECS` (default 30), the scheduler may move the
waiting HTTP response out of its execution slots. This is a fairness interval,
**not a worker timeout or a no-progress decision**. The same task, claim lease,
admission guard, durable execution and HTTP future remain live. It is not marked
failed or done, dependencies keep waiting, and no duplicate POST is issued.

`LEGACY_MERGE_WAIT_CONCURRENCY` bounds additional waiting requests per master
(default 32; zero disables yielding). When full, a request keeps its original
execution slot until its response arrives or waiting capacity becomes available.
On arrival, the original task first registers in the fair execution queue, then
releases the response-wait budget before blocking for its execution slot. It
reacquires that slot before acknowledging/continuing serial fan-out. This gives
existing tasks precedence over new claims without deadlocking the two capacity
limits.

Only legacy HTTP waits use this mechanism. Local merge/encoding, compaction,
indexing, claim RPCs and fenced merge execution retain their current budgets.
`/executor` reports `waiting_legacy_response` and `awaiting_merge_slot` phases;
`master_legacy_response_waiters` counts parked requests and
`master_merge_wal_response_yields_total` counts slot yields. Neither is storage
progress or authority to retire a worker. Drain still joins the original task.

The existing serial fan-out and durable execution protocol are unchanged. There
is no new persisted schema or upgrade requirement for other masters/workers.
Existing requests already accepted by an old master are not transferred by a
new binary. Permanently stuck requests can exhaust the bounded waiting capacity;
this change does not prove worker0/worker9 retirement, fence old storage writes,
or authorize resetting their locks. Verified migration to the fenced protocol
remains necessary for safe automatic replacement of those old writers.

Tests cover one scheduler slot serving another table while the original request
and dependency chain remain live; late completion of that same task unblocks its
dependent work; drain waits for response and continuation; full capacity retains
the original request; and simultaneous full execution/wait pools can recover
without deadlocking. No production performance claim follows from these tests.
