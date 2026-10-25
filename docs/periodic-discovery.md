# Periodic maintenance discovery

Automatic WAL and compaction sweeps elect a persistent owner for each distinct
discovery policy. Replicas with the same data directory, task kind, interval,
and selection settings share an owner. Followers watch its etcd key; they do
not rescan candidate statistics or repeatedly attempt the coordination lock.
Different thresholds, compaction options, or quiet hours retain independent
policies. A disabled interval starts no periodic discovery loop.

The owner runs scans independently of task dispatch. Candidate publication is
fenced by the current owner token in the enqueue transaction. Losing the lease,
owner Watch, or admission stops its scans, and another replica takes over.
This ownership authorizes queue metadata, not storage commits; task claims and
merge fencing remain unchanged.

Writer requests and resident merge continuations still reach the existing
demand controller immediately. A zero-result merge does not suppress future
writer requests. Manual sweep endpoints retain their behavior, and automatic
sweeps still take the existing short sweep lock for mutual exclusion with
manual calls and older binaries. Older binaries continue their replica-local
timers until upgraded, so a mixed fleet can still perform repeated sweeps.

The first sweep runs when ownership is acquired, then follows its configured
interval. Slow scans do not accumulate a burst of missed ticks. Graceful drain
releases idle leadership without waiting for the next interval.

Counters are per master process and tagged only by `kind` (`merge_wal` or
`compact`): `master_periodic_sweep_started_total`,
`master_periodic_sweep_enqueued_total`, and `master_periodic_sweep_errors_total`.
Use summed counter rates across processes; do not interpret a missing series
on an old binary as a measured zero. Compare these rates with completed WAL
work, read health, and etcd request latency during rollout.
