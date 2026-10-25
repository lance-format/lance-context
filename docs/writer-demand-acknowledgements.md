# Acknowledged writer demand

The resident flush sweeper observes each writer's WAL watermark even when no new
generation was sealed. Previously each observation read the durable demand key
before discovering that its value was unchanged. A local two-worker idle test
observed sixteen such reads in eight seconds.

Each server now retains a bounded LRU of successfully acknowledged writer
observations, keyed by table and shard. The same sealed generation, writer epoch
and byte watermark needs no further etcd read unless its observation contributes
an earlier flush timestamp. New generations and older out-of-order observations
still reach the durable join. Scan/executor events and events carrying additional
merge or age evidence bypass this cache.

A successful unchanged join is an acknowledgement too. Errors, timeouts, panics
and cancelled futures never populate the cache, including a committed write
whose response is lost. Concurrent old completions cannot replace a newer cached
watermark. Eviction and server restart replay safely through the existing join.
The cache contains metadata only, retains neither datasets nor server state, and
is capped at 16,384 entries and 1,024 combined table/shard key bytes per entry.
Larger keys bypass caching.

Demand records are durable monotonic joins, not leased ownership records. This
optimization assumes their acknowledged state is retained: manually deleting
them or restoring etcd to an older snapshot requires republishing writer state
(for example, by restarting the affected server). It does not authorize writes,
replace execution fencing, or cache merge-request acknowledgements. Independent
merge requests still publish before the telemetry-cache check, even when the WAL
watermark is unchanged and local self-merge is disabled.

`rollout_wal_demand_events_total{result="cached"}` counts local skips. The existing
`result="unchanged"` continues to mean that a durable read occurred. Compare rates
for the same workers across rollout, alongside etcd Range/Txn rates and failures;
a reduction in this path alone does not establish a fleet-wide improvement.
