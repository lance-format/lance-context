# Shared node disk admission

`Node::with_disk_budget(DiskOptions)` binds one admission policy to the physical
engine before its group owners or network services start. All hosted groups,
local image builders, incoming snapshots, orphan files and native maintenance
share that policy. Reopening measures the original directory, including partial
files from an interrupted receiver. Policy is local and does not change the
replicated layout or application identity.

`data_bytes` stops new data proposals. Their reservation survives until durable
application or an uncertain completion. Control proposals can use the remaining
`max_bytes` window. Every encoded engine write contributes to a shared monotonic
counter, including follower replication, HardState and sync markers. Successful
admission takes a small accounting lock and does not walk the directory or
decode payloads. `prepare_disk_measurement` returns a task for a blocking worker;
it retains the original engine lock across runtime shutdown.

New image files preserve `headroom_bytes` below the total ceiling. Configuration
requires at least all hosted uncommitted windows plus two native target files.
The local builder counts bounded metadata through its actual encoder and sums
opaque payload lengths before creating either file. A receiver uses its
validated descriptor's exact lengths. Each length is rounded to 4 KiB for
reservation. Pending file work retains its charge across cancellation, then
converts it to counted growth after all buffered writes finish. A later scan may
replace this conservative charge with actual retained lengths/allocation; only
actual deletion returns the space used by an orphan or installed file.

Physical maintenance reserves an additional current-directory-size window
before native rewriting. Rejection occurs before mutation and leaves the node
healthy. A successful native pass refreshes measured usage while appends are
excluded. Image collection instead holds a separate filesystem lifecycle guard:
ordinary log appends can continue during its bounded deletion batches. A fully
drained image scan refreshes accounting after deletion. Neither lock substitutes
for the image owners that protect installed snapshots and captured readers.

The snapshot transport requires a node with this policy configured. Its optional
`Admission` callback can add service-specific restrictions; omitting that extra
policy cannot bypass the node's shared reservation. The real three-process TLS
recovery test uses only the built-in policy, including a killed receiver after a
payload chunk, catch-up, leader loss and full restart from original files.

This is admission with reserved headroom, **not a hard filesystem quota**.
Accepted peer traffic must still persist; followers do not yet advertise disk
credits to leaders. Filesystem allocation and native rewrite peaks can exceed
logical sizes or estimates. Disk space can also be consumed outside this owned
directory. Exhaustion or uncertain persistence still fails closed and requires
recovery. Deployment must provide enough physical space, monitor the remaining
window and schedule measurements/checkpointing/collection. Automatic scheduling,
peer pressure coordination and complete ingestion/materializer integration remain
required before claiming bounded sustained production capacity.

Tests cover shared data/control limits, rejection before receive-file creation,
partial files through reopen and collection, all payloads and exact retries after
engine restart, and a real append completing while the image deletion lifecycle
is held. These establish the stated accounting and recovery boundaries; they do
not establish a throughput gain or resilience to losing a durable majority.

Tracking: #378.
