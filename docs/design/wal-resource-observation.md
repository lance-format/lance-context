# Observe one Rust experiment process

Tracking: https://github.com/lance-format/lance-context/issues/378

The optional Linux resource sampler observes an existing operation from
`context-wal-runner`. It never launches, restarts or stops the target:

```sh
cargo build --release -p lance-context-wal --features runner --bin context-wal-resource
context-wal-resource --operation /tmp/example-operation --interval-ms 500
```

The operation must already have a recorded child identity. The sampler validates
the boot ID, PID and process start time. It checks that identity before and after
each sample and emits a completion record when the recorded incarnation ends.
PID reuse cannot silently switch the measured process. Run the sampler in the
same container as the target: it requires cgroup v2 with a namespace rooted at
`0::/` and validates that target and observer belong to the same cgroup.
It also verifies that the mounted cgroup root directly contains the sampler,
so entering a cgroup namespace alone cannot attribute an ancestor's counters.

JSON Lines output records process CPU ticks, RSS and procfs I/O; cgroup CPU usage,
throttling, current memory and memory events; actual elapsed times and collection
duration. An optional `--disk-directory /tmp/example-wal` counts allocated blocks
in a flat directory, deduplicates hard links and rejects symlinks or subdirectories.
Files removed concurrently by collection may disappear during the scan.

The process record also separates `user_ticks` and `system_ticks` and reports
cumulative `minor_faults` and `major_faults` from the existing stat read, excluding
waited-for children. `cpu_ticks` remains the sum of user and system ticks.
With `--threads`, each observed thread exposes the same counters. Compare deltas
only for the same PID/TID and start time; thread snapshots may omit exited
threads and must not be added to process totals. These fields require no additional
procfs reads. Minor faults do not imply disk I/O or identify an allocation or
copying bottleneck by themselves; major faults and cgroup memory events have
different meanings and scopes.

`--tcp-port 7800` adds an optional `tcp` object. The sampler intersects socket
inodes from the target's open file descriptors with its network namespace's
IPv4/IPv6 TCP tables. Namespace neighbors are excluded and duplicated descriptors
count as one TCP socket. `socket_descriptors` can also include non-TCP sockets;
`owned_tcp_sockets` and the state maps include only matching TCP entries.
Without the option, `tcp` is null and no socket scan is performed.

`states` counts all owned TCP sockets. `local_port_states` and
`remote_port_states` select endpoints matching the configured port, aggregating
all addresses for that process. These two maps can overlap and must not be added.
Keys are Linux state codes: `01` is ESTABLISHED, `08` is CLOSE_WAIT and `0A` is
LISTEN. A peer-closed socket can remain owned by the process and consume a
connection slot; counting only ESTABLISHED can hide it. Separate LISTEN entries
when counting accepted connections. Socket scans contribute to the existing
`sample_wall_seconds` overhead and can miss connections between samples.

The target can lose procfs I/O access during exit before its process state shows
a zombie. On a permission-denied sample, the observer waits at most 100 ms for
positive evidence that the exact incarnation ended. Only then may it discard the
incomplete sample and finish. A permission error for a still-live target remains
an error. Read failures include the location being observed.

These are sampled observations, not continuous peaks, quotas or physical device
write amplification. Directory scans and counter reads are not atomic. Process
and cgroup counters have different scopes and must not be added: other processes
and the sampler itself can contribute to the container counters. Collection starts
after the target identity is available and can miss startup or short-lived peaks.
Record the actual observation interval; its CPU total is not an ingestion-only
cost unless that interval has independently been aligned to ingestion. Observer
overhead is included and needs a separate comparison to quantify.

Unit tests cover stat parsing, PID reuse, directory accounting and real procfs
permission transitions for exiting versus still-live targets. TCP tests use a
real child process, namespace neighbors, duplicated descriptors, a listener and
a peer FIN to verify CLOSE_WAIT accounting; parsing tests cover IPv6. A separate
process test verifies sampling with and without TCP enabled and completion after
target exit. CI explicitly
enables these optional-feature tests; the integration test runs unprivileged in
private mount/cgroup namespaces with a namespace-rooted cgroup2 mount. The
ignored subprocess fixtures are invoked by their parent tests.
The tool does not change production ingestion semantics.
