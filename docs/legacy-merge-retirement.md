# Retire a pre-fencing merge execution

New fenced workers use the existing etcd execution record, progress watchdog,
commit-version permits and storage recovery barriers. The master now selects
that protocol per worker, including when the table is not enrolled for native
catch-up. Capability failure never falls back to an untracked POST. A worker
which has not resolved its persistent admission mode does not admit a merge.

An already running legacy HTTP execution is different: it never recorded its
commit permits. It cannot be recovered by deleting a lock or expiring a lease.
The master supports a one-time, identity-verified transition for these records.

`POST /api/v1/merge-retirements` accepts one explicit target and execution ID:

```json
{
  "action": "prepare",
  "target": "example",
  "execution_id": "the-existing-execution-uuid",
  "namespace": "rollout-db",
  "pod": "rocketkeep-server-9",
  "container": "rocketkeep-server"
}
```

The same request can be read from a file with
`lance-context-master --legacy-retirement-request /path/request.json`. This
command uses the existing configuration and service account and exits without
starting a scheduler, opening the stats dataset or scanning tables. It requires
Pod GET access. It does not require permission to modify Pods, and does not add
any RBAC or ConfigMaps.

1. **Prepare while the old process is live.** The master checks the configured
   worker's direct Pod address, brackets its instance/shard capability probe
   with Kubernetes identity reads, and saves the exact execution, Pod UID,
   container ID and writer ID in etcd. The execution and target lock stay held.
2. **Replace the old process with a fenced worker.** This API never sends a
   signal or interprets elapsed time as a reason to kill. The replacement must
   have `MERGE_FENCE_THIS_WORKER=true`, a stable `MERGE_EXECUTOR_ENDPOINT`, the same
   writer ID and the same data/coordinator configuration. Deploy the master's
   mixed-worker routing support before enabling fenced worker admission.
   This worker-specific switch persists against its stable writer ID in etcd;
   it does not set the cluster-wide `MERGE_FENCE_LEGACY` marker or change other
   workers' startup mode. Existing cluster-wide fencing remains authoritative.
   A restarted worker cannot disable its persisted mode by removing this flag.
   Older binaries that do not understand this marker are not a supported
   rollback: retaining the upgraded worker binary is part of the handoff.
3. **The next canonical merge task completes the prepared handoff.** Normal
   task retries read the exact execution's journal and run completion without
   another agent call. For immediate completion, the same request with
   `"action":"complete"` is also supported. The master
   requires the same Pod UID, a different running container/instance, and a
   Kubernetes `lastState.terminated` receipt for the exact old container. A
   missing Pod, a changed Pod UID, Ready alone, or lease expiry does not qualify.
   The replacement must confirm the same durable execution from its own etcd
   client, the same data directory and shard, and fenced legacy admission.
4. The master reads only the base and this writer's shard manifests, persists
   their `latest + 1` bounds, and conditionally occupies versions beyond those
   bounds. After actual process death, old application code cannot select new
   versions or retry conflicts; delayed immutable storage writes are therefore
   covered. The barrier preserves current WAL entries and writer epoch. A
   missing shard remains unresolved rather than being treated as quiescence.
5. Only after both barriers succeed does an etcd compare-and-swap mark the
   exact old execution terminal. The API never deletes or replaces the target
   lock or task claim. Normal scheduler reconciliation releases it using a
   valid claim, then schedules the replacement. Persisted plans survive retries;
   an old recovery request cannot change a successor execution.

`master_legacy_retirement_attempts_total{result="completed"|"unresolved"}`
reports canonical completion attempts. An unresolved proof keeps the execution
and lock intact. No per-target preparation is inferred from another table's
journal: old protocol-0 records do not persist their process instance, so an
endpoint alone is insufficient evidence to adopt an unprepared execution.

This is a migration tool, not an automatic process killer. It deliberately
requires preparation before termination and a same-Pod container replacement.
Other retirement mechanisms need their own identity and termination proof.
Before production use, validate the exact old and replacement images with a
stalled legacy merge, delayed base/shard publication, concurrent ingestion,
process termination, repeated completion requests, and full row/blob recovery.
