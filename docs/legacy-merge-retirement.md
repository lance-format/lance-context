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
   missing shard records an initial-publication bound of version 1. Completion
   conditionally creates the empty initial manifest (epoch 0, without opening
   a writer), then crosses that bound. Context writers must publish their
   initial manifest before appending WAL, so an absent initial manifest cannot
   hide acknowledged WAL from that writer. A late initial PUT conflicts; if
   another publication won first, its WAL entries and epoch are preserved.
   This applies only to a persisted plan that originally observed absence. A
   previously published shard that disappears remains unresolved.
5. Only after both barriers succeed does an etcd compare-and-swap mark the
   exact old execution terminal. The API never deletes or replaces the target
   lock or task claim. Normal scheduler reconciliation releases it using a
   valid claim, then schedules the replacement. Persisted plans survive retries;
   an old recovery request cannot change a successor execution.

`master_legacy_retirement_attempts_total{result="completed"|"unresolved"}`
reports canonical completion attempts. An unresolved proof keeps the execution
and lock intact.

## Requests admitted during the handoff window

A master can record another legacy request after the preparation inventory,
while the old worker is shutting down. An HTTP 503 with the exact body
`worker is draining` is the worker's pre-handler rejection: the master records
that failure and releases this unadmitted request through its normal claim.
Other 503 bodies, truncated responses and response timeouts remain protected.

For an already retained record discovered after replacement, explicitly add
`"witness_execution_id":"a-completed-retirement-uuid"` to a `prepare` request.
The witness must be a completed retirement for the same endpoint, namespace,
Pod and container. The master rechecks its exact old container's death, the
same Pod UID, and the fenced successor's storage and coordinator identity.
An etcd transaction compares the witness, current execution and its target
lock before saving the new journal. Only the verified process identity is
inherited: this target must obtain its own storage snapshot and barriers.
Normal canonical completion and lock release then apply. Remove the optional
witness field when sending an explicit `complete` request.

Completed target witnesses are not selected automatically from an endpoint or
empty progress telemetry. A changed Pod UID, an incomplete witness, or changed
ownership keeps recovery protected. Another restart requires the persisted
worker termination receipt described below; without it the exact original
Kubernetes termination record is still required.

## Workers with no current execution

An empty inventory is not an admission barrier: a previously probed caller can
publish a legacy execution during replacement. Before replacing an otherwise
idle worker, capture a **worker retirement intent** through the same API or
`--legacy-retirement-request` CLI flag:

```json
{
  "action": "prepare_worker",
  "intent_id": "67d91ff4-7cd1-45ec-a9cd-a14a7c135975",
  "endpoint": "http://worker-7.workers.production.svc.cluster.local:3000",
  "namespace": "production",
  "pod": "worker-7",
  "container": "worker"
}
```

Use a new UUID for the intent and retain that request for retries. The master
captures the live Pod UID, container ID, worker instance and shard identity;
the client cannot supply process evidence. One transaction persists the intent
and its exact-endpoint index in the application's etcd. It creates no target
execution, task claim, storage version or target lock. The intent ID and endpoint
cannot subsequently be rebound to a different process.

Perform a qualified **graceful same-Pod replacement**. This path requires the
old container's exact termination record with exit code 0 and reason `Completed`;
an OOM, forced termination, changed Pod UID or missing receipt is rejected.
The replacement must enable fenced legacy admission. Verify acknowledged rows
and blobs across shutdown, including writes not yet sealed, before using this
path in production. Registering an intent alone never makes a restart safe.

If an ordinary task subsequently discovers a legacy execution for that endpoint,
the master finds the explicitly registered intent, rechecks graceful process
death and the fenced successor, and prepares this target's own journal. The
existing storage barriers and canonical completion then run automatically.
No agent needs to bind each late target. For a manual `prepare` request,
`"worker_witness_id":"the-intent-uuid"` is supported instead of
`witness_execution_id`; supplying both is rejected. Explicit `complete`
requests omit both witness fields.

### Preserve termination evidence before another restart

Kubernetes keeps only the immediately preceding container termination. Before
a second same-Pod replacement, send the original worker request with
`"action":"confirm_worker_termination"`, retaining the original `intent_id`,
endpoint, namespace, Pod and container. This action requires a master binary
that implements it; it does not restart or drain the worker.

The master rechecks the captured old container's graceful termination and
brackets the live fenced successor's capability probe with Kubernetes identity
reads. It persists an immutable `worker_termination` receipt in application
etcd, comparing both the original intent and its endpoint index. The receipt
contains the original process, the observed successor and the termination time.
It creates no execution, task, storage version or target lock. Confirm it while
the original `lastState.terminated` evidence still exists; a missing receipt
cannot be reconstructed from elapsed time, readiness or an empty queue.

Canonical preparation of a late target also saves this receipt automatically.
An idle worker may have no such task, so explicitly confirm before its next
replacement. Retrying confirmation returns the original receipt while still
checking a current fenced successor. It cannot replace the original intent or
termination evidence.

After another restart, recovery may use the persisted proof of the old
process's death. It still probes the current ready process in the same Pod UID,
with the same writer/shard identity and fenced admission. Each late target must
still validate its own storage/coordinator namespace, prepare its own storage
barriers and complete through the normal ownership comparisons. A receipt
never authorizes deleting a lock, recovering an unknown process, accepting a
changed Pod UID, or rolling back to an unfenced binary.

### Preserve an earlier target retirement without a worker intent

Some earlier migrations registered only a target, not a worker-wide intent.
Before another same-Pod restart, use the same API or CLI with:

```json
{
  "action": "preserve_worker_retirement_witness",
  "intent_id": "a-new-registration-uuid",
  "witness_execution_id": "the-completed-target-execution-uuid",
  "endpoint": "http://worker-9.workers:3000",
  "namespace": "rollout-db",
  "pod": "worker-9",
  "container": "server"
}
```

The master derives the **original** process from the completed target journal.
It requires Kubernetes to still expose that exact container's termination and
checks a stable, ready, fenced successor in the same Pod. This route accepts a
nonzero original exit because the target retirement already verified death and
completed its storage barriers; it does not certify data durability across an
ungraceful shutdown. An incomplete journal, missing original termination,
conflicting endpoint registration, or changed Pod UID is rejected.

One transaction compares the completed journal and atomically installs the
worker registration, endpoint index, and immutable death receipt. A compatible
existing registration is reused and its real ID is returned. The new ID labels
this registration; it does not pretend a worker intent existed before death.
No execution or target lock changes. After a subsequent restart, canonical
late-target recovery can use the receipt, but must still build and complete
**each target's own** storage barriers. The witness's barriers are never copied.
Other worker actions reject `witness_execution_id`.

The intent and receipt are retained for late admissions. They support a legacy
transition, not a replacement for native fenced-worker recovery.

This is a migration tool, not an automatic process killer. It deliberately
requires verified process evidence and a same-Pod container replacement.
Other retirement mechanisms need their own identity and termination proof.
Before production use, validate the exact old and replacement images with a
stalled legacy merge, delayed base/shard publication, concurrent ingestion,
process termination, repeated completion requests, and full row/blob recovery.
