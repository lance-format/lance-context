# Deployment: Distributed Rollout Ingest with Server-ID Sharding

**Status:** Draft
**Author:** Beinan Wang
**Date:** 2026-07-09
**Scope:** How to deploy the `lance-context-server` rollout store as a multi-instance service behind a plain load balancer, and how server-id MemWAL sharding makes high fan-in ingest correct by construction.

---

## 1. Summary

Rollout ingest is high fan-in: thousands of generation workers append trajectories concurrently (schema-design §2). To serve that from more than one server instance without a coordinator, each instance writes through Lance's **MemWAL** to a shard it owns exclusively, keyed by its **stable server/instance id**.

The result is a service you can scale horizontally behind an ordinary round-robin load balancer:

- **No write coordination.** Each instance owns one shard; no two instances share a shard, so MemWAL's single-writer + epoch-fencing invariant holds *by construction* — there is never a write war, no matter how the load balancer spreads requests.
- **No read affinity.** Every append is flushed to object storage before the request returns, and the read path rebuilds from object storage across all shards. Any instance sees every instance's writes.
- **No client changes.** Sharding is entirely server-side. Workers just POST to a load-balancer VIP.

The one hard rule: **each instance must present a distinct, stable id.** A Kubernetes StatefulSet gives you exactly that for free.

---

## 2. The sharding model

```text
              ┌──────── clients: generation workers (append) + learner (read) ────────┐
              │   POST /rollouts/{name}        GET /rollouts/{name}/{id}               │
              └───────────────────────────────────┬───────────────────────────────────┘
                                                   │  round-robin, no session affinity
                                          ┌────────▼────────┐
                                          │  Service  (LB)  │
                                          └───┬─────────┬───┘
                        ┌─────────────────────┘         └─────────────────────┐
                  ┌─────▼─────┐                                          ┌─────▼─────┐
                  │ rollout-0 │  owns shard = uuid5("rollout-0")         │ rollout-1 │  owns shard = uuid5("rollout-1")
                  │   (Pod)   │  single active writer                    │   (Pod)   │  single active writer
                  └─────┬─────┘                                          └─────┬─────┘
                        │       close-per-append → flush to storage            │
                        └───────────────────────┬──────────────────────────────┘
                                        ┌────────▼────────┐
                                        │  object storage │   ONE Lance dataset:
                                        │   (S3 / GCS)    │   base  ∪  _mem_wal/{shard}/generations
                                        └─────────────────┘
```

- **One instance = one shard.** The write shard is derived deterministically from the instance id: `shard_uuid = uuid_v5(NAMESPACE_OID, instance_id)` (`derive_shard_id` in `rollout_store.rs`). The same id always maps to the same shard across restarts and reschedules.
- **One dataset, many shards.** All instances open the *same* Lance dataset (same URI on shared object storage) and write disjoint shards of it under the `_mem_wal/{shard}/` namespace. They do not each get a private dataset.
- **Single active writer per shard, two ways:** across instances by distinct shard ids; within an instance because `RolloutStore::add` takes `&mut self` and the server holds a per-store write lock, serializing that instance's own appends.

### Instance-id resolution

The server resolves its id in this order (`ServerConfig::resolved_instance_id` in `config.rs`):

1. `--instance-id` flag or `INSTANCE_ID` env var, if set;
2. otherwise the `HOSTNAME` env var (a StatefulSet pod's hostname is its stable ordinal name, e.g. `rollout-0`);
3. otherwise a single fixed `"default"` shard.

Because Kubernetes sets `HOSTNAME` to the pod name automatically, a StatefulSet needs **no explicit configuration** to shard correctly — though setting `INSTANCE_ID` explicitly (§5) makes the intent obvious and is recommended.

> **The `default` fallback is single-instance only.** If you run several instances that all resolve to `default` (or you set the *same* id on two of them), they contend for one shard and MemWAL fences all but one writer — appends start failing. Multi-instance deployments **must** give each instance a distinct id.

---

## 3. Why this is correct: single-writer + epoch fencing

MemWAL is a log-structured writer. Each shard has a manifest chain in object storage, and advancing it is a compare-and-set (`PUT-IF-NOT-EXISTS` with epoch fencing): if two writers try to extend the same shard concurrently, exactly one wins and the other is fenced and must fail or retry. This is the invariant that makes MemWAL safe without a lock service — **but it assumes a single active writer per shard.**

Consistent hashing, a write-forwarding leader, or a distributed lock are the usual ways to guarantee that. Server-id sharding gets it for free: since the shard id is a pure function of the instance id, and instance ids are distinct and stable, **no two instances ever target the same shard.** The fencing path is never exercised in normal operation; it exists only as a backstop against the misconfiguration in §2 (two writers, one id).

This is why a *dumb* load balancer is sufficient. Correctness does not depend on *which* instance a request lands on — only on each instance owning its own shard. Round-robin, least-connections, random: all fine.

---

## 4. Read-your-writes and cross-instance visibility

Writes are made durable synchronously, so reads need no affinity:

- **Write (`RolloutStore::add`):** append the batch to the shard's memtable (`put`) and then `close()` the writer. `close()` freezes the memtable and blocks until the flush produces a Lance fragment in `_mem_wal/{shard}/` on object storage. When `add` returns, the rows are durable — not merely buffered in the writing instance's memory.
- **Read (`RolloutStore::lsm_scanner`):** on every call, list the dataset's shard ids (`list_mem_wal_latest_shard_ids`), read each shard's latest manifest from object storage (`ShardManifestStore::read_latest`), and scan the union of the base table and every shard's flushed generations, de-duplicated by `id`.

Because the read snapshot is rebuilt from object storage — not from any instance's local state — an instance sees **every** instance's flushed appends, including its own most recent one. A worker can `POST` a rollout to `rollout-0` and the learner can immediately `GET` it via `rollout-1`; the load balancer may route the two requests anywhere.

Trade-off: this is the durability-favoring end of the dial. `close`-per-append flushes each batch to object storage, which bounds throughput per shard by flush latency. High aggregate fan-in is absorbed by **adding shards (instances)**, not by batching many appends into one flush. If you later need higher per-shard throughput, group multiple trajectories into a single `add` call at the client.

---

## 5. Kubernetes deployment

Two pieces: a **StatefulSet** (stable per-pod ids) and a **Service** (the load-balancer VIP clients talk to). A headless Service backs the StatefulSet's stable DNS; a normal Service fronts it for load balancing.

```yaml
# Headless service: stable network identity for the StatefulSet pods.
apiVersion: v1
kind: Service
metadata:
  name: rollout-headless
spec:
  clusterIP: None
  selector: { app: rollout }
  ports: [{ name: http, port: 3000 }]
---
# Load-balancer service: the single VIP clients POST/GET against.
apiVersion: v1
kind: Service
metadata:
  name: rollout
spec:
  type: LoadBalancer          # or ClusterIP behind an Ingress
  selector: { app: rollout }
  ports: [{ name: http, port: 80, targetPort: 3000 }]
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: rollout
spec:
  serviceName: rollout-headless
  replicas: 4                 # rollout-0 .. rollout-3, each owns one shard
  selector: { matchLabels: { app: rollout } }
  template:
    metadata: { labels: { app: rollout } }
    spec:
      containers:
        - name: server
          image: lance-context-server:latest
          args: ["--host", "0.0.0.0", "--port", "3000",
                 "--data-dir", "s3://my-bucket/rollouts"]
          env:
            # Stable per-pod id → stable shard. metadata.name is rollout-0,
            # rollout-1, ... (This is also what HOSTNAME resolves to, so this
            # block is belt-and-suspenders — but explicit is clearer.)
            - name: INSTANCE_ID
              valueFrom: { fieldRef: { fieldPath: metadata.name } }
            # Object-store credentials (example: S3).
            - name: AWS_REGION
              value: us-east-1
            # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY from a Secret, etc.
          ports: [{ containerPort: 3000 }]
```

Key points:

- **Shared dataset on object storage.** `--data-dir` must point at shared object storage (`s3://…`, `gs://…`, …) that **all** pods open at the **same** URI. Rollout datasets live at `{data_dir}/{name}.rollout.lance`. Do **not** give each pod a private `PersistentVolumeClaim` for the data dir — that would create N separate datasets, and reads would not union across them. (PVCs for logs/scratch are fine.)
- **StatefulSet, not Deployment.** A Deployment assigns random pod names, so ids would not be stable across rollouts. A StatefulSet's ordinal names (`rollout-0…`) are stable and reused on reschedule.
- **Replicas = shard count.** Scaling `replicas` changes how many shards absorb the ingest fan-in.

---

## 6. Scaling, rescheduling, and failure

Every append is flushed before `add` returns and every flushed generation is recorded in the shard manifest on object storage, so instance lifecycle is safe:

| Event | What happens | Safe? |
|---|---|---|
| **Pod reschedule** (`rollout-2` dies, K8s recreates it) | Same pod name → same instance id → same shard uuid. The new pod resumes writing the same physical shard. Nothing was buffered in memory (close-per-append), so nothing is lost. | ✅ |
| **Scale up** (`replicas: 4 → 6`) | New pods `rollout-4`, `rollout-5` derive fresh distinct shards and begin writing immediately; their rows are visible to all readers on first flush. | ✅ |
| **Scale down** (`replicas: 6 → 4`) | Retired pods stop writing. Their already-flushed shard generations remain in object storage and are still read (the shard manifest persists). The shard simply receives no new appends. | ✅ |
| **Two writers, one id** (misconfig: same `INSTANCE_ID` on two pods, or several `default`s) | Both target one shard; MemWAL epoch-fencing lets one win and fails the other. Appends error. | ❌ Avoid — see §2. |

### 6.1 Bounding read amplification: count-triggered self-merge

By default (`--rollout-merge-after-generations 0`) there is no compaction step: rollout rows are append-only (§7), so each `add` leaves a new flushed generation in `_mem_wal/{shard}/` and every read unions all of them. That is correct, but read cost grows linearly with the number of un-merged generations — a shard that has absorbed thousands of appends makes every `list`/`get` open thousands of generation datasets.

To bound this, an instance can **merge its own shard back into the base table** on a size trigger. Set `--rollout-merge-after-generations N` (env `ROLLOUT_MERGE_AFTER_GENERATIONS`, or `RolloutStoreOptions::merge_after_generations`). After an append flushes a generation, if this instance's shard has accumulated ≥ N un-merged generations, the same `add` call synchronously:

1. reads every flushed generation (each is a self-contained Lance dataset under `_mem_wal/{shard}/`),
2. appends their rows to the **base table** (`Dataset::append`), and
3. `commit_update`s the shard manifest to drain `flushed_generations` back to empty — leaving `replay_after_wal_entry_position` untouched, so a reopened writer never re-replays already-merged WAL entries.

This is the "external compactor" path that Lance's MemWAL LSM design explicitly anticipates. Two properties make it safe under the §2 deployment model:

- **No fencing of peers.** The merge `claim_epoch`s the shard, which bumps its writer epoch — but each instance merges *only the shard it writes*, and that shard has no other live writer to fence (§3). The next `add` on this instance simply re-claims the now-current epoch.
- **Crash-safe by immutability.** Rollout rows are immutable and de-duplicated by `id` at read time. If a crash lands between step 2 and step 3 (rows in base, manifest not yet drained), a reader sees the rows via *both* the base table and the still-listed generation and de-dups them — no double counting. The next merge attempt drains the manifest.

**Trade-off — synchronous tail latency.** The merge runs inline, so the single append that crosses the threshold pays the full merge cost while the other N−1 appends stay fast. Choose N to amortize that spike: larger N ⇒ rarer but heavier merges and higher steady-state read amplification; smaller N ⇒ smoother reads but a merge cost folded into more appends. `N = 0` keeps the pure-accumulation 0.6.0 behavior.

**Measured impact.** A micro-benchmark (`bench_merge_read_amplification`, single-row appends, local FS, release build) shows how sharply read cost grows with un-merged generations:

| Read (`list` scan of 200 rows) | Latency |
|---|---|
| over **200 un-merged generations** | ~654 ms |
| after **merge into the base table** | ~50 ms |
| **speedup** | **~13×** |

The merge itself was cheap here — the slowest single `add` (which folded the shard on every append at `N=1`) was ~20 ms. The read amplification is the real cost accumulation avoids; the exact ratio grows with generation count.

> This is *self-merge*, not a coordinator: no cronjob, no admin endpoint, no cross-instance lock. A load balancer can keep spraying appends across instances; each instance independently keeps its own shard compact.

---

## 7. Reproducibility is a filter, not a `checkout`

Rollout rows are **append-only and immutable** — no updates, no deletes. This changes how "exactly which rollouts trained checkpoint N" is answered, versus the earlier atomic-append design:

- MemWAL appends land in the `_mem_wal/` namespace and **do not advance the base dataset version.** So a per-`add` `checkout(version)` no longer isolates a single append the way plain `Dataset::append` did. `RolloutStore::add` still returns the base version, but for API compatibility only — not as a per-append snapshot handle.
- Reproducibility is instead a **filter over immutable rows** — e.g. `policy_version = 'ckpt-N'`, or an explicit set of `rollout_id`s recorded by the trainer. This is correct precisely *because* the rows never change: the same filter always selects the same bytes.

```python
rows = await store.list(filters={
    "policy_version": "ckpt-N",
    "include_in_training": True,
})
```

`checkout` remains available for base-table time travel, but is not the mechanism for per-checkpoint rollout reproducibility. (See schema-design §3 and §7 — and note that `learner_iteration` is likewise a column, not a dataset version.)

---

## 8. Artifacts are stored inline, not blob-offloaded

Schema-design §6 anticipated storing `binary_payload` as a **blob v2 offloaded** column. That is **incompatible with the MemWAL read path**, which is the only read path for rollouts: the LSM scanner has no blob-materialization step, so a blob-v2 (`lance-encoding:blob`) column reads back as `None`, and `get_blob` / the `…/blob` endpoint would return nothing. This resolves the schema-design §9 open item ("confirm the blob v2 encoding marker against the pinned Lance version"): **do not** use blob-v2 offload for rollout artifacts.

Artifacts are therefore stored as a plain inline `LargeBinary` column. The schema-design §2 property — *the learner does not pay for artifacts it does not read* — is preserved **columnar-ly** rather than by physical offload:

- `list` and `get_by_id` **project `binary_payload` out**, so bulk/hot-path scans never materialize artifact bytes.
- `get_blob` (HTTP `GET /rollouts/{name}/{id}/blob`) projects the column **in** and returns the bytes on demand.

From a client's perspective the behavior is unchanged: metadata scans are cheap, and artifact bytes are fetched explicitly by id.

---

## 9. Operational checklist

- [ ] Deploy as a **StatefulSet** (stable ordinal pod names), not a Deployment.
- [ ] Point `--data-dir` at **shared object storage** at the **same URI for every pod**; no per-pod PVC for the dataset.
- [ ] Ensure each instance has a **distinct stable id** — `INSTANCE_ID` from `metadata.name`, or rely on the pod `HOSTNAME`. Never reuse an id across live instances.
- [ ] Front the pods with an ordinary round-robin **Service / LB**; no session affinity or consistent hashing required.
- [ ] Size **replicas** to the ingest fan-in; scale up/down freely (§6).
- [ ] For long-running ingest, set **`--rollout-merge-after-generations N`** so each instance folds its own shard back into the base table and read cost stays bounded (§6.1). Leave `0` only for short-lived or low-volume stores.
- [ ] Reproduce training sets by **filtering immutable rows** (e.g. `policy_version`), not by `checkout` (§7).
