# Persistent native publisher admission

`native_supervisor.py` tracks the existing external persistent-publisher
supervisor alongside the native catch-up deployment assets. It adds Pod UID
admission fencing. It does not replace or enable the master's native Kubernetes
Job controller described in `docs/master-catchup.md`, and does not create, delete,
restart, or take ownership of any Pod or table.

A Kubernetes Pod name can be reused. Previously a replacement under the same
name could accept an operator record intended for its predecessor. Before every
attempt, this supervisor now requires all of:

- operator `phase` is `native_parallel_active`;
- `native_job` and the exact persistent `catchup-active/<target hex>` owner match
  `CATCHUP_JOB_NAME`;
- `native_pod` matches the Pod name;
- `native_pod_uid` matches `POD_UID`, supplied by the downward API.

A missing local UID fails startup before readiness. A missing or mismatched
authorized UID leaves the supervisor in standby without touching the attempt or
failure ledger. The attempt CAS still compares the **entire raw operator value**,
the owner, and the old attempt. A concurrent lifecycle change after the UID check
therefore rejects admission. Attempts include the admitted `pod_uid` for diagnosis.

## Deployment contract

Use `native-publisher.example.json` as the standalone Pod shape. Mount an immutable,
content-addressed ConfigMap containing this script, and pin a tested image with
Python 3 and `/usr/local/bin/lance-context-master`. Do not feed this standalone
Pod manifest into `CATCHUP_POD_TEMPLATE`, which accepts the native Job controller's
different PodSpec contract.

**Require one container and `restartPolicy: Never`, and verify both on the live
Pod before activation.** The Pod UID then identifies one supervisor incarnation.
An `Always` or `OnFailure` Pod can restart its container under the same UID and is
not supported by this protocol. The example uses the downward API for both UID
and name; do not supply a literal UID or reuse another Pod's value. The supervisor
has no Kubernetes credential and cannot independently inspect its restart policy.

The reviewed environment/Secret must configure the target, persistent job,
storage, etcd prefix/endpoints, worker endpoints, historical shard identities,
owned target, append settings and byte budgets for the native binary. In
particular `MERGE_OWNED_TARGETS` must include `CATCHUP_TARGET`. Size memory limits
above the supervisor's existing 12 GiB admission watermark; that watermark is not
a bound on one admitted operation. Keep native byte reservations and Lance memory
limits in force. Configure `CATCHUP_SLICE_SECS` as a soft admission slice and allow
already admitted native work to join during graceful termination.

Create each new Pod in standby, read its actual Ready UID/container identity, then
CAS the operator to authorize that UID while preserving the persistent owner and
all attempt/failure records. A same-name foreign Pod created between observation
and activation cannot use the stale UID, even if the activation CAS succeeds.
Do not activate a replacement until the predecessor has qualified terminal/join
evidence and native execution recovery has completed. Pod disappearance, an API
timeout, or an expired lease alone is not that evidence.

The wrapper retains the existing backoff and child-join behavior: SIGTERM stops
new admissions and waits for the native child; unknown outcomes do not clear
ownership or bypass the native execution fence. The native executable remains
responsible for claims, cancellation, memory reservations, storage barriers and
watermarks. This script has no general automatic recreation loop. Migration to
the native Job controller remains separate work.

## Tests

```sh
ETCD_BIN=/path/to/etcd python3 -m unittest discover \
  -s deploy/catchup -p 'test_native_supervisor.py' -v
```

Tests run real supervisor processes and a disposable loopback-only etcd. The
native child is a recording stub: they prove stale UID rejection, full-operator
CAS races, subsequent admission checks, preserved backoff and joined shutdown,
not Lance payload correctness or Kubernetes lifecycle behavior. Validate the
changed supervisor separately with real Kubernetes downward API identities and
the native binary against isolated storage before production migration.
