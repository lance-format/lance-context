"""Admission supervisor for externally managed persistent native publishers.

Every admission is bound to the actual Pod UID, not just its reusable name.
Data work and execution fencing remain in the native binary. This supervisor
never clears ownership, replays an uncertain attempt, or kills a native child.
Deploy only in a one-container Pod with restartPolicy Never; see native-publisher.md.
"""

import base64
import json
import os
import pathlib
import re
import signal
import subprocess
import threading
import time
import urllib.request
import uuid

NATIVE_BINARY = "/usr/local/bin/lance-context-master"
MEMORY_CURRENT = pathlib.Path("/sys/fs/cgroup/memory.current")
READY_FILE = pathlib.Path("/tmp/native-supervisor-ready")


def authorized(operator, job, pod_name, pod_uid):
    """Reject missing identity and stale authorization for a reused Pod name."""
    return bool(
        pod_uid
        and operator.get("phase") == "native_parallel_active"
        and operator.get("native_job") == job
        and operator.get("native_pod") == pod_name
        and operator.get("native_pod_uid") == pod_uid
    )


def main():
    UID = os.environ.get("POD_UID", "").strip()
    if not UID:
        raise SystemExit(
            "POD_UID from metadata.uid is required; refusing native admission"
        )
    T = os.environ["CATCHUP_TARGET"]
    J = os.environ["CATCHUP_JOB_NAME"]
    P = os.environ["ETCD_PREFIX"].rstrip("/")
    H = T.encode().hex()
    E = os.environ["ETCD_ENDPOINTS"].rstrip("/") + "/v3/"
    stop = threading.Event()
    proc = None
    enc = lambda s: base64.b64encode(s.encode()).decode()

    def log(event, **kw):
        print(
            json.dumps(
                {"at": time.time(), "event": event, "target": T, "job": J, **kw}
            ),
            flush=True,
        )

    def read(suffix):
        r = urllib.request.Request(
            E + "kv/range",
            data=json.dumps({"key": enc(P + "/" + suffix)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r, timeout=5) as f:
            k = json.load(f).get("kvs", [])
        return base64.b64decode(k[0]["value"]).decode() if k else None

    def cas_attempt(old, new, operator):
        if not authorized(json.loads(operator), J, os.environ["HOSTNAME"], UID):
            return None
        key = P + "/merge-rollout-attempts/" + H
        compares = [
            {
                "key": enc(P + "/catchup-active/" + H),
                "target": "VALUE",
                "result": "EQUAL",
                "value": enc(J),
            },
            {
                "key": enc(P + "/merge-rollout-operator/" + H),
                "target": "VALUE",
                "result": "EQUAL",
                "value": enc(operator),
            },
        ]
        compares.append(
            {"key": enc(key), "target": "VERSION", "result": "EQUAL", "version": "0"}
            if old is None
            else {
                "key": enc(key),
                "target": "VALUE",
                "result": "EQUAL",
                "value": enc(old),
            }
        )
        value = json.dumps(new, separators=(",", ":"))
        r = urllib.request.Request(
            E + "kv/txn",
            data=json.dumps(
                {
                    "compare": compares,
                    "success": [
                        {"request_put": {"key": enc(key), "value": enc(value)}}
                    ],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r, timeout=5) as f:
            out = json.load(f)
        return value if out.get("succeeded") else None

    def halt(*args):
        stop.set()
        log("admission_stopping_waiting_for_native_exit")

    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    READY_FILE.touch()
    log("standby")
    while not stop.is_set():
        try:
            operator = read("merge-rollout-operator/" + H)
            op = json.loads(operator or "{}")
            if (
                not authorized(op, J, os.environ["HOSTNAME"], UID)
                or read("catchup-active/" + H) != J
            ):
                stop.wait(10)
                continue
            # The admission supervisor does not bypass native persistent failure backoff.
            failure = json.loads(
                read("merge-failures/" + H + "/" + b"master:catchup".hex()) or "{}"
            )
            old_attempt = read("merge-rollout-attempts/" + H)
            previous = json.loads(old_attempt or "{}")
            if previous and previous.get("job") != J:
                raise RuntimeError("attempt ledger belongs to a different job")
            delay = max(
                0,
                max(failure.get("next_retry_ms", 0), previous.get("next_retry_ms", 0))
                / 1000
                - time.time(),
            )
            if delay > 0:
                log("persistent_backoff", retry_after_seconds=round(delay))
                stop.wait(min(delay, 60))
                continue
            if int(MEMORY_CURRENT.read_text()) > 12 * 2**30:
                log("memory_admission_delayed")
                stop.wait(15)
                continue
            # The native executable claims only J's target, reconciles old execution,
            # enforces shared byte reservations and idle progress, then commits watermarks.
            attempt = {
                "job": J,
                "pod_uid": UID,
                "id": str(uuid.uuid4()),
                "started_at": time.time(),
                "consecutive_failures": previous.get("consecutive_failures", 0) + 1,
                "state": "admitted",
            }
            attempt["next_retry_ms"] = int(
                (
                    time.time()
                    + min(3600, 60 * 2 ** min(attempt["consecutive_failures"], 6))
                )
                * 1000
            )
            attempt["needs_attention"] = attempt["consecutive_failures"] >= 3
            admitted = cas_attempt(old_attempt, attempt, operator)
            if admitted is None:
                stop.wait(10)
                continue
            started = time.monotonic()
            reclaimed = 0
            pass_errors = 0
            proc = subprocess.Popen(
                [NATIVE_BINARY],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            log("native_pass_started", pid=proc.pid)
            for line in proc.stdout:
                print(line.rstrip(), flush=True)
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
                if "parallel rollout append completed" in clean:
                    n = re.search(r"\breclaimed=(\d+)", clean)
                    if n:
                        reclaimed += int(n.group(1))
                    n = re.search(r"\bfailed=(\d+)", clean)
                    if n:
                        pass_errors += int(n.group(1))
            code = proc.wait()
            seconds = time.monotonic() - started
            proc = None
            terminal = dict(
                attempt,
                state="succeeded" if code == 0 else "failed",
                exit_code=code,
                finished_at=time.time(),
                reclaimed=reclaimed,
                seconds=seconds,
            )
            if code == 0:
                terminal.update(
                    consecutive_failures=0,
                    needs_attention=False,
                    next_retry_ms=int((time.time() + (2 if reclaimed else 30)) * 1000),
                )
            else:
                terminal["next_retry_ms"] = int(
                    (
                        time.time()
                        + min(3600, 60 * 2 ** min(attempt["consecutive_failures"], 6))
                    )
                    * 1000
                )
            if cas_attempt(admitted, terminal, operator) is None:
                log("attempt_record_changed_ownership_untouched")
            log(
                "native_pass_finished",
                exit_code=code,
                reclaimed=reclaimed,
                seconds=seconds,
                generations_per_second=reclaimed / seconds,
                failed_shards=pass_errors,
            )
            # Let ordinary flushes proceed; zero-work passes must not hammer etcd/catalogs.
            stop.wait(2 if code == 0 and reclaimed > 0 else 30)
        except Exception as e:  # noqa: BLE001 - preserve native child join on any supervisor error
            log("supervisor_error_ownership_untouched", error=repr(e))
            if proc is not None:
                # Do not launch a second child after a controller error. Wait for native exit.
                proc.wait()
                proc = None
            stop.wait(30)
    log("supervisor_stopped")


if __name__ == "__main__":
    main()
