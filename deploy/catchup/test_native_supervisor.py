"""Process/real-etcd admission regressions; only disposable loopback fixtures.

ETCD_BIN must name an etcd binary. Native work is a recording, blocking stub;
these tests prove supervisor admission/CAS/join behavior, not Lance throughput.
"""

import base64
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import uuid
from pathlib import Path


def enc(value):
    return base64.b64encode(value.encode()).decode()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class SupervisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("ETCD_BIN") or shutil.which("etcd")
        if not binary:
            raise unittest.SkipTest("set ETCD_BIN to run the real-etcd regressions")
        cls.directory = tempfile.TemporaryDirectory(prefix="native-supervisor-")
        cls.endpoint = "http://127.0.0.1:" + str(free_port())
        peer = "http://127.0.0.1:" + str(free_port())
        cls.etcd_log = cls.enterClassContext(
            open(Path(cls.directory.name) / "etcd.log", "w")
        )
        cls.etcd = subprocess.Popen(
            [
                binary,
                "--name",
                "test",
                "--data-dir",
                cls.directory.name + "/data",
                "--listen-client-urls",
                cls.endpoint,
                "--advertise-client-urls",
                cls.endpoint,
                "--listen-peer-urls",
                peer,
                "--initial-advertise-peer-urls",
                peer,
                "--initial-cluster",
                "test=" + peer,
            ],
            stdout=cls.etcd_log,
            stderr=cls.etcd_log,
        )
        try:
            for _ in range(100):
                if cls.etcd.poll() is not None:
                    raise RuntimeError("disposable etcd exited")
                try:
                    cls.rpc("kv/range", {"key": enc("/health")})
                    return
                except OSError:
                    time.sleep(0.1)
            raise RuntimeError("disposable etcd startup timed out")
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.etcd.terminate()
        cls.etcd.wait(timeout=10)
        cls.etcd_log.close()
        cls.directory.cleanup()

    @classmethod
    def rpc(cls, path, body):
        request = urllib.request.Request(
            cls.endpoint + "/v3/" + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    def setUp(self):
        self.folder = Path(self.directory.name) / uuid.uuid4().hex
        self.folder.mkdir()
        self.prefix = "/tests/native-supervisor/" + self.folder.name
        self.keys = {
            name: self.prefix + "/" + suffix
            for name, suffix in {
                "owner": "catchup-active/" + b"table".hex(),
                "operator": "merge-rollout-operator/" + b"table".hex(),
                "attempt": "merge-rollout-attempts/" + b"table".hex(),
                "failure": "merge-failures/"
                + b"table".hex()
                + "/"
                + b"master:catchup".hex(),
            }.items()
        }
        self.process = None
        self.proxy = None
        self.logs = self.enterContext(open(self.folder / "supervisor.log", "w"))
        self.stub = self.folder / "native"
        self.stub.write_text(
            "#!" + sys.executable + "\n"
            "import os,pathlib,time\n"
            'p=pathlib.Path(os.environ["FIXTURE_DIR"])\n'
            'with (p/"started").open("a") as f:f.write(str(os.getpid())+"\\n")\n'
            "deadline=time.monotonic()+10\n"
            'while not (p/"release").exists() and time.monotonic()<deadline:time.sleep(.02)\n'
            'print("parallel rollout append completed reclaimed=1 failed=0",flush=True)\n'
        )
        self.stub.chmod(0o700)
        (self.folder / "memory").write_text("0")
        self.operator = {
            "phase": "native_parallel_active",
            "native_job": "persistent-job",
            "native_pod": "publisher",
            "native_pod_uid": "uid-a",
        }
        self.put("owner", "persistent-job")
        self.put("operator", json.dumps(self.operator))

    def tearDown(self):
        (self.folder / "release").touch()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=15)
        if self.proxy:
            self.proxy.shutdown()
            self.proxy.server_close()
            self.proxy_thread.join(timeout=5)
        self.logs.close()

    def put(self, name, value):
        self.rpc("kv/put", {"key": enc(self.keys[name]), "value": enc(value)})

    def read(self, name):
        rows = self.rpc("kv/range", {"key": enc(self.keys[name])}).get("kvs", [])
        return base64.b64decode(rows[0]["value"]).decode() if rows else None

    def wait_for(self, condition, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.03)
        self.fail(
            "condition timed out; supervisor logs:\n"
            + (self.folder / "supervisor.log").read_text()
        )

    def start(self, uid="uid-a", endpoint=None):
        env = {
            **os.environ,
            "CATCHUP_TARGET": "table",
            "CATCHUP_JOB_NAME": "persistent-job",
            "ETCD_PREFIX": self.prefix,
            "ETCD_ENDPOINTS": endpoint or self.endpoint,
            "HOSTNAME": "publisher",
            "FIXTURE_DIR": str(self.folder),
            "POD_UID": uid,
        }
        module = str(Path(__file__).with_name("native_supervisor.py"))
        # Test-only overrides isolate memory/readiness paths and replace native
        # storage work with the recording stub; deployed CLI has no such flags.
        code = (
            "import importlib.util,pathlib,sys; "
            's=importlib.util.spec_from_file_location("supervisor",sys.argv[1]); '
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
            'p=pathlib.Path(sys.argv[2]);m.NATIVE_BINARY=str(p/"native"); '
            'm.MEMORY_CURRENT=p/"memory";m.READY_FILE=p/"ready";m.main()'
        )
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", code, module, str(self.folder)],
            env=env,
            stdout=self.logs,
            stderr=self.logs,
        )

    def assert_no_attempt(self):
        self.assertIsNone(self.read("attempt"))
        self.assertFalse((self.folder / "started").exists())

    def test_missing_pod_uid_exits_before_readiness_or_admission(self):
        self.start(uid="")
        self.assertNotEqual(self.process.wait(timeout=5), 0)
        self.assertFalse((self.folder / "ready").exists())
        self.assert_no_attempt()

    def test_same_name_foreign_uid_cannot_use_stale_activation(self):
        # Controller observed uid-a, but uid-b now occupies the same Pod name.
        # Even a successful activation CAS retaining uid-a must not admit uid-b.
        attempt = json.dumps(
            {"job": "persistent-job", "state": "failed", "next_retry_ms": 0}
        )
        failure = json.dumps({"next_retry_ms": 0, "last_error": "retained"})
        self.put("attempt", attempt)
        self.put("failure", failure)
        self.start(uid="uid-b")
        self.wait_for(lambda: (self.folder / "ready").exists())
        time.sleep(0.3)
        self.assertFalse((self.folder / "started").exists())
        self.assertEqual(self.read("attempt"), attempt)
        self.assertEqual(self.read("failure"), failure)
        self.assertEqual(self.read("owner"), "persistent-job")

    def test_missing_operator_uid_is_not_legacy_authorization(self):
        self.operator.pop("native_pod_uid")
        self.put("operator", json.dumps(self.operator))
        self.start()
        self.wait_for(lambda: (self.folder / "ready").exists())
        time.sleep(0.3)
        self.assert_no_attempt()

    def test_matching_uid_admits_and_sigterm_joins_native_child(self):
        self.start()
        self.wait_for(lambda: (self.folder / "started").exists())
        admitted = json.loads(self.read("attempt"))
        self.assertEqual(admitted["pod_uid"], "uid-a")
        self.assertEqual(admitted["state"], "admitted")
        self.process.terminate()
        time.sleep(0.2)
        self.assertIsNone(
            self.process.poll(), "supervisor exited before joining its native child"
        )
        (self.folder / "release").touch()
        self.assertEqual(self.process.wait(timeout=5), 0)
        terminal = json.loads(self.read("attempt"))
        self.assertEqual(terminal["state"], "succeeded")
        self.assertEqual(terminal["reclaimed"], 1)
        self.assertEqual(terminal["id"], admitted["id"])
        self.assertEqual(len((self.folder / "started").read_text().splitlines()), 1)

    def test_identity_is_rechecked_on_next_admission(self):
        self.start()
        self.wait_for(lambda: (self.folder / "started").exists())
        (self.folder / "release").touch()
        self.wait_for(lambda: json.loads(self.read("attempt"))["state"] == "succeeded")
        terminal = self.read("attempt")
        self.operator["native_pod_uid"] = "uid-b"
        self.put("operator", json.dumps(self.operator))
        time.sleep(2.3)
        self.assertEqual(len((self.folder / "started").read_text().splitlines()), 1)
        self.assertEqual(self.read("attempt"), terminal)

    def test_matching_uid_preserves_failure_and_attempt_backoff(self):
        deadline = int((time.time() + 20) * 1000)
        attempt = json.dumps(
            {"job": "persistent-job", "state": "failed", "next_retry_ms": deadline}
        )
        failure = json.dumps(
            {"next_retry_ms": deadline + 1000, "last_error": "retained"}
        )
        self.put("attempt", attempt)
        self.put("failure", failure)
        self.start()
        self.wait_for(
            lambda: "persistent_backoff" in (self.folder / "supervisor.log").read_text()
        )
        self.assertEqual(self.read("attempt"), attempt)
        self.assertEqual(self.read("failure"), failure)
        self.assertFalse((self.folder / "started").exists())

    def test_operator_change_after_uid_check_is_rejected_by_full_cas(self):
        fixture = self
        raced = threading.Event()

        class Proxy(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/v3/kv/txn" and not raced.is_set():
                    # Change only a non-identity operator field. Comparing just
                    # UID/name/job would admit stale lifecycle authorization.
                    fixture.operator["recreation_token"] = "changed-after-read"
                    fixture.put("operator", json.dumps(fixture.operator))
                    raced.set()
                result = fixture.rpc(self.path.removeprefix("/v3/"), body)
                payload = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self.proxy = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        self.proxy_thread = threading.Thread(
            target=self.proxy.serve_forever, daemon=True
        )
        self.proxy_thread.start()
        self.start(endpoint="http://127.0.0.1:" + str(self.proxy.server_port))
        self.assertTrue(raced.wait(timeout=5))
        time.sleep(0.3)
        self.assert_no_attempt()
        self.assertEqual(self.read("owner"), "persistent-job")


if __name__ == "__main__":
    unittest.main()
