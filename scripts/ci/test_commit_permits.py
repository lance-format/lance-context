#!/usr/bin/env python3
"""Build and run commit-permit regressions against a private loopback etcd.

Requires cargo, curl, Python 3, and etcd (validated locally with 3.5.16).
No existing cluster endpoints or credentials are used. Output includes per-phase
RPC counts and fault-injection receipts; all child processes are joined on exit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--etcd', default=shutil.which('etcd'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.etcd:
        parser.error('etcd not found; specify --etcd /path/to/etcd')
    root = Path(__file__).resolve().parents[2]
    output = args.output or Path(tempfile.mkdtemp(prefix='commit-permit-rpc-'))
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error('output directory must be empty')
    output = output.resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ETCD', 'RK_'))}
    record = {
        'checkout': str(root),
        'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)).hexdigest(),
        'etcd_version': subprocess.check_output([args.etcd, '--version'], text=True).strip(),
    }
    # Build before launching the server; compilation must not consume its test budget.
    with (output / 'build.log').open('w') as log:
        build = subprocess.run([
            'cargo', 'test', '--locked', '-p', 'lance-context-merge',
            '--test', 'private_permit_rpc', '--no-run', '--message-format=json',
        ], cwd=root, env=env, stdout=subprocess.PIPE, stderr=log, text=True, timeout=600)
    if build.returncode:
        raise RuntimeError(f'build failed; see {output / "build.log"}')
    binaries = [json.loads(line)['executable'] for line in build.stdout.splitlines()
                if line.startswith('{') and json.loads(line).get('executable')]
    if len(binaries) != 1:
        raise RuntimeError(f'expected one test executable, got {binaries}')
    reservations = [socket.socket() for _ in range(3)]
    for reservation in reservations:
        reservation.bind(('127.0.0.1', 0))
    ports = [reservation.getsockname()[1] for reservation in reservations]
    for reservation in reservations:
        reservation.close()
    client, peer, proxy = [f'http://127.0.0.1:{port}' for port in ports]
    processes = []
    logs = []
    try:
        log = (output / 'etcd.log').open('w')
        logs.append(log)
        etcd = subprocess.Popen([
            args.etcd, '--name', 'private-permit', '--data-dir', str(output / 'data'),
            '--listen-client-urls', client, '--advertise-client-urls', client,
            '--listen-peer-urls', peer, '--initial-advertise-peer-urls', peer,
            '--initial-cluster', 'private-permit=' + peer,
            '--initial-cluster-token', output.name,
        ], env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(('etcd', etcd))
        for _ in range(100):
            if etcd.poll() is not None:
                raise RuntimeError('private etcd exited during startup')
            try:
                with urllib.request.urlopen(client + '/health', timeout=.5) as response:
                    if json.load(response)['health'] == 'true':
                        break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError('private etcd startup deadline')
        log = (output / 'proxy.log').open('w')
        logs.append(log)
        relay = subprocess.Popen([
            sys.executable, str(root / 'scripts/ci/private_etcd_response_proxy.py'),
            '--listen-port', str(ports[2]), '--upstream-port', str(ports[0]),
            '--state-dir', str(output),
        ], env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append(('proxy', relay))
        for _ in range(50):
            if relay.poll() is not None:
                raise RuntimeError('private proxy exited during startup')
            try:
                with socket.create_connection(('127.0.0.1', ports[2]), timeout=.1):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError('private proxy startup deadline')
        env.update(RK_PRIVATE_ETCD=client, RK_PRIVATE_PROXY=proxy,
                   RK_PRIVATE_PROXY_DIR=str(output), RK_PRIVATE_RESULT=str(output / 'result.json'))
        with (output / 'test.log').open('w') as log:
            result = subprocess.run([binaries[0], '--ignored', '--nocapture', '--test-threads=1'],
                                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=240)
        record['test_exit'] = result.returncode
    finally:
        for name, process in reversed(processes):
            record[name] = {'pid': process.pid, 'exit': stop(process), 'joined': True}
        for log in logs:
            log.close()
        record['finished'] = time.time()
        (output / 'process.json').write_text(json.dumps(record, indent=2))
        print(f'Commit permit artifacts: {output}', flush=True)
    if record['test_exit']:
        print((output / 'test.log').read_text()[-5000:])
    return record['test_exit']


if __name__ == '__main__':
    sys.exit(main())
