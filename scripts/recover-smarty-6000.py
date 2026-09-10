"""One explicitly authorized, UUID-scoped recovery; no reboot fallback.

Default mode captures a read-only baseline. --execute enables the closed reset
procedure. Use only with separate operator authorization and an active claim.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

GPU = 'GPU-a71210ca-e14a-755a-88bb-77f53a2102f6'
SERVICES = [
    ('comfyui.server', 8050, '/system_stats'),
    ('alt-image-gen.server/base', 4004, '/health'),
    ('models/qwen-3.8-27b', 2053, '/health'),
    ('models/lfm2.5-vl-3b', 2055, '/health'),
    ('models/hy-mt2-7b', 2060, '/health'),
    ('vision.server', 4001, '/health'),
]
events = []


def record(**row):
    events.append(dict(time=time.time(), **row))
    print(json.dumps(events[-1]), flush=True)


def run(argv, timeout=45, check=False):
    start = time.time()
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        row = dict(command=argv, returncode=result.returncode, stdout=result.stdout,
                   stderr=result.stderr, seconds=time.time()-start)
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
        row = dict(command=argv, returncode=124, stdout=text(exc.stdout),
                   stderr=text(exc.stderr), seconds=time.time()-start)
    record(**row)
    if check and row['returncode']:
        raise RuntimeError('Command failed: ' + ' '.join(argv))
    return row


def health(port, path):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=3) as r:
            return dict(status=r.status, body=r.read().decode())
    except Exception as exc:
        return dict(status=0, error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] == 'smarty'
    assert sys.platform == 'linux'
    assert Path('/sys/bus/pci/devices/0000:01:00.0/reset_method').read_text().split()[0] == 'flr'
    run(['nvidia-smi', '-L'], check=True)
    run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv'])
    run(['sudo', 'fuser', '-v', '/dev/nvidia0', '/dev/nvidia1'])
    sessions = run(['loginctl', 'list-sessions', '--no-legend'], check=True)['stdout']
    greeters = []
    for line in sessions.splitlines():
        parts = line.split()
        if 'seat0' in parts:
            assert parts[2] == 'gdm', 'Do not interrupt a graphical user session'
            details = run(['loginctl', 'show-session', parts[0], '-p', 'Class', '-p', 'Name'], check=True)['stdout']
            assert 'Class=greeter' in details and 'Name=gdm' in details
            greeters.append(parts[0])
    greeter_running = run(['systemctl', 'is-active', 'gdm3'])['stdout'].strip() == 'active'
    restore = []
    for service, port, path in SERVICES:
        status = run(['ktxsvc', 'status', service])['stdout']
        active = 'Active: active (running)' in status
        pid = re.search(r'Main PID: (\d+)', status)
        if active:
            assert pid, 'A running managed service must have a known PID'
            env = (Path('/proc') / pid[1] / 'environ').read_bytes().split(b'\0')
            assert ('CUDA_VISIBLE_DEVICES=' + GPU).encode() in env, service
            restore.append((service, port, path, int(pid[1])))
        record(service=service, was_running=active, health=health(port, path))
    record(restore_set=restore, restore_greeter=greeter_running, execute=args.execute)
    if not args.execute:
        return
    stopped = []
    greeter_stopped = False
    reset_ok = False
    try:
        for service, port, path, pid in restore:
            stopped.append((service, port, path, pid))
            run(['ktxsvc', 'stop', service], check=True)
        if greeter_running:
            greeter_stopped = True
            # GDM is an OS service, not a managed Kortexa service.
            run(['sudo', 'systemctl', 'stop', 'gdm3'], check=True)
        run(['sudo', 'fuser', '-v', '/dev/nvidia0', '/dev/nvidia1'])
        result = run(['sudo', 'nvidia-smi', '--gpu-reset', '-i', GPU], timeout=30)
        reset_ok = result['returncode'] == 0
        if reset_ok:
            run(['sudo', 'nvidia-smi', '-i', GPU, '--power-limit=450'])
            code = ('import torch; assert torch.cuda.device_count()==1; '
                    'assert "RTX PRO 6000" in torch.cuda.get_device_name(0); '
                    'x=torch.arange(16,device="cuda"); '
                    'assert (x*x).sum().item()==1240; '
                    'print(torch.cuda.get_device_name(0),torch.cuda.mem_get_info(),"CUDA PASS")')
            run(['env', 'CUDA_VISIBLE_DEVICES='+GPU,
                 str(Path.home()/'src/hamster-orchestra/.venv/bin/python'), '-c', code], check=True)
        else:
            record(reset_succeeded=False, fallback='No broader reset is authorized by this script')
    finally:
        # Even a failed reset or an interrupted stop must attempt restoration.
        for service, port, path, pid in stopped:
            run(['ktxsvc', 'start', service], timeout=45)
        if greeter_stopped:
            run(['sudo', 'systemctl', 'start', 'gdm3'], timeout=30)
        deadline = time.monotonic()+120
        pending = {service: (port, path, pid) for service, port, path, pid in stopped}
        while pending and time.monotonic() < deadline:
            for service, (port, path, pid) in list(pending.items()):
                result = health(port, path)
                if result['status'] == 200:
                    record(restored_service=service, original_pid=pid, health=result)
                    del pending[service]
            if pending:
                time.sleep(3)
        for service, port, path, pid in stopped:
            run(['ktxsvc', 'status', service])
        run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv'])
        record(reset_succeeded=reset_ok, services_without_http_health=list(pending))


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    main()
