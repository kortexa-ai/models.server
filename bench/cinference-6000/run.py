#!/usr/bin/env python3
"""Bounded cold-context comparison; owns only the server processes it starts."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / '.engines/cinference'
GPU = 'GPU-a71210ca-e14a-755a-88bb-77f53a2102f6'
PORT = 18080
URL = f'http://127.0.0.1:{PORT}'
MODEL = 'huihui-cinference-bench'


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def api(path, body=None, timeout=1800):
    request = urllib.request.Request(URL + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode()}') from exc


def gpu_snapshot(pid):
    raw = command('nvidia-smi', f'--id={GPU}',
        '--query-gpu=memory.used,memory.free,power.draw,power.limit,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits')
    used, free, power, limit, util, temp = [float(x.strip()) for x in raw.split(',')]
    apps = command('nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                   '--format=csv,noheader,nounits')
    process_mib = 0
    for row in apps.splitlines():
        uuid, process, memory = [x.strip() for x in row.split(',')]
        if int(process) == pid:
            if uuid != GPU:
                raise RuntimeError('Benchmark server appeared on the wrong GPU')
            process_mib += float(memory)
    return dict(time=time.time(), process_mib=process_mib, total_mib=used,
                free_mib=free, power_w=power, limit_w=limit, utilization=util,
                temperature_c=temp)


class Server:
    def __init__(self, args, directory, env):
        self.args, self.directory, self.env = args, directory, env
        self.proc = None
        self.samples = []
        self.monitor_error = None
        self.done = threading.Event()

    def __enter__(self):
        # Bind check is read-only; do not replace an existing listener.
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', PORT)) == 0:
                raise RuntimeError(f'Port {PORT} already occupied')
        self.directory.mkdir(parents=True, exist_ok=False)
        save(self.directory / 'command.json', self.args)
        save(self.directory / 'environment.json', {k: v for k, v in self.env.items()
             if k.startswith(('CUDA_', 'GGML_')) or k == 'LD_LIBRARY_PATH'})
        self.log = (self.directory / 'server.log').open('w')
        self.proc = subprocess.Popen(self.args, cwd=ROOT, env=self.env,
                                    stdout=self.log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f'Server exited {self.proc.returncode}: {self.directory}')
                try:
                    if api('/health', timeout=2).get('status') == 'ok':
                        self.idle = gpu_snapshot(self.proc.pid)
                        save(self.directory / 'idle.json', self.idle)
                        save(self.directory / 'live-process.json', {
                            'pid': self.proc.pid,
                            'argv': Path(f'/proc/{self.proc.pid}/cmdline').read_bytes().decode().split('\0'),
                            'executable': os.readlink(f'/proc/{self.proc.pid}/exe'),
                            'environment': [entry for entry in Path(f'/proc/{self.proc.pid}/environ').read_bytes().decode().split('\0')
                                            if entry.startswith(('CUDA_', 'GGML_', 'LLAMA_'))]})
                        print(f'READY {self.directory.name}: {self.idle}', flush=True)
                        return self
                except (OSError, RuntimeError):
                    pass
                time.sleep(1)
            raise RuntimeError('Server readiness timeout')
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def monitor(self):
        with (self.directory / 'gpu.jsonl').open('w') as stream:
            while not self.done.is_set():
                try:
                    sample = gpu_snapshot(self.proc.pid)
                    self.samples.append(sample)
                    stream.write(json.dumps(sample) + '\n')
                    stream.flush()
                    if sample['free_mib'] < 10240:
                        raise RuntimeError('Free VRAM crossed 10 GiB abort floor')
                except Exception as exc:
                    self.monitor_error = str(exc)
                    print(f'GPU MONITOR ABORT: {exc}', flush=True)
                    if self.proc.poll() is None:
                        os.killpg(self.proc.pid, signal.SIGTERM)
                    break
                self.done.wait(0.5)

    def __exit__(self, *_):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=30)
        self.done.set()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=10)
        if hasattr(self, 'log'):
            self.log.close()
        save(self.directory / 'memory-summary.json', {
            'samples': len(self.samples), 'monitor_error': self.monitor_error,
            'peak_process_mib': max((s['process_mib'] for s in self.samples), default=0),
            'peak_total_mib': max((s['total_mib'] for s in self.samples), default=0),
            'minimum_free_mib': min((s['free_mib'] for s in self.samples), default=0)})


def request(prompt, thinking=False, max_tokens=512):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'seed': 42, 'max_tokens': max_tokens,
            'reasoning_effort': 'medium' if thinking else 'none',
            'chat_template_kwargs': {'enable_thinking': thinking},
            'cache_prompt': False, 'stream': True,
            'stream_options': {'include_usage': True}}
    req = urllib.request.Request(URL + '/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    first = None
    content, reasoning = '', ''
    result = {'started_unix': time.time(), 'thinking': thinking}
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b'data: '):
                continue
            value = raw[6:].strip()
            if value == b'[DONE]':
                break
            event = json.loads(value)
            if 'error' in event:
                raise RuntimeError(event['error'])
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or ''
                thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                if (text or thought) and first is None:
                    first = time.monotonic()
                content += text
                reasoning += thought
                if choice.get('finish_reason'):
                    result['finish_reason'] = choice['finish_reason']
            for key in ('usage', 'timings'):
                if event.get(key):
                    result[key] = event[key]
    result.update(client_seconds=time.monotonic() - started,
                  client_ttft_seconds=None if first is None else first - started,
                  content=content, reasoning=reasoning,
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                  ended_unix=time.time())
    return result


def make_prompt(count, nonce, workload):
    markers = {int(count * x): f'KEY_{i}=VAL_{nonce}_{i}_7294' for i, x in
               enumerate((.01, .1, .25, .5, .75, .95))}
    lines = [f'{nonce} Archive of service observations. Each row is independent.']
    for i in range(count):
        if i in markers:
            lines.append(markers[i])
        lines.append(f'Record {i}: service shard {i % 97}; queue {i % 31}; '
                     f'latency {i % 211} ms; status healthy; retries {i % 5}.')
    if workload == 'recall':
        lines.append('Return all six KEY_0 through KEY_5 values exactly, one per line. No explanation.')
    else:
        lines.append('Write a detailed 1000-word engineering analysis of this service archive. '
                     'Discuss queueing, tail latency, retries, observability, and capacity planning. '
                     'Use full prose with concrete examples and avoid repeating the raw records.')
    return '\n'.join(lines), list(markers.values())


def sized_prompt(target, nonce, workload, stock=False):
    count = max(1, target // 32)
    for _ in range(5):
        prompt, markers = make_prompt(count, nonce, workload)
        if stock:
            measured = len(api('/tokenize', {'content': prompt, 'add_special': True})['tokens'])
        else:
            measured = api('/v1/messages/count_tokens', {'model': MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'thinking': {'type': 'disabled'}})['input_tokens']
        if abs(measured - target) < 100:
            return prompt, markers, measured
        count = max(1, int(count * (target - 30) / measured))
    return prompt, markers, measured


def record_request(server, directory, name, prompt, markers=(), thinking=False):
    result = request(prompt, thinking=thinking)
    result['name'] = name
    result['recall_expected'] = list(markers)
    result['recall_found'] = sum(m.split('=', 1)[1] in result['content'] for m in markers)
    samples = [s for s in server.samples
               if result['started_unix'] <= s['time'] <= result['ended_unix']]
    result['peak_process_mib'] = max((s['process_mib'] for s in samples), default=0)
    result['peak_total_mib'] = max((s['total_mib'] for s in samples), default=0)
    save(directory / f'{name}.json', result)
    print(json.dumps({k: v for k, v in result.items()
        if k not in ('content', 'reasoning', 'recall_expected', 'prompt_sha256')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', nargs='+', default=['stock', 'published', 'stock-like'])
    parser.add_argument('--output', default='bench-results/cinference-27')
    args = parser.parse_args()
    if socket.gethostname() != 'smarty' or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError('Must run through run-block.sh on the pinned Smarty 6000')
    if gpu_snapshot(-1)['free_mib'] < 80 * 1024:
        raise RuntimeError('Need 80 GiB free before the benchmark block')
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    model_config = json.loads((ROOT / 'qwen-3.8-27b/model.json').read_text())
    save(output / 'stock-model.json', model_config)
    manifest = json.loads((RUNTIME / 'runtime-manifest.json').read_text())
    artifact = RUNTIME / 'models' / manifest['models']['target']['directory'] / manifest['models']['target']['filename']
    env = os.environ.copy()
    env.pop('GGML_CUDA_ENABLE_UNIFIED_MEMORY', None)
    env['LD_LIBRARY_PATH'] = str(RUNTIME / '.cuda-toolkit/nvidia/cu13/lib')
    env['PATH'] = str(RUNTIME / '.venv/bin') + ':' + env['PATH']
    save(output / 'manifest.json', {'models_commit': command('git', 'rev-parse', 'HEAD'),
         'stock_config_sha256': hashlib.sha256((ROOT / 'qwen-3.8-27b/model.json').read_bytes()).hexdigest(),
         'llama_commit': command('git', '-C', str(ROOT.parent / 'llama.cpp'), 'rev-parse', 'HEAD'),
         'cinference_commit': command('git', '-C', str(RUNTIME / 'runtime/ninfer'), 'rev-parse', 'HEAD'),
         'runtime_manifest': manifest, 'gpu': command('nvidia-smi', f'--id={GPU}',
         '--query-gpu=name,uuid,driver_version,power.limit,memory.total', '--format=csv'),
         'restore_services': os.environ.get('SMARTY_GPU_RECORDED_SERVICES'),
         'host': socket.gethostname(), 'started_unix': time.time()})
    prompts = output / 'prompts'
    prompts.mkdir(exist_ok=True)
    for profile in args.profiles:
        directory = output / profile
        if profile == 'stock':
            config = env | {'PORT': str(PORT), 'HOST': '127.0.0.1'}
            stock_path = Path.home() / '.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502'
            server_args = ['bash', str(ROOT / 'run.sh'), 'qwen-3.8-27b',
                           '--alias', MODEL, '--model', str(stock_path / 'Qwen3.8-27B-UD-Q4_K_XL.gguf'),
                           '--mmproj', str(stock_path / 'mmproj-BF16.gguf'), '--offline']
        else:
            config = env
            deadline = time.monotonic() + 1800
            binary = RUNTIME / 'runtime/ninfer/build/apps/ninfer-serve'
            while not (artifact.exists() and binary.exists()):
                if time.monotonic() > deadline:
                    raise RuntimeError('Runtime/model preparation deadline exceeded')
                print('Waiting for the pinned runtime and model download.', flush=True)
                time.sleep(30)
            published = profile == 'published'
            server_args = [str(RUNTIME / 'runtime/ninfer/build/apps/ninfer-serve'),
                str(artifact), '--host', '127.0.0.1', '--port', str(PORT),
                '--model-id', MODEL, '--max-context', '262144' if published else '524288',
                '--kv-capacity', '262144' if published else '524288',
                '--max-concurrency', '1' if published else '8',
                '--kv-dtype', 'k8v4' if published else 'int8', '--prefill-chunk', '1024',
                '--spec', 'mtp', '--draft-tokens', '10' if published else '3',
                '--lm-head-draft', '--preserve-thinking', '--request-log-jsonl',
                str(directory / 'requests.jsonl')]
            if not published:
                server_args += ['--vision']
        try:
            with Server(server_args, directory, config) as server:
                canary = request('What is 7 plus 5? Answer with only the number.', max_tokens=32)
                save(directory / 'canary.json', canary)
                if '12' not in canary['content']:
                    raise RuntimeError('Arithmetic canary failed')
                for target in (8192, 32768, 131072, 260000):
                    for workload in ('recall', 'prose'):
                        name = f'{workload}-{target}'
                        prompt_file = prompts / f'{name}.json'
                        if not prompt_file.exists():
                            text, markers, tokens = sized_prompt(target, f'run{target}{workload}', workload,
                                                                 stock=profile == 'stock')
                            save(prompt_file, {'prompt': text, 'markers': markers if workload == 'recall' else [],
                                               'count_endpoint_tokens': tokens})
                        fixture = json.loads(prompt_file.read_text())
                        record_request(server, directory, name, fixture['prompt'], fixture['markers'])
                # Match the production medium-thinking request mode separately.
                fixture = json.loads((prompts / 'prose-32768.json').read_text())
                record_request(server, directory, 'medium-thinking-32768',
                               fixture['prompt'] + '\nCompare two plausible capacity plans.', thinking=True)
                if profile != 'published':
                    # One request may use most of the unified 512K allocation.
                    path = prompts / 'recall-500000.json'
                    if not path.exists():
                        text, markers, tokens = sized_prompt(500000, 'run500000recall', 'recall',
                                                            stock=profile == 'stock')
                        save(path, {'prompt': text, 'markers': markers, 'count_endpoint_tokens': tokens})
                    fixture = json.loads(path.read_text())
                    try:
                        record_request(server, directory, 'recall-500000', fixture['prompt'], fixture['markers'])
                    except (RuntimeError, urllib.error.HTTPError) as exc:
                        save(directory / 'recall-500000-error.json', {'error': str(exc)})
                        print(f'512K ceiling probe rejected: {exc}', flush=True)
                    # Exercise all eight lanes with a bounded shared-KV load.
                    fixture = json.loads((prompts / 'prose-8192.json').read_text())
                    started = time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                        futures = [pool.submit(record_request, server, directory, f'concurrent-{i}',
                            f'Independent client {i}.\n' + fixture['prompt']) for i in range(8)]
                        batch = [future.result() for future in futures]
                    elapsed = time.monotonic() - started
                    save(directory / 'concurrency.json', {'clients': 8, 'seconds': elapsed,
                         'completion_tokens': sum(x['usage']['completion_tokens'] for x in batch),
                         'aggregate_completion_tps': sum(x['usage']['completion_tokens'] for x in batch) / elapsed})
                if server.monitor_error:
                    raise RuntimeError(server.monitor_error)
        except Exception as exc:
            save(output / f'{profile}-error.json', {'error': str(exc)})
            print(f'PROFILE FAILED {profile}: {exc}', flush=True)
            if 'abort floor' in str(exc):
                raise
        time.sleep(3)


if __name__ == '__main__':
    def stop(signum, _frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    main()
