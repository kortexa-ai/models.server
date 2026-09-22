#!/usr/bin/env python3
"""Compare cold code generation and short, reasoning-disabled conversation."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import run as bench


VOICE = [
    'Coffee question: explain why coffee tastes bitter, in two short spoken sentences.',
    'Garden question: suggest a simple way to remember to water my basil, in two short spoken sentences.',
    'Computer question: explain what a cache is using a kitchen analogy, in two short spoken sentences.',
    'Dinner question: I have rice, eggs and peas. Suggest a quick meal in two short spoken sentences.',
    'Rest question: suggest a five-minute break after a long coding session, in two short spoken sentences.',
]
CODE_TASK = '''
The repository above is source context for an inference service. Implement a
standalone Python 3 metrics reader for its benchmark request records. Each JSONL
row may contain name, usage.prompt_tokens, usage.completion_tokens,
timings.cache_n, timings.prompt_ms, timings.predicted_ms, and
client_ttft_seconds. Use only the standard library.
Provide a RequestMetrics dataclass, parse_record, read_jsonl, percentile,
summarize, and a CLI entry point accepting one input path. Skip malformed JSON
and incomplete records with a warning to stderr. Validate that durations are
positive, token counts are nonnegative integers, and numbers are finite.
Report median and p95 TTFT, aggregate prompt and completion tokens per second,
and cache-hit fraction. Handle an empty input explicitly. Use interpolation
for percentiles and type hints. Output only the complete Python code.
'''


def source_corpus():
    repo = bench.RUNTIME / 'runtime/ninfer'
    paths = [repo / 'docs/serving.md'] + sorted(
        p for folder in ('include', 'src') for p in (repo / folder).rglob('*')
        if p.is_file() and p.suffix in ('.h', '.cpp', '.cu', '.cuh'))
    inventory, parts = [], []
    for path in paths:
        data = path.read_bytes()
        inventory.append({'path': str(path.relative_to(repo)),
                          'sha256': hashlib.sha256(data).hexdigest()})
        parts.append(f'\nFILE {path.relative_to(repo)}\n' + data.decode())
    return '\n'.join(parts), inventory


def main(campaign):
    output = campaign / 'workloads'
    output.mkdir(exist_ok=False)
    fixtures = output / 'prompts'
    fixtures.mkdir()
    bench.MODEL = 'vanilla-qwen-workload-bench'
    bench.SAMPLING_OVERRIDES = {'presence_penalty': 0.0, 'frequency_penalty': 0.0,
                              'top_p': 0.95, 'top_k': 20}
    env = os.environ.copy()
    env.pop('GGML_CUDA_ENABLE_UNIFIED_MEMORY', None)
    env['LD_LIBRARY_PATH'] = str(bench.RUNTIME / '.cuda-toolkit/nvidia/cu13/lib')
    env['PATH'] = str(bench.RUNTIME / '.venv/bin') + ':' + env['PATH']
    corpus, inventory = source_corpus()
    bench.save(output / 'source-inventory.json', inventory)
    bench.save(output / 'manifest.json', {
        'models_commit': bench.command('git', '-C', str(bench.ROOT), 'rev-parse', 'HEAD'),
        'stock_manifest': json.loads((campaign / 'manifest.json').read_text()),
        'vanilla_manifest': json.loads((campaign / 'vanilla/manifest.json').read_text()),
        'sampling_overrides': bench.SAMPLING_OVERRIDES,
        'source_corpus_sha256': hashlib.sha256(corpus.encode()).hexdigest(),
        'restore_services': env.get('SMARTY_GPU_RECORDED_SERVICES'),
        'started_unix': time.time()})
    results = {}

    def coding_prompt(target):
        path = fixtures / f'coding-{target}.json'
        if not path.exists():
            count = min(len(corpus), target * 3)
            for _ in range(8):
                prompt = f'Coding trial {target}. Read this repository source context.\n' + corpus[:count] + CODE_TASK
                tokens = len(bench.api('/tokenize', {'content': prompt, 'add_special': True})['tokens'])
                if abs(tokens - target) < 80:
                    break
                count = min(len(corpus), max(1, int(count * (target - 100) / tokens)))
            if abs(tokens - target) > 300:
                raise RuntimeError(f'Cannot size code fixture to {target}: {tokens}')
            bench.save(path, {'prompt': prompt, 'count_endpoint_tokens': tokens})
        return json.loads(path.read_text())['prompt']

    def profile(name, argv, code_sizes, voice=False, code_nothink=False):
        directory = output / name
        records = []
        with bench.Server(argv, directory, env | {'PORT': str(bench.PORT), 'HOST': '127.0.0.1'}) as server:
            canary = bench.request('What is 7 plus 5? Answer with only the number.', max_tokens=32)
            bench.save(directory / 'canary.json', canary)
            if canary['content'].strip() != '12':
                raise RuntimeError('Workload canary failed')
            if name == 'stock':
                bench.save(directory / 'props.json', bench.api('/props'))
                executable = Path(f'/proc/{server.proc.pid}/exe').resolve()
                (directory / 'executable.sha256').write_text(
                    hashlib.sha256(executable.read_bytes()).hexdigest() + '\n')
            if voice:
                for i, prompt in enumerate(VOICE):
                    records.append(bench.record_request(server, directory, f'voice-{i}', prompt, max_tokens=128))
            for size in code_sizes:
                records.append(bench.record_request(server, directory, f'coding-{size}',
                    coding_prompt(size), thinking=True, max_tokens=2048))
            if code_nothink:
                records.append(bench.record_request(server, directory, 'coding-nothink-131072',
                    'Direct code trial.\n' + coding_prompt(131072), max_tokens=1024))
            if server.monitor_error:
                raise RuntimeError(server.monitor_error)
        results[name] = records
        time.sleep(3)
        return records

    stock_path = Path.home() / '.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502'
    stock_argv = ['bash', str(bench.ROOT / 'run.sh'), 'qwen-3.8-27b', '--alias', bench.MODEL,
                  '--model', str(stock_path / 'Qwen3.8-27B-UD-Q4_K_XL.gguf'),
                  '--mmproj', str(stock_path / 'mmproj-BF16.gguf'), '--offline']
    profile('stock', stock_argv, (131072, 260000), voice=True, code_nothink=True)
    chunk = json.loads((campaign / 'vanilla/chunk-selection.json').read_text())['selected']
    candidates = (f'stock-like-{chunk}', 'tuned-k8v4-mtp3', 'tuned-mtp', 'tuned-dflash2')

    def native_args(source, name):
        argv = json.loads((campaign / 'vanilla' / source / 'command.json').read_text())
        argv[argv.index('--model-id') + 1] = bench.MODEL
        argv[argv.index('--request-log-jsonl') + 1] = str(output / name / 'requests.jsonl')
        return argv

    for name in candidates:
        profile(name, native_args(name, name), (131072,), voice=True)
    coding = {name: next(r for r in results[name] if r['name'] == 'coding-131072') for name in candidates}
    fastest = min(r['timings']['prompt_ms'] for r in coding.values())
    eligible = [name for name, r in coding.items() if r['timings']['prompt_ms'] <= fastest * 1.05]
    winner = max(eligible, key=lambda name: coding[name]['timings']['predicted_per_second'])
    voice_latency = {name: statistics.median(r['client_ttft_seconds'] for r in results[name]
                     if r['name'].startswith('voice-')) for name in candidates}
    bench.save(output / 'selection.json', {'coding_selected': winner,
        'criterion': 'minimum 131K cold prefill; fastest code decode among profiles within 5%',
        'voice_median_ttft_seconds': voice_latency})
    print(f'Coding selection: {winner}; voice median TTFT: {voice_latency}', flush=True)
    profile('coding-selected-260K', native_args(winner, 'coding-selected-260K'), (260000,), code_nothink=True)
