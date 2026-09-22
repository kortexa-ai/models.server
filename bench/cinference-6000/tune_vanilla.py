#!/usr/bin/env python3
"""Vanilla checkpoint: bounded prefill-chunk and speculative-backend comparison."""
import concurrent.futures
import json
import os
from pathlib import Path
import time

import run as bench


def main(campaign):
    bench.MODEL = 'vanilla-qwen-cinference-bench'
    artifact_dir = bench.RUNTIME / 'models/Qwen3.8-27B-nvfp4-NInfer'
    expected = '74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82'
    if (artifact_dir / 'verified.sha256').read_text().strip() != expected:
        raise RuntimeError('Vanilla artifact has not passed the pinned checksum')
    artifact = artifact_dir / 'qwen3_8_27b_nvfp4.ninfer'
    output = campaign / 'vanilla'
    output.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.pop('GGML_CUDA_ENABLE_UNIFIED_MEMORY', None)
    env['LD_LIBRARY_PATH'] = str(bench.RUNTIME / '.cuda-toolkit/nvidia/cu13/lib')
    bench.save(output / 'manifest.json', {
        'models_commit': bench.command('git', '-C', str(bench.ROOT), 'rev-parse', 'HEAD'),
        'runtime_commit': bench.command('git', '-C', str(bench.RUNTIME / 'runtime/ninfer'), 'rev-parse', 'HEAD'),
        'model_repo': 'neroued/Qwen3.8-27B-nvfp4-NInfer',
        'model_revision': 'f0b43ad436b9fa8142c6ed6647c470a6fe409484',
        'model_sha256': expected, 'gpu_uuid': bench.GPU,
        'restore_services': env.get('SMARTY_GPU_RECORDED_SERVICES'),
        'started_unix': time.time(),
        'artifact_manifest': json.loads((artifact_dir / 'artifact-manifest.json').read_text())})
    fixtures = {path.stem: json.loads(path.read_text()) for path in (campaign / 'prompts').glob('*.json')}
    results = {}

    def profile(name, chunk, slots, kv, backend, drafts, vision, cases, extra=False):
        directory = output / name
        argv = [str(bench.RUNTIME / 'runtime/ninfer/build/apps/ninfer-serve'), str(artifact),
            '--host', '127.0.0.1', '--port', str(bench.PORT), '--model-id', bench.MODEL,
            '--max-context', '262144', '--kv-capacity', '524288' if slots == 8 else '262144',
            '--max-concurrency', str(slots), '--kv-dtype', kv,
            '--prefill-chunk', str(chunk), '--spec', backend, '--draft-tokens', str(drafts),
            '--lm-head-draft', '--preserve-thinking', '--request-log-jsonl',
            str(directory / 'requests.jsonl')]
        if vision:
            argv.append('--vision')
        records = []
        with bench.Server(argv, directory, env) as server:
            canary = bench.request('What is 7 plus 5? Answer with only the number.', max_tokens=32)
            bench.save(directory / 'canary.json', canary)
            if '12' not in canary['content']:
                raise RuntimeError('Vanilla arithmetic canary failed')
            for case in cases:
                fixture = fixtures[case]
                records.append(bench.record_request(server, directory, case, fixture['prompt'], fixture['markers']))
            if extra:
                fixture = fixtures['prose-32768']
                records.append(bench.record_request(server, directory, 'medium-thinking-32768',
                    'Reasoning trial.\n' + fixture['prompt'], thinking=True))
                # A fresh prefix makes this a second cold long-context observation.
                fixture = fixtures['prose-260000']
                records.append(bench.record_request(server, directory, 'prose-260000-repeat',
                    'Independent cold trial.\n' + fixture['prompt']))
                started = time.monotonic()
                fixture = fixtures['prose-8192']
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    jobs = [pool.submit(bench.record_request, server, directory, f'concurrent-{i}',
                        f'Concurrent independent client {i}.\n' + fixture['prompt']) for i in range(8)]
                    batch = [job.result() for job in jobs]
                elapsed = time.monotonic() - started
                tokens = sum(r['usage']['completion_tokens'] for r in batch)
                bench.save(directory / 'concurrency.json', {'clients': 8, 'seconds': elapsed,
                    'completion_tokens': tokens, 'aggregate_completion_tps': tokens / elapsed})
                # Demonstrate continuation latency separately from the cold numbers.
                warm = bench.request('Warm prefix test.\n' + fixtures['prose-131072']['prompt'], max_tokens=64)
                bench.save(directory / 'warm-seed.json', warm)
                warm2 = bench.request('Warm prefix test.\n' + fixtures['prose-131072']['prompt'], max_tokens=64)
                bench.save(directory / 'warm-repeat.json', warm2)
            if server.monitor_error:
                raise RuntimeError(server.monitor_error)
        results[name] = records
        time.sleep(3)
        return records

    # First establish the same advertised profile using the original checkpoint.
    profile('fast-1024', 1024, 1, 'k8v4', 'mtp', 10, False,
        [f'{kind}-{size}' for size in (8192, 131072, 260000) for kind in ('recall', 'prose')])
    # Match the stock pool, slots, 8-bit KV and vision residency while changing only chunk size.
    chunk_times = {}
    for chunk in (1024, 4096, 8192):
        records = profile(f'stock-like-{chunk}', chunk, 8, 'int8', 'mtp', 3, True,
                          ['prose-131072', 'prose-260000'])
        chunk_times[chunk] = sum(r['timings']['prompt_ms'] for r in records)
    best_chunk = min(chunk_times, key=chunk_times.get)
    bench.save(output / 'chunk-selection.json', {'sum_prompt_ms': chunk_times, 'selected': best_chunk})
    print(f'Selected prefill chunk {best_chunk}: {chunk_times}', flush=True)
    # Keep production-sized capacity while applying the faster KV and speculative profiles.
    cases = [f'{kind}-{size}' for size in (8192, 131072, 260000) for kind in ('recall', 'prose')]
    for backend, drafts in (('mtp', 10), ('dflash2', 7)):
        profile(f'tuned-{backend}', best_chunk, 8, 'k8v4', backend, drafts, True, cases)
    # Use long-context prose decode as the selection metric, not easy recall.
    candidates = ('tuned-mtp', 'tuned-dflash2')
    winner = max(candidates, key=lambda name: next(r['timings']['predicted_per_second']
        for r in results[name] if r['name'] == 'prose-260000'))
    backend, drafts = ('mtp', 10) if winner == 'tuned-mtp' else ('dflash2', 7)
    bench.save(output / 'backend-selection.json', {'selected': winner,
        'criterion': 'single-request prose decode throughput at approximately 260K tokens'})
    # Confirm the winning profile, eight active lanes, thinking and warm-prefix behavior.
    profile('selected-confirmation', best_chunk, 8, 'k8v4', backend, drafts, True,
            ['prose-8192', 'prose-131072', 'prose-260000', 'recall-260000'], extra=True)
