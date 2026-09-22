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
    # Match stock sampling; Cinference's non-thinking default presence penalty
    # is 1.5, whereas the production llama.cpp configuration uses zero.
    bench.SAMPLING_OVERRIDES = {'presence_penalty': 0.0, 'frequency_penalty': 0.0,
                              'top_p': 0.95, 'top_k': 20}
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
        'sampling_overrides': bench.SAMPLING_OVERRIDES,
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
        ['prose-8192', 'prose-131072', 'prose-260000', 'recall-260000'])
    # Match the stock pool, slots, 8-bit KV and vision residency while changing only chunk size.
    chunk_times = {}
    for chunk in (1024, 4096, 8192):
        records = profile(f'stock-like-{chunk}', chunk, 8, 'int8', 'mtp', 3, True,
                          ['prose-131072', 'prose-260000'])
        chunk_times[chunk] = sum(r['timings']['prompt_ms'] for r in records)
    # Continue only while larger chunks deliver a meaningful gain and fit well
    # inside the existing 10 GiB card reserve.
    for chunk in (16384, 32768):
        previous = chunk // 2
        earlier = previous // 2
        memory = json.loads((output / f'stock-like-{previous}' / 'memory-summary.json').read_text())
        if chunk_times[previous] > 0.95 * chunk_times[earlier] or memory['peak_process_mib'] > 60 * 1024:
            break
        records = profile(f'stock-like-{chunk}', chunk, 8, 'int8', 'mtp', 3, True,
                          ['prose-131072', 'prose-260000'])
        chunk_times[chunk] = sum(r['timings']['prompt_ms'] for r in records)
    best_chunk = min(chunk_times, key=chunk_times.get)
    bench.save(output / 'chunk-selection.json', {'sum_prompt_ms': chunk_times, 'selected': best_chunk})
    print(f'Selected prefill chunk {best_chunk}: {chunk_times}', flush=True)
    # Keep production-sized capacity while applying the faster KV and speculative profiles.
    cases = ['prose-8192', 'prose-131072', 'prose-260000', 'recall-260000']
    # Isolate the effect of cache precision from draft count and pool size.
    profile('tuned-k8v4-mtp3', best_chunk, 8, 'k8v4', 'mtp', 3, True, cases)
    for backend, drafts in (('mtp', 10), ('dflash2', 7)):
        profile(f'tuned-{backend}', best_chunk, 8, 'k8v4', backend, drafts, True, cases)
    # Franci prioritizes cold prefill. Break near-ties (within 5%) on prose decode.
    candidates = (f'stock-like-{best_chunk}', 'tuned-k8v4-mtp3', 'tuned-mtp', 'tuned-dflash2')
    long_results = {name: next(r for r in results[name] if r['name'] == 'prose-260000')
                    for name in candidates}
    fastest_prefill = min(r['timings']['prompt_ms'] for r in long_results.values())
    eligible = [name for name, r in long_results.items()
                if r['timings']['prompt_ms'] <= 1.05 * fastest_prefill]
    winner = max(eligible, key=lambda name: long_results[name]['timings']['predicted_per_second'])
    if winner.startswith('stock-like-'):
        backend, drafts, kv = 'mtp', 3, 'int8'
    elif winner == 'tuned-k8v4-mtp3':
        backend, drafts, kv = 'mtp', 3, 'k8v4'
    else:
        backend, drafts, kv = ('mtp', 10, 'k8v4') if winner == 'tuned-mtp' else ('dflash2', 7, 'k8v4')
    bench.save(output / 'backend-selection.json', {'selected': winner,
        'criterion': 'minimum 260K cold prefill; fastest prose decode among profiles within 5%'})
    # Confirm the winning profile, eight active lanes, thinking and warm-prefix behavior.
    profile('selected-confirmation', best_chunk, 8, kv, backend, drafts, True,
            ['prose-8192', 'prose-131072', 'prose-260000', 'recall-260000'], extra=True)
