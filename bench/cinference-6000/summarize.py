#!/usr/bin/env python3
"""Reduce the local benchmark records to a small, publishable evidence file."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = {'manifest': json.loads((args.run / 'manifest.json').read_text()),
              'profiles': {}, 'errors': {}}
    for name in ('stock-model', 'chunk-selection', 'backend-selection', 'selection'):
        path = args.run / f'{name}.json'
        if path.exists():
            result[name] = json.loads(path.read_text())
    for path in args.run.glob('*-error.json'):
        result['errors'][path.stem] = json.loads(path.read_text())
    for profile in sorted(p.name for p in args.run.iterdir()
                          if p.is_dir() and (p / 'command.json').exists()):
        directory = args.run / profile
        if not directory.exists():
            continue
        data = {'requests': []}
        for name in ('idle', 'memory-summary', 'command', 'environment', 'live-process',
                     'concurrency', 'canary', 'props', 'warm-seed', 'warm-repeat',
                     'models', 'vision', 'tool', 'tool-followup'):
            path = directory / f'{name}.json'
            if path.exists():
                data[name] = json.loads(path.read_text())
        log = directory / 'requests.jsonl'
        if log.exists():
            for line in log.open():
                event = json.loads(line)
                if event.get('event') == 'server_start':
                    data['server_start'] = event
                    break
        digest = directory / 'executable.sha256' if profile == 'stock' else args.run / 'cinference-executable.sha256'
        if not digest.exists() and profile != 'stock':
            digest = args.run.parent / 'cinference-executable.sha256'
        if digest.exists():
            data['executable_sha256'] = digest.read_text().split()[0]
        for path in sorted(directory.glob('*.json')):
            if not path.stem.startswith(('recall-', 'prose-', 'medium-thinking-', 'concurrent-', 'coding-', 'voice-')):
                continue
            record = json.loads(path.read_text())
            for field in ('content', 'reasoning'):
                text = record.pop(field, '')
                record[field + '_sha256'] = hashlib.sha256(text.encode()).hexdigest()
                record[field + '_characters'] = len(text)
            data['requests'].append(record)
        result['profiles'][profile] = data
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for profile, data in result['profiles'].items():
        print(profile, data.get('memory-summary'))
        for request in data['requests']:
            timing = request.get('timings', {})
            print(request.get('name'), request.get('usage', {}).get('prompt_tokens'),
                  round(timing.get('prompt_per_second', 0), 1),
                  round(timing.get('predicted_per_second', 0), 1),
                  round(request.get('client_ttft_seconds') or 0, 2),
                  request.get('peak_process_mib'), request.get('recall_found'))


if __name__ == '__main__':
    main()
