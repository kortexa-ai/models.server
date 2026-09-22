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
    for path in args.run.glob('*-error.json'):
        result['errors'][path.stem] = json.loads(path.read_text())
    for profile in ('stock', 'published', 'stock-like'):
        directory = args.run / profile
        if not directory.exists():
            continue
        data = {'requests': []}
        for name in ('idle', 'memory-summary', 'command', 'environment', 'live-process', 'concurrency', 'canary'):
            path = directory / f'{name}.json'
            if path.exists():
                data[name] = json.loads(path.read_text())
        for path in sorted(directory.glob('*.json')):
            if not path.stem.startswith(('recall-', 'prose-', 'medium-thinking-', 'concurrent-')):
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
