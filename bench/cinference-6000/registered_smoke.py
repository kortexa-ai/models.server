#!/usr/bin/env python3
"""Exercise the registered launchers without installing managed services."""
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import zlib

import run as bench


def red_png():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = (b'\x00' + b'\xff\x00\x00' * 64) * 64
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 64, 64, 8, 2, 0, 0, 0))
    return 'data:image/png;base64,' + base64.b64encode(data + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b'')).decode()


def main():
    destination = bench.ROOT / 'bench-results/cinference-27/registered'
    destination.mkdir(exist_ok=False)
    env = os.environ.copy()
    bench.SAMPLING_OVERRIDES = {'presence_penalty': 0.0, 'frequency_penalty': 0.0, 'top_p': 0.95, 'top_k': 20}
    bench.save(destination / 'manifest.json', {
        'models_commit': bench.command('git', '-C', str(bench.ROOT), 'rev-parse', 'HEAD'),
        'runtime_commit': bench.command('git', '-C', str(bench.RUNTIME / 'runtime/ninfer'), 'rev-parse', 'HEAD'),
        'gpu_uuid': bench.GPU, 'restore_services': env.get('SMARTY_GPU_RECORDED_SERVICES'),
        'started_unix': time.time(), 'configurations': {
            name: json.loads((bench.ROOT / name / 'model.json').read_text())
            for name in ('qwen-3.8-27b-fast', 'qwen-3.8-27b-fast-abliterated')}})
    for name, port in (('qwen-3.8-27b-fast', 2064), ('qwen-3.8-27b-fast-abliterated', 2065)):
        bench.MODEL, bench.PORT = name, port
        bench.URL = f'http://127.0.0.1:{port}'
        directory = destination / name
        argv = [str(bench.ROOT / 'run.sh'), name, '--host', '127.0.0.1',
                '--request-log-jsonl', str(directory / 'requests.jsonl')]
        with bench.Server(argv, directory, env) as server:
            canary = bench.request('What is 7 plus 5? Answer with only the number.', max_tokens=32)
            bench.save(directory / 'canary.json', canary)
            assert canary['content'].strip() == '12', canary
            models = bench.api('/v1/models')
            bench.save(directory / 'models.json', models)
            assert models['data'][0]['id'] == name, models
            body = {'model': name, 'temperature': 0, 'max_tokens': 128, 'reasoning_effort': 'none',
                    'messages': [{'role': 'user', 'content': [
                        {'type': 'text', 'text': 'What color is the image? Reply with one color word.'},
                        {'type': 'image_url', 'image_url': {'url': red_png()}}]}]}
            vision = bench.api('/v1/chat/completions', body)
            bench.save(directory / 'vision.json', vision)
            assert 'red' in vision['choices'][0]['message']['content'].lower(), vision
            body['messages'] = [{'role': 'user', 'content': 'Use the get_weather tool to get the weather in Seattle.'}]
            body['tools'] = [{'type': 'function', 'function': {'name': 'get_weather',
                'description': 'Get current weather for a city.', 'parameters': {
                    'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]
            tool = bench.api('/v1/chat/completions', body)
            bench.save(directory / 'tool.json', tool)
            calls = tool['choices'][0]['message'].get('tool_calls', [])
            assert calls and calls[0]['function']['name'] == 'get_weather', tool
            assert 'Seattle' in json.loads(calls[0]['function']['arguments'])['city'], tool
            body['messages'].extend([tool['choices'][0]['message'],
                {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '{"temperature_c": 18, "condition": "sunny"}'}])
            followup = bench.api('/v1/chat/completions', body)
            bench.save(directory / 'tool-followup.json', followup)
            assert '18' in followup['choices'][0]['message']['content'], followup
            fixture = json.loads((bench.ROOT / 'bench-results/cinference-27/prompts/recall-131072.json').read_text())
            recalled = bench.record_request(server, directory, 'recall-131072', fixture['prompt'], fixture['markers'])
            assert recalled['recall_found'] == len(fixture['markers']), recalled
            code = json.loads((bench.ROOT / 'bench-results/cinference-27/workloads/prompts/coding-131072.json').read_text())
            bench.record_request(server, directory, 'coding-nothink-131072',
                                 'Registered launcher direct code trial.\n' + code['prompt'], max_tokens=1024)
            if server.monitor_error:
                raise RuntimeError(server.monitor_error)
            print(f'PASS registered launcher: {name}; arithmetic, alias, image, tool round-trip, recall and code', flush=True)
        time.sleep(3)


if __name__ == '__main__':
    main()
