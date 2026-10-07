"""Pinned single-GB10 EXL3 setup and foreground container ownership."""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def output(args):
    return subprocess.check_output([str(x) for x in args], text=True).strip()


def run(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def engine_root(config):
    return ROOT / '.engines/tensorfold-exl3' / config['id']


def check_platform(config):
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise ValueError('This EXL3 recipe requires Linux ARM64 on GB10.')
    if socket.gethostname().split('.')[0] != config['tensorfold_exl3']['machine']:
        raise ValueError('This model is pinned to Shock; Qwen runs independently on Static.')


def identity(config):
    e = config['tensorfold_exl3']
    return {key: e[key] for key in ['recipe_url', 'recipe_revision', 'runtime_url',
                                   'runtime_revision', 'patches', 'base_image', 'checkpoint', 'checkpoint_revision']}


def checkout(url, revision, target):
    if not target.exists():
        run(['git', 'clone', '--no-checkout', url, target])
        run(['git', '-C', target, 'checkout', '--detach', revision])
    if output(['git', '-C', target, 'remote', 'get-url', 'origin']) != url:
        raise ValueError(f'Unexpected Git origin at {target}')
    if output(['git', '-C', target, 'rev-parse', 'HEAD']) != revision:
        raise ValueError(f'Checkout differs from its pinned revision: {target}')


def dockerfile(config):
    return ('ARG BASE_IMAGE=' + config['tensorfold_exl3']['base_image'] + '\n'
            'FROM ${BASE_IMAGE}\nCOPY TensorFold /opt/tensorfold\n'
            'RUN python3 -m pip install --no-build-isolation --no-deps /opt/tensorfold '
            '&& python3 -c "import tensorfold; assert tensorfold.__version__ == \'0.6.5\'; '
            'import torch,triton,numpy,tokenizers,safetensors,jinja2"\n'
            'ENV TENSORFOLD_NO_UPDATE_CHECK=1 TORCH_EXTENSIONS_DIR=/cache/torch '
            'TRITON_CACHE_DIR=/cache/triton MAX_JOBS=4\nENTRYPOINT ["tensorfold"]\n')


def setup(config):
    check_platform(config)
    e, root = config['tensorfold_exl3'], engine_root(config)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / 'runtime.json'
    if receipt_path.exists():
        verify_runtime(config)
        print(f'Runtime already prepared: {root}')
        return
    recipe, source, checkpoint = root / 'recipe', root / 'TensorFold', root / 'checkpoint'
    checkout(e['recipe_url'], e['recipe_revision'], recipe)
    if output(['git', '-C', recipe, 'status', '--porcelain']):
        raise ValueError('Recipe checkout is dirty.')
    checkout(e['runtime_url'], e['runtime_revision'], source)
    prepared = root / 'sources.json'
    if not prepared.exists():
        if output(['git', '-C', source, 'status', '--porcelain']):
            raise ValueError('Runtime source has unrecorded changes; refusing to overwrite it.')
        for patch in e['patches']:
            path = recipe / patch['path']
            if sha256(path) != patch['sha256']:
                raise ValueError('Recipe patch checksum mismatch.')
            run(['git', '-C', source, 'apply', '--check', path])
            run(['git', '-C', source, 'apply', path])
        write_json(prepared, identity(config))
    if read_json(prepared) != identity(config):
        raise ValueError('Prepared sources differ from the pinned recipe.')
    env = dict(os.environ, HF_HOME=str(root / 'hf-home'), UV_CACHE_DIR=str(root / 'uv-cache'),
               HF_XET_HIGH_PERFORMANCE='1')
    venv = root / '.venv'
    if not venv.exists():
        run(['uv', 'venv', '--python', '/usr/bin/python3', venv], env=env)
    if not (venv / 'bin/hf').is_file():
        run(['uv', 'pip', 'install', '--python', venv / 'bin/python', 'huggingface-hub==2.1.1'], env=env)
    run([venv / 'bin/hf', 'download', e['checkpoint'], '--revision', e['checkpoint_revision'],
         '--local-dir', checkpoint, '--max-workers', '8'], env=env)
    run([venv / 'bin/hf', 'cache', 'verify', e['checkpoint'], '--revision', e['checkpoint_revision'],
         '--local-dir', checkpoint, '--fail-on-missing-files'], env=env)
    metadata_url = f"https://huggingface.co/api/models/{e['checkpoint']}/revision/{e['checkpoint_revision']}?blobs=true"
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    if metadata['sha'] != e['checkpoint_revision']:
        raise ValueError('Hub metadata revision mismatch.')
    files = []
    for item in metadata['siblings']:
        path = checkpoint / item['rfilename']
        if not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError('Checkpoint file missing or wrong size: ' + item['rfilename'])
        files.append({'path': item['rfilename'], 'bytes': path.stat().st_size,
                      'mtime_ns': path.stat().st_mtime_ns, 'sha256': (item.get('lfs') or {}).get('sha256')})
    write_json(root / 'checkpoint.json', {'revision': e['checkpoint_revision'], 'files': files})
    (root / 'Dockerfile').write_text(dockerfile(config))
    (root / '.dockerignore').write_text('**\n!Dockerfile\n!TensorFold\n!TensorFold/**\nTensorFold/.git\n')
    tag = 'local/kortexa-' + config['id'] + ':pinned'
    run(['docker', 'build', '--build-arg', 'BASE_IMAGE=' + e['base_image'], '-t', tag, root])
    image_id = output(['docker', 'image', 'inspect', tag, '--format', '{{.Id}}'])
    write_json(receipt_path, {'identity': identity(config), 'image_id': image_id})
    (root / 'cache').mkdir(exist_ok=True)
    print(f'Prepared {config["id"]}: {image_id}')


def verify_runtime(config):
    root = engine_root(config)
    receipt = read_json(root / 'runtime.json')
    if receipt['identity'] != identity(config):
        raise ValueError('Runtime receipt does not match the model pins.')
    if output(['docker', 'image', 'inspect', receipt['image_id'], '--format', '{{.Id}}']) != receipt['image_id']:
        raise ValueError('Pinned runtime image is missing.')
    checkpoint = read_json(root / 'checkpoint.json')
    if checkpoint['revision'] != config['tensorfold_exl3']['checkpoint_revision']:
        raise ValueError('Checkpoint revision mismatch.')
    for item in checkpoint['files']:
        path = root / 'checkpoint' / item['path']
        if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise ValueError('Checkpoint changed since checksum verification: ' + item['path'])
    return receipt


def container_state(config):
    name = config['tensorfold_exl3']['container_name']
    query = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if query.returncode:
        return None
    data = json.loads(query.stdout)[0]
    if data['Config'].get('Labels', {}).get('kortexa.model') != config['id']:
        raise ValueError('Container name belongs to another workload; refusing to stop it.')
    return data


def stop(config):
    check_platform(config)
    if container_state(config) is not None:
        run(['docker', 'stop', '--timeout', '30', config['tensorfold_exl3']['container_name']])


def serve_args(config, options):
    e = config['tensorfold_exl3']
    return ['serve', '/model', '--backend', 'cuda', '--tp', '1', '--parallel', '1',
            '--host', options.host or config['host'], '--port', str(options.port or config['port']),
            '--name', config['id'], '--context', str(config['context']), '--kv-dtype', 'bf16',
            '--mtp-drafts', str(e['mtp_drafts']), '--drafter', 'none',
            '--thinking' if e['thinking'] else '--no-thinking', '--reasoning-effort', e['reasoning_effort'],
            '--thinking-budget', str(e['thinking_budget']), '--max-tokens', str(e['max_tokens']),
            '--temperature', str(e['temperature']), '--top-p', str(e['top_p']),
            '--min-p', str(e['min_p']), '--no-update-check']


def serve(config, options):
    check_platform(config)
    receipt = verify_runtime(config)
    e, root = config['tensorfold_exl3'], engine_root(config)
    if container_state(config) is not None:
        raise ValueError('This model container already exists; inspect it with ktxsvc.')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', options.port or config['port'])) == 0:
            raise ValueError('The requested port already has a listener.')
    busy = output(['nvidia-smi', '--id=' + e['gpu_uuid'], '--query-compute-apps=pid', '--format=csv,noheader'])
    if busy:
        raise ValueError('Shock GPU has another compute process; inspect ownership before switching.')
    fields = {line.split(':')[0]:int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    if fields['MemAvailable'] < e['startup_free_gib']*GIB:
        raise ValueError(f'This recipe needs {e["startup_free_gib"]} GiB available at startup.')
    args = ['docker', 'run', '--rm', '--name', e['container_name'], '--label', 'kortexa.model=' + config['id'],
            '--user', f'{os.getuid()}:{os.getgid()}', '--gpus', 'device=' + e['gpu_uuid'], '--network', 'host',
            '--memory', str(e['container_memory_gib']*GIB), '--memory-swap', str(e['container_memory_gib']*GIB),
            '-e', 'TENSORFOLD_MEMORY_RESERVE_GIB=' + str(e['reserve_gib']),
            '-e', 'TENSORFOLD_CUDA_MEMORY_LIMIT_GB=' + str(e['container_memory_gib']),
            '-e', 'TF_GLM_CACHE_GIB=' + str(e['prefix_cache_gib']),
            '-v', str(root / 'checkpoint') + ':/model:ro', '-v', str(root / 'cache') + ':/cache',
            receipt['image_id'], *serve_args(config, options)]
    os.execvp(args[0], args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'run', 'stop'])
    parser.add_argument('model')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    path = Path(args.model)
    path = ROOT / path if not path.exists() else path
    config = read_json(path / 'model.json' if path.is_dir() else path)
    try:
        {'setup': setup, 'stop': stop, 'run': lambda c: serve(c, args)}[args.action](config)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
