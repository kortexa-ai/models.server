"""Optional resumable download of the pinned public ARM image on a slow link.

Writes a Docker archive only. Does not load an image or touch a running service.
Every registry object is verified against the pinned manifest before packaging.
"""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import tarfile
import threading
import time
import urllib.request
from urllib.parse import urlparse

REPOSITORY = "vllm/vllm-openai"
MANIFEST = "sha256:f853a6833fea509edf1298cad31b630003c242a8b2b70c353e831fa716c8da0b"
IMAGE = "sha256:7d928d3c9b8a4027cf58d3c50c861ed381cf322ab34dc06a2363fc88e08d3ead"
TAG = "local/glm53-main-base:7d928d3c9b8a"
ROOT = Path(__file__).resolve().parents[2] / ".engines/glm53-main/base-image"
LOCK = threading.Lock()
AUTH = {}


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        result = super().redirect_request(request, fp, code, message, headers, url)
        if result and urlparse(request.full_url).netloc != urlparse(url).netloc:
            result.remove_header("Authorization")
        return result


def request(path, **headers):
    with LOCK:
        if time.monotonic() > AUTH.get("expires", 0):
            url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + REPOSITORY + ":pull"
            with urllib.request.urlopen(url, timeout=30) as response:
                body = json.load(response)
            AUTH.update(token=body["token"], expires=time.monotonic() + 240)
        token = AUTH["token"]
    req = urllib.request.Request("https://registry-1.docker.io/v2/" + REPOSITORY + "/" + path,
                                 headers={"Authorization": "Bearer " + token, **headers})
    return urllib.request.build_opener(Redirect()).open(req, timeout=90)


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def part(task):
    blob, start, end = task
    key = blob["digest"].split(":")[1]
    destination = ROOT / f"{key}.{start:012d}.part"
    size = end - start + 1
    if destination.exists() and destination.stat().st_size == size:
        return destination
    temporary = destination.with_suffix(".partial")
    for attempt in range(3):
        try:
            with request("blobs/" + blob["digest"], Range=f"bytes={start}-{end}") as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{blob['size']}":
                    raise ValueError("Registry did not honor the exact requested range")
                with temporary.open("wb") as stream:
                    remaining = size
                    while remaining:
                        block = response.read(min(1024 ** 2, remaining))
                        if not block:
                            raise ValueError("Truncated range")
                        stream.write(block)
                        remaining -= len(block)
            temporary.replace(destination)
            return destination
        except Exception as exc:
            if attempt == 2:
                # Do not print signed CDN URLs or the authorization header.
                raise RuntimeError(f"Range failed: {key[:12]} {start}, {type(exc).__name__}") from None
            time.sleep(2)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with request("manifests/" + MANIFEST, Accept="application/vnd.docker.distribution.manifest.v2+json") as response:
        raw = response.read()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != MANIFEST:
        raise ValueError("Registry manifest digest mismatch")
    manifest = json.loads(raw)
    if manifest["config"]["digest"] != IMAGE:
        raise ValueError("Wrong image config")
    (ROOT / "registry-manifest.json").write_bytes(raw)
    blobs = {blob["digest"]: blob for blob in [manifest["config"], *manifest["layers"]]}
    pending = []
    for blob in blobs.values():
        destination = ROOT / blob["digest"].split(":")[1]
        if destination.exists() and digest(destination) == blob["digest"]:
            continue
        for start in range(0, blob["size"], 32 * 1024 ** 2):
            pending.append((blob, start, min(start + 32 * 1024 ** 2, blob["size"]) - 1))
    started = time.monotonic()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(part, task): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            future.result()
            _, start, end = futures[future]
            completed += end - start + 1
            print(json.dumps({"completed_mib": round(completed / 1024 ** 2),
                              "elapsed_s": round(time.monotonic() - started)}), flush=True)
    for blob in blobs.values():
        key = blob["digest"].split(":")[1]
        destination = ROOT / key
        if destination.exists() and digest(destination) == blob["digest"]:
            continue
        with destination.open("wb") as target:
            for start in range(0, blob["size"], 32 * 1024 ** 2):
                with (ROOT / f"{key}.{start:012d}.part").open("rb") as source:
                    while data := source.read(1024 ** 2):
                        target.write(data)
        if digest(destination) != blob["digest"]:
            raise ValueError("Downloaded blob digest mismatch: " + key)
    archive = ROOT / "base-image.tar"
    docker_manifest = [{"Config": IMAGE.split(":")[1], "RepoTags": [TAG],
                        "Layers": [blob["digest"].split(":")[1] for blob in manifest["layers"]]}]
    with tarfile.open(archive, "w") as output:
        for key in blobs:
            path = ROOT / key.split(":")[1]
            output.add(path, arcname=path.name, recursive=False)
        data = json.dumps(docker_manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        output.addfile(info, io.BytesIO(data))
    print(json.dumps({"archive": str(archive), "image_id": IMAGE, "tag": TAG,
                      "manifest_digest": MANIFEST}), flush=True)


if __name__ == "__main__":
    main()
