#!/usr/bin/env python3
"""Serve the pinned shingi-27b package with this host's model.json memory floors.

The public server requires 14 GiB free before load and 4 GiB of headroom on
cards up to 32 GiB. The shared RTX 4090 cannot meet that next to TTS and ASR,
so this launcher replaces only those two floors. The package's own UUID and
20 GiB minimum-size checks still run. Weights resolve from the standard
Hugging Face cache at a pinned revision; the server still verifies their
pinned SHA-256. An optional vision projector resolves the same way and is
passed as --mmproj; without --projector-file the server runs text-only (--no-vision).

Usage: shingi-server.py --preload-mib N --headroom-mib N --weights-repo REPO
       --weights-revision SHA --model-file F --calibration-file F
       [--projector-file F] --executable READOUT [--host H] [--port P]
       [server args...]
"""
import argparse
import sys


def override_floors(preload_mib, headroom_mib):
    """Keep the package's GPU identity and size checks, but use our floors."""
    from shingi import backend, gpu

    original = gpu.gpu_profile

    def gpu_profile():
        uuid, _, _ = original()
        print(f"Using models.server floors: {preload_mib} MiB free before load, "
              f"{headroom_mib} MiB headroom", flush=True)
        return uuid, preload_mib, headroom_mib

    gpu.gpu_profile = gpu_profile
    # backend imported the name directly, so patch its reference too.
    backend.gpu_profile = gpu_profile


def resolve(repo, revision, filename):
    """Return a cached file path, downloading only when it is not cached yet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    try:
        return hf_hub_download(repo, filename, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(repo, filename, revision=revision)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preload-mib", type=int, required=True)
    parser.add_argument("--headroom-mib", type=int, required=True)
    parser.add_argument("--weights-repo", required=True)
    parser.add_argument("--weights-revision", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--calibration-file", required=True)
    parser.add_argument("--projector-file")
    args, server_args = parser.parse_known_args(argv)

    override_floors(args.preload_mib, args.headroom_mib)
    model = resolve(args.weights_repo, args.weights_revision, args.model_file)
    calibration = resolve(args.weights_repo, args.weights_revision, args.calibration_file)
    projector_args = ["--no-vision"]
    if args.projector_file:
        projector = resolve(args.weights_repo, args.weights_revision, args.projector_file)
        projector_args = ["--mmproj", str(projector)]

    from shingi import server

    sys.argv = ["shingi-27b", "--model", str(model), "--calibration", str(calibration),
                *projector_args, *server_args]
    server.main()


if __name__ == "__main__":
    main()
