#!/usr/bin/env python3
"""Run upstream SPEED-Bench with its public dataset pinned to one revision."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    known, remaining = parser.parse_known_args()

    spec = importlib.util.spec_from_file_location("upstream_speed_bench", known.source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load upstream client: {known.source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    original_configs = module.get_dataset_config_names
    original_load = module.load_dataset

    def pinned_configs(path: str, *args: object, **kwargs: object) -> object:
        kwargs["revision"] = known.dataset_revision
        return original_configs(path, *args, **kwargs)

    def pinned_load(path: str, *args: object, **kwargs: object) -> object:
        kwargs["revision"] = known.dataset_revision
        return original_load(path, *args, **kwargs)

    module.get_dataset_config_names = pinned_configs
    module.load_dataset = pinned_load
    print(f"speed_bench: dataset revision={known.dataset_revision}")
    return int(module.main(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
