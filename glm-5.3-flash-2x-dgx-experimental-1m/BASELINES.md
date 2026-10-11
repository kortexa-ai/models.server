# Preserved 512K recipes

Both complete recipes and their launchers are preserved at Git commit
`81d66be78fa990d582458d1f823475df1b7a40d4`. The 1M experiment must not edit
their manifests, launcher files, shared weights or setup receipts. Keep their
Docker images and stopped containers on both Sparks.

| Setting | Production baseline | Memory-saving experiment |
|---|---|---|
| Recipe | `glm-5.3-flash-2x-dgx` | `glm-5.3-flash-2x-dgx-experimental` |
| API port | 2070 | 2071 |
| Request ceiling | 524288 | 524288 |
| Scheduler slots | 8 | 8 |
| Tensor / decode context parallelism | TP2 / DCP1 | TP2 / DCP2 |
| Cache budget per Spark | 6 GiB | 4 GiB |
| Batch budget | 6919 | 4672 |
| Draft | DFlash2, up to 7 tokens | DFlash2, up to 7 tokens |
| Target / draft attention cache | FP8 | FP8 |
| NCCL buffer override | Default | 1 MiB |

Production image:
`sha256:881716c0f7d0904d3aa2c4f138a15e916c4c1aaca48070b39785032ed485863a`.
Production runtime fingerprint:
`fd2b8a5d35b559a70dfb7319a37180ae5ba8feb5f2585ebb92bb3060f00b8ebe`.
Production manifest SHA-256:
`522fb4f2cefcfca0fdeed2bcef00b868b322a14180d3d47d82bd31752ba2fee6`.

Experimental image:
`sha256:0b8f9a4093328153bc90dbdd79833b98dca59c106de79aec379440d2985550e7`.
Experimental runtime fingerprint:
`e2de66f576859de50ae8a0ddff6f42c86b1957b0de7676c134439572d2b85603`.
Experimental manifest SHA-256:
`efd5701f634d69bf58fb38c60d4a5c5348dc2721c42526e988e6fec301f7e337`.
It pins vLLM source `276fbcff2717bd934cfa37c8a2e4c391f3e7237b` and native wheel
`187a0eb98aa42341d703f83421d693fa7585581b`, with the tested local SM120 DCP
adapter, interleave 4, Triton draft attention and decode CUDA graphs.

Launcher SHA-256 values:

- `scripts/glm53_tp2_runtime.py`:
  `ea69dfb269b715d8f5cecd52de7d669ebb614b374ec85249edb9f86a56510d95`.
- `scripts/glm53_main_runtime.py`:
  `dc22b3da3ce2a455b78870a50a1be0f547ad1f283088d4b64c6454c40c0c8ac9`.

Each manifest pins the complete target and draft artifact inventories.
The earlier comparison and successful live rollback in both directions are
recorded in [models.server #60](https://github.com/kortexa-ai/models.server/issues/60).

## Return from the 1M experiment

First check that no client request is active. Stop the API before its worker:

```bash
ssh static 'ktxsvc stop models/glm-5.3-flash-2x-dgx-experimental-1m'
ssh shock 'ktxsvc stop models/glm-5.3-flash-2x-dgx-experimental-1m'
ssh static 'ktxsvc disable models/glm-5.3-flash-2x-dgx-experimental-1m'
ssh shock 'ktxsvc disable models/glm-5.3-flash-2x-dgx-experimental-1m'
```

For the memory-saving 512K recipe, enable both units, then start worker before
API. These use the original unmodified setup receipts:

```bash
ssh shock 'ktxsvc enable models/glm-5.3-flash-2x-dgx-experimental'
ssh static 'ktxsvc enable models/glm-5.3-flash-2x-dgx-experimental'
ssh shock 'ktxsvc start models/glm-5.3-flash-2x-dgx-experimental'
ssh static 'ktxsvc start models/glm-5.3-flash-2x-dgx-experimental'
curl -f http://192.168.2.101:2071/health
curl -f http://192.168.2.101:2071/v1/models
```

For the production baseline, substitute `models/glm-5.3-flash-2x-dgx` and
port 2070, and keep both experimental recipes disabled. Always verify a real
streamed completion, both service PIDs and exactly one model container per
Spark. Never start a second recipe while another occupies the pair.
