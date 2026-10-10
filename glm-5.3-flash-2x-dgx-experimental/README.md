# GLM-5.3 Flash: pinned-main Spark experiment

This is the working experiment for Static and Shock. The stable
[`glm-5.3-flash-2x-dgx`](../glm-5.3-flash-2x-dgx/README.md) recipe remains the
recovery path. Promotion is a separate decision after measurement. See
[PLAN.md](PLAN.md) for the experiment and [CACHE.md](CACHE.md) for cache controls.

Static serves port **2071**; Shock is headless. Port **29670** is private
distributed bootstrap on the direct RoCE link. Run only one recipe on the pair.
The weights and their hashes are the same as the stable recipe. No checkpoint
download, conversion, image build or package installation occurs during start.

## Reproduce the image

The Dockerfile pins vLLM source `276fbcff2717bd934cfa37c8a2e4c391f3e7237b` and
native wheel `187a0eb98aa42341d703f83421d693fa7585581b`. The build rejects changes
to native inputs between those commits. The Python HTTP frontend avoids the
intervening Rust frontend change. The immutable ARM64 base manifest and the
preserved transport image are checked separately.

Build on Static while the pair is stopped, with at least 16 GiB free:

```bash
./scripts/setup-vllm-glm53-main.sh --build
```

The build prints its input hash, tag and immutable image ID and saves a receipt
under `.engines/glm53-main/build-receipts/`. Use that exact image ID in the
manifest. Transfer the image directly from Static to Shock over RoCE, then
verify both image IDs. A new build is a new candidate until checked; it must
not silently replace an already validated image pin.

If the registry pull stalls, run the resumable ranged download on Static:

```bash
python3 glm-5.3-flash-2x-dgx-experimental/build/fetch_base.py
docker load --input .engines/glm53-main/base-image/base-image.tar
```

It verifies the manifest, image config and every layer before writing the
Docker archive. Loading that archive creates the local base tag accepted by
the build; it does not start a container.

The mixed-weight adapter preserves the existing FP8 dense weights. The loader
checks the mixed checkpoint manifest and rejects unexpected filter counts.
The draft metadata-builder adapter supplies the draft's own dimensions and
cache precision. Cache and worker observers record allocations without changing
them. The old model, scheduler, allocator and sampling overlays are not loaded.
The preserved RoCE collective is a separate optional arm, controlled by
`GLM_ROCE_ALLREDUCE`; it is disabled for the first upstream baseline.
The byte-preserving `glm_fast_load.py` from the same pinned Knapcio revision
is also opt-in (`GLM_FAST_LOAD=1`). It stages checkpoint tensors in bounded
anonymous/pinned memory to avoid slow file-backed host copies on GB10.
`GLM_FAST_LOAD_VERIFY` checks sampled tensors against the original reader.
Keep loader selection identical between memory comparison arms.

## Managed lifecycle

Inspect GPU workloads, service ownership and ports first. Stop the active API
rank before its worker. Disable the inactive recipe at boot through `ktxsvc`.
After the image and manifest match, run explicit setup on both hosts:

```bash
./scripts/setup-vllm-glm53-main.sh glm-5.3-flash-2x-dgx-experimental
./run.sh glm-5.3-flash-2x-dgx-experimental --dry-run
```

Setup reuses the stable recipe's still-valid artifact verification and rejects
changed files. It writes a separate draft `config.json` under the experimental
receipt directory and mounts it read-only over the checkpoint metadata. This
adds quantization exclusions for the ten BF16 DFlash2 convolution projections,
including their global layer aliases. The original FP8 conversion did not need
those exclusions because the old implementation kept these projections BF16
unconditionally. Shared checkpoint files are never edited. Start verifies the
derived metadata before creating a container.

Install the worker, then API. `ktxsvc install` enables and starts
the unit. During experiments, disable automatic crash restart so a failed rank
stays available for inspection:

```bash
ssh shock 'ktxsvc install models/glm-5.3-flash-2x-dgx-experimental'
ssh shock 'ktxsvc keep-alive off models/glm-5.3-flash-2x-dgx-experimental'
ssh static 'ktxsvc install models/glm-5.3-flash-2x-dgx-experimental'
ssh static 'ktxsvc keep-alive off models/glm-5.3-flash-2x-dgx-experimental'
curl -f http://192.168.2.101:2071/health
curl -f http://192.168.2.101:2071/v1/models
```

A manifest or launcher change requires setup again on both nodes. A stopped
container remains inspectable; the new fingerprint creates a separate owned
container. Use `ktxsvc stop` on Static, then Shock, before changing a candidate.
Start Shock before Static. Never start a duplicate manual server.

## Measure and recover

Use the shared [benchmark programs](../bench/glm53-tp2/README.md). Retain each
tested manifest and raw result outside Git. Record image IDs, startup allocation
logs, both ranks' host memory, cache group shapes, token counts, draft acceptance
and active request counts. A fixed cache pool can increase capacity without
freeing physical memory; reduce its byte budget in a separate test to measure
actual memory released. Forced-length synthetic decoding is not coding quality.

For rollback, stop this recipe on Static and Shock and disable both units.
Re-enable the stable units, start Shock, then Static, and check port 2070 with
a real completion. Preserve the stable image, weights, source checkouts, setup
receipts and containers. The experiment does not change client defaults.

Changing execution state and test evidence belong in
[models.server #60](https://github.com/kortexa-ai/models.server/issues/60).
