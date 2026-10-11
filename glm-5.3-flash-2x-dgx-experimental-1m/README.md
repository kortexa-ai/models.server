# GLM-5.3 Flash: 1M Spark experiment

This separate recipe tests the model's native 1048576-token context on Static
and Shock. It preserves both the production recipe and the 512K DCP experiment.
Their exact recovery points are in [BASELINES.md](BASELINES.md).

Static serves port **2070**; Shock is headless. Private bootstrap uses port
**29671** over the direct RoCE link. All three paired recipes expose API model
`glm-5.3-flash-2x-dgx` at `http://192.168.2.101:2070/v1`. Keep distinct recipe
IDs for service/container/receipt ownership. Run only one recipe on the pair.

The runtime, image, weights, precision and DFlash settings match the 512K DCP
experiment. This recipe raises the request ceiling to 1048576 and the FP8 cache
budget to 6 GiB per rank. It retains TP2/DCP2, eight scheduler slots, interleave
4, the 4672-token batch budget, DFlash2-7, decode graphs and 1 MiB NCCL buffers.
Eight scheduler slots are not eight independently reserved 1M caches.

Use the existing pinned image; do not rebuild or alter shared weights. Run
explicit setup on each host before managed installation:

```bash
./scripts/setup-vllm-glm53-main.sh glm-5.3-flash-2x-dgx-experimental-1m
./run.sh glm-5.3-flash-2x-dgx-experimental-1m --dry-run
```

After stopping and disabling the active recipe, install the worker before the
API rank with `ktxsvc install models/glm-5.3-flash-2x-dgx-experimental-1m`.
The repository unit is inert (`Restart=no`); `ktxsvc install` installs an enabled
copy with crash restart. This is the selected default recipe on both Sparks.
Keep both 512K services disabled. Check `/health`, `/v1/models` and an actual
streamed response on port 2070 using model `glm-5.3-flash-2x-dgx`.
For recovery, stop the API before the worker and follow [BASELINES.md](BASELINES.md).

The [test plan](PLAN.md) separates cold prefill, prefix reuse and decode speed.
Allocator admission is not proof that temporary prefill memory fits. Live
measurements and operational state belong in [issue #61](https://github.com/kortexa-ai/models.server/issues/61).
Selection of this recipe does not change harness default models. Public and
`kortexa-dual` clients retain the same GLM model ID and endpoint across recipe
switches. The server catalog reports the active recipe's context ceiling;
clients with a fixed 512K limit retain that conservative limit when switching
between 512K and 1M recipes.
