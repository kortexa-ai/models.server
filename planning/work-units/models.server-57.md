# GLM two-Spark recipe and client selection

Issue: https://github.com/kortexa-ai/models.server/issues/57

Register `glm-5.3-flash-2x-dgx` on model port 2070. Preserve the verified
NVFP4/DFlash2 TP2 experiment's pins and memory limits. Explicit setup verifies
local artifacts and obtains source through Git. Runtime starts one local rank;
the same managed service is installed on Static and Shock. Static owns the API.
Uninstall the replaced Static Qwen and Shock EXL3 services through `ktxsvc`.

Add the public model and `kortexa-dual` direct provider to installed Hermes,
OMP and pi configurations. Select GPT-6.1 Sol for both Snappy Hermes profiles,
native Windows Scrappy Hermes and all three pi installations. Select the new
GLM for Snappy OMP and GPT-6 Luna for Smarty/Scrappy OMP. Preserve auxiliary
roles, per-machine credentials, and unrelated configuration.

Validate recipe invariants, rank commands, lifecycle ownership and configuration
preservation. Deploy reviewed commits through Git to both Sparks and Smarty;
check managed service ownership, direct/public streaming and tools. Replay the
recorded OMP pre-corruption conversation with a bounded continuation and no
execution of generated commands. The original system prompt and exact tool
schemas were not recorded, so label this reconstruction explicitly.

Rollback stops both new services before starting the preserved experiment
containers (Shock worker, then Static head). Old service recipes and weights
remain available for an explicit later rollback. Private configuration backups
remain on each owning machine. Runtime results and delivery evidence belong in
the issue, not this execution contract.
