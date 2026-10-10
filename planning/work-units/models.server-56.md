# GLM NVFP4 TP2 temporary experiment

Owner: https://github.com/kortexa-ai/models.server/issues/56

Run the requested two-Spark GLM-5.3-Flash NVFP4 and DFlash2 recipe in isolated runtime directories. Preserve the existing model files, service definitions, provider configuration, and default model selections. This experiment does not register a permanent model configuration.

Record the Static Qwen TensorFold and Shock GLM EXL3 service baselines before stopping those exact services through `ktxsvc`. Reuse the NVIDIA checkpoint from Smarty's vault. Transfer weights directly from Smarty to Static, then from Static to Shock over their direct link. Synchronize source repositories through Git.

Pin the upstream TP2 recipe and its knapcio dependency, use a separate experimental port and container names, and start with the documented 4 GiB KV budget. Verify both ranks, RoCE transport, health, streamed text, tool-result round trips, and bounded initial throughput. Save detailed evidence under `~/Desktop/ai reports/glm53-tp2-experiment-20261009/`.

If bring-up fails, stop only the experiment containers and restore `qwen-3.8-flash-next-fast` on Static and `glm-5.3-flash-exl3` on Shock through `ktxsvc start`. Verify their original health endpoints. If the experiment succeeds, leave it available for the user's next performance trial and record exact start, stop, and restoration commands in the evidence report.
