# GLM TP2 screenshot history

Work order: https://github.com/kortexa-ai/models.server/issues/58

The image-count contract covers the complete request, including older tool
screenshots. Accept eight images so OMP's five-image custom-provider budget fits.
Keep the encoder batch, KV cache, processor cache, context and parallelism fixed.

Validate the generated launch arguments and existing lifecycle tests, synchronize
through Git, and rerun explicit setup. Stop Static then Shock through ktxsvc;
start Shock then Static. Verify direct and public multi-image requests and memory
headroom. Preserve the old recipe commit and stopped containers for rollback.
