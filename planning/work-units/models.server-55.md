# Shingi recurrent CUDA kernel deployment

Owner: https://github.com/kortexa-ai/models.server/issues/55

Dependency: https://github.com/kortexa-ai/shingi-27b/issues/6

Pin the verified public Shingi revision. Build the runtime in a separate owned directory with CUDA architectures 89 and 120 before production downtime. The SM120 recurrent kernel reuses four columns of input; the Ada path preserves its existing arithmetic. Keep the shared Bonsai runtime separate.

Preserve the old native binary, stamps, package source and runtime. Stop only models/shingi-27b through ktxsvc, select the staged runtime, install the pinned package, and start the same service. The user authorized this targeted downtime. Verify actual positive and negative decisions, prefix reuse, source pins, GPU placement, headroom, and unchanged neighboring services. If a check fails, use the preserved artifacts and the known-good package pin to restore the previous service.
