# MiniCPM-V 4.7 35B-A3B roster entry

Owning issue: https://github.com/kortexa-ai/models.server/issues/65

Register OpenBMB's official GGUF on port 2072 with Q4_K_M, the default
F16 vision projector, q8_0 KV and a 262,144-token pool shared by eight
slots. Preserve the 262,144-token advertised per-request context. Use the
existing isolated llama.cpp setup path, pinned to the upstream commit
that merged MiniCPM-V 4.7 support.

Add both platform service definitions, document setup and the quantization
exception, and extend the required context inventory. Validate artifact
metadata, config parsing, launcher dispatch, runtime admission, shell/plist
syntax and the existing model/runtime tests. Deliver through Git, then
synchronize and validate the roster on Smarty. Keep inference, weight
downloads, service installation and default-model selection outside this
registration task. Record validation and delivery receipts in the issue.
