# Liquid d1-3B roster support

Owning issue: https://github.com/kortexa-ai/models.server/issues/63

Register the official open d1-3B decision checkpoint at port 2071 using
Q8_0 weights and projector, q8_0 KV, a 32,768-token context and one slot.
Use the existing isolated llama.cpp setup and launch path with an upstream
revision that supports LFM2 d1 and `/v1/systemone` on Metal and CUDA.

Add both managed-service definitions and document text and image requests.
Validate roster coverage, launch dispatch, service syntax and real decision
responses on Snappy. Deliver through Git and validate the registration on
Smarty. Preserve the current defaults and the running Shingi service.
Keep execution and validation evidence in the owning issue.
