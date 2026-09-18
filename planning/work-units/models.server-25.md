# Bonsai shared KV capacity and OMP defaults

Work unit: https://github.com/kortexa-ai/models.server/issues/25

Set the llama profile to 393216 total shared KV tokens and eight request
slots. Preserve native model context metadata, quantization, and GPU binding.
Update the existing launch checks and document the shared capacity.

Deploy through Git, validate on smarty, and restart only Bonsai with
`ktxsvc`. Verify eight concurrent requests, the effective context allocation,
and GPU memory. If runtime validation fails, revert this profile change
through Git and restart the previous verified configuration.

Add a separate Bonsai custom provider to OMP on snappy and smarty, pointing
to smarty's port 2062. Select it as the default after server validation,
preserve other provider and role settings, and verify an OMP request on each
host. Keep execution and deployment evidence in the linked issue.
