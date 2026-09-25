# Defer large BF16 selections to fit the vault

Work order: https://github.com/kortexa-ai/models.server/issues/37

Record the official pinned GLM-5.3 BF16 selection and defer it together with
Qwen 2.4T BF16, as requested. Keep the existing FP8/NVFP4 selections active.
Preserve inventories, downloaded files, partials and verification receipts.
Use one recoverable manifest transaction; exclude deferred selections from
active progress and capacity totals. Keep the 512 GiB disk reserve.

Validate recovery at each transaction stage, explicit deferral scope, receipt
preservation, and active capacity accounting. Deploy only the vault service
through Git and ktxsvc. Verify the active queue and resumption, then refresh the
Desktop catalog as a plain table: one row per model, status first (checkmark or
clock), exact repositories, active size, and brief format/deferred notes.
