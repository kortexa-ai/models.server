# Expand the vault and export its catalog

Work order: https://github.com/kortexa-ai/models.server/issues/36

Verify exact publisher repositories, pinned revisions and file inventories for
the requested Qwen 2.4T, Muse Glimmer, five modified-model entries, Whisper v3,
Kokoro and Bonsai MLX additions. Include matching NVFP4 copies under the standing
archive preference. Investigate the separate GLM-5.3 BF16 release and document
its precision and storage difference from the queued FP8 release.

Append through a recoverable journal while the archive service is stopped.
Preserve existing files, receipts and partial downloads. Keep the disk reserve;
record any explicitly accepted queue capacity shortfall. Test interrupted
manifest/state updates and additive scope validation. Deploy through Git and
verify the expanded service actually downloads files. Export a complete grouped
Markdown inventory of the active queue to the Snappy Desktop.
