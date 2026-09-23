# Oversized vault transfers

Work order: https://github.com/kortexa-ai/models.server/issues/35

The pinned Hub client requires Xet for individual files larger than 50 GB.
Select that transport before importing the client in the file worker. Retain
the existing HTTP path for smaller files. Use sequential HDD reconstruction
and bounded concurrency, with cache state on the vault disk.

Watch size and write timestamps to support preallocated partial files. Test
transport configuration and sparse file progress, preserve existing receipts
and partial files, deploy through Git, then restart only the archive queue and
verify actual writes to the previously deferred Bonsai F16 file.
