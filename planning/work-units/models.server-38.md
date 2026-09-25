# Measure HTTP versus Xet on the vault HDD

Work order: https://github.com/kortexa-ai/models.server/issues/38

Compare four fresh, similarly sized pinned shards in HTTP–Xet–Xet–HTTP order.
Use the existing authentication, one file at a time, four Xet range requests,
sequential HDD writes, checksum verification, and disk flush. Count successful
trials as archive progress; retain all partials and receipts. Bound the detached
comparison and resume only the managed vault service on completion or failure.

Switch ordinary weight shards to Xet only after a consistent measured gain.
Persist the decision atomically so completion survives a Snappy disconnect.
Validate selection, decision, failure recovery, and transport safeguards; sync
through Git, run remote tests, and verify the resumed queue uses the selected
transport. Deferred BF16 selections and inference services remain unchanged.
