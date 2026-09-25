# Model vault on Smarty

This queue archives an explicit file inventory to `~/storage/models/vault`.
It does not start models or use the GPU. Original weights and selected NVFP4
copies share one queue; duplicate selections from the same revision are merged.
Liquid model selections are limited to the approved LFM2.5 family.

The inventory is an operator artifact, not a Git-tracked model catalog. Keep
the source JSON in the vault, then prepare it once. Preparation pins every
repository to a commit and checks every selected path, size and Git object ID
against Hub metadata. It rejects changed files. It does not add new releases.
It requires the full inventory size plus a 512 GiB reserve on the external disk.

```bash
cd ~/src/models.server/vault
uv venv .venv
uv pip install --python .venv/bin/python 'huggingface-hub==1.31.0'
.venv/bin/python run.py prepare --source ~/storage/models/vault/source-inventory.json
ktxsvc install models/vault
```

`ktxsvc install` enables startup at boot. The queue retries failed transfer
workers itself. Snappy, SSH, a
terminal, and an agent session are not needed after installation. The checked-in
service is inert until installed. Only this new service is managed by these
commands. Pause, resume and inspect it on Smarty:

```bash
ktxsvc stop models/vault
ktxsvc start models/vault
python3 ~/src/models.server/vault/run.py status
journalctl -u kortexa-ai-model-vault.service -n 30 --no-pager
```

Stopping retains partial files. To pause across a reboot, also use
`ktxsvc disable models/vault`; resume with `enable` and `start`.

The runner downloads one file at a time, starting with smaller repositories.
Hugging Face downloads up to 50 GB initially use resumable HTTP ranges. A
successful comparison can enable Xet for files of at least 1 MB through
`transport-policy.json`. Larger individual files above 50 GB always require
Xet; these use sequential disk writes, four range requests by default (16 after
a winning default-concurrency comparison), and
no chunk cache. Xet keeps its small transfer state in the vault. Progress uses
allocated file bytes, and the watchdog checks write timestamps as well as size
so sparse preallocation does not look like a stalled download. HTTP requests
have a 60-second read timeout; 15 minutes without file write progress kills and
retries either transport's worker. Verification
has a separate 12-hour limit to allow a busy mechanical disk to read a large
file. Failed repositories yield to the rest of the queue and retry with backoff,
capped at six hours, indefinitely. Missing disk, wrong disk, or low free space
stops writes; the service retries each minute. Read-only `status.json` can be
stale while stopped; check systemd state and its update timestamp together.

The runner checks the WD Elements filesystem UUID before work and throughout
downloads. It keeps at least 512 GiB free and requires enough free space for
each whole file before starting it. Runtime cache and partial files stay in the
vault, separate from serving caches. CPU and disk scheduling have low priority.
A file lock prevents two queue instances. Systemd owns the worker process tree.

Each finished file is verified with the publisher's SHA-256 (or the original
Git blob/LFS pointer identity), flushed to disk, and recorded in an atomically
written receipt. Restarts reuse receipts only when the size and mtime still
match; changed files are verified again. Invalid completed files are retained
with a `.corrupt-*` suffix for investigation and retried. Do not modify the
prepared manifest or move files while this queue is active.

To add selected repositories, first create a pinned additions manifest with the
same `repos` and file metadata structure. Validate its selected files against
Hub metadata. Then stop only this service, append, and resume:

```bash
ktxsvc stop models/vault
cd ~/src/models.server/vault
.venv/bin/python run.py append --source /path/to/pinned-additions.json
ktxsvc start models/vault
```

Append refuses changed versions of existing repositories. It preserves all
receipts and partial downloads, and orders the expanded queue by repository
size. The operation holds the queue lock and commits a durable journal before
changing the manifest/state pair. Startup completes an interrupted committed
append automatically. Backups and the completed journal stay under
`manifest-history/`.

Use `append --defer-repo OWNER/REPO` to retain an exact pinned selection while
excluding it from active downloads and totals. Repeat the option for multiple
repositories; it can be combined with `--source` to record a new selection as
deferred immediately. No model files or verification receipts are deleted.
Deferred inventories remain in `manifest.json` under `deferred_repos`; the
catalog marks their formats as deferred. A later explicit queue update is required to
reactivate them. Deferral and additions share the same recoverable transaction.

Append checks capacity for the remaining queue plus the reserve. Only an explicit
`--allow-capacity-shortfall` queues more than the disk can currently hold. This
does not weaken the runtime free-space guard: more space is required before
the archive can finish. `capacity.json` records the estimate; it conservatively
ignores stored partial bytes. It is a planning snapshot, not live free space.

`catalog.py` writes a compact Markdown table from the manifest and receipts,
with one row per model combining its archive formats. It includes status,
active size, short notes, and exact repository links to pinned revisions.

`benchmark.py --report /path/in/vault/comparison.json` compares four fresh,
similarly sized 4–6 GB shards from one pinned repository in HTTP–Xet–Xet–HTTP
order, using Xet's default 16 parallel range requests and sequential writes.
Stop the managed queue first; the comparison holds its lock and credits
verified files to the normal receipts. It retains all files and partials. Each
trial has a 15-minute limit. Timings include download, checksum verification,
and disk flush. The report checkpoints each trial and records failures without
changing transport policy. A complete comparison enables Xet only if aggregate
throughput improves by at least 20% and both Xet trials beat both HTTP trials
by at least 10%. Run it under a detached, bounded wrapper with an EXIT trap to
`ktxsvc start models/vault`, so the queue resumes even if the caller disconnects.
The normal queue reads the persisted decision before each file.

SAM 3 weights download directly from the public `facebook/sam3` ModelScope mirror
at its pinned commit; the original small metadata files come from Hugging Face.
No Hugging Face token is sent to ModelScope. All mirror files must match the
original HF inventory checksums. Other gated repositories stay deferred until
the existing `hf auth login` account has access. The queue does not accept
license terms or apply for access automatically.

Archive layout:

```text
vault/
  source-inventory.json       # approved original selection
  manifest.json              # pinned revisions, file sizes and checksums
  state.json                 # durable per-file verification receipts and retries
  status.json                # progress, current file and deferred repositories
  current.json / result.json # current worker request and result
  capacity.json              # most recent expansion capacity estimate
  manifest-history/         # append backups and committed change journals
  huggingface/OWNER/REPO/COMMIT/...
```

ModelScope SAM 3 uses the same canonical `huggingface/facebook/sam3/COMMIT`
layout because these are byte-verified files from that original HF revision.
After every file is verified, the service stays idle with phase `complete`.
Logs use the host journal's rotation; they never contain tokens or signed URLs.
