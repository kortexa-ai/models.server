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

`ktxsvc install` enables startup at boot and process restart. Snappy, SSH, a
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
Hugging Face downloads use resumable HTTP ranges with Xet disabled, following
the working transport on Smarty. Each file has a 60-second network read timeout;
15 minutes without byte progress kills and retries the worker. Verification
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
  huggingface/OWNER/REPO/COMMIT/...
```

ModelScope SAM 3 uses the same canonical `huggingface/facebook/sam3/COMMIT`
layout because these are byte-verified files from that original HF revision.
After every file is verified, the service stays idle with phase `complete`.
Logs use the host journal's rotation; they never contain tokens or signed URLs.
