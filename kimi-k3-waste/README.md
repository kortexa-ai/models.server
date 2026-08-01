# Kimi K3 with WASTE on Smarty

This directory preserves the work from the July 2026 Kimi K3/WASTE
experiment. It contains documentation, the two Smarty-specific operational
scripts, and the experimental patch needed on top of WASTE v0.6.0.

It intentionally contains no weights, converted expert banks, caches, logs,
or credentials. The large artifacts and run history remain on Smarty.

## Current status

Paused because Smarty demonstrated repeatable volatile-memory/kernel
corruption under a large buffered read workload. Do not resume the conversion
until the DDR4/CPU memory-controller/platform has passed an offline memory
test. Avoid another buffered `hf cache verify` over the 1.42 TiB checkpoint;
that workload hard-locked the machine twice.

The experiment did not reach a runnable WASTE container:

- all 96 Kimi K3 source shards are present;
- three known-corrupt source shards were redownloaded and atomically replaced;
- expert-bank and per-layer codebook files for layers 1-3 were published;
- layers 1-3 are **provisional**, because they were produced before the memory
  fault was understood and have not passed a complete container round trip;
- attempt 5 was prepared but never launched;
- no trunk, merged codebook, or manifest was published, so the output
  directory is not currently an openable container.

## Pinned inputs and machine

| Item | Value |
|---|---|
| Hugging Face repository | `moonshotai/Kimi-K3` |
| Hub revision | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` |
| Source index SHA-256 | `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd` |
| Source shard count | 96 |
| Source shard bytes | `1,560,936,091,448` (about 1.42 TiB) |
| WASTE repository | `https://github.com/sqliteai/waste.git` |
| WASTE base | tag `v0.6.0`, commit `ce96e38e573cb1befd45623d0213027d09dce8a5` |
| Host | Smarty, Intel Core i9-14900K, 128 GB non-ECC DDR4 |
| GPU | RTX PRO 6000 Blackwell, 97,887 MiB VRAM |
| Last tested kernel | `7.0.0-28-generic` |
| NVIDIA driver | `595.84` |

WASTE inference remains CPU-side. The GPU was used only for the PyTorch VQ
portion of conversion. With Smarty's ordinary production GPU services up,
about 27 GiB VRAM was free; the guarded launcher requires at least 16 GiB at
start and aborts below 8 GiB.

## Smarty paths

| Purpose | Path |
|---|---|
| Kimi K3 source checkpoint | `/mnt/data/k3` |
| Partial WASTE output | `/home/francip/models/k3.waste` |
| Logs and status files | `/home/francip/models/k3.waste.runs` |
| WASTE checkout | `/home/francip/src/waste` |
| Repair download directory | `/mnt/data/k3-repair.160e5P` |

The two scripts in this directory preserve the contents from
`/home/francip/models/k3.waste.runs`. They contain these absolute paths and
are Linux/Smarty-specific.

The last known partial output is:

```text
codebooks-L1.bin       37,008 bytes
codebooks-L2.bin       37,008 bytes
codebooks-L3.bin       37,008 bytes
experts-L1.bin     11,116,478,464 bytes
experts-L2.bin     11,116,478,464 bytes
experts-L3.bin     11,116,478,464 bytes
```

## What was tried

### Download and authentication

The Hugging Face CLI was authenticated as `francip`. Default Xet repair
downloads reached roughly 11 MB/s. Restarting the exact download with
`HF_XET_HIGH_PERFORMANCE=1` and three workers reached roughly 34 MB/s and
downloaded the three replacement shards in 22 minutes 40 seconds.

The repair command was revision-pinned and conceptually equivalent to:

```bash
taskset -c 16-31 env HF_XET_HIGH_PERFORMANCE=1 \
  hf download moonshotai/Kimi-K3 \
  model-00006-of-000096.safetensors \
  model-00070-of-000096.safetensors \
  model-00092-of-000096.safetensors \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --local-dir /mnt/data/k3-repair.160e5P \
  --max-workers 3
```

Pass `HF_TOKEN` through the environment if the saved CLI login is not
available. Never put the token in this repository or in a command-line
argument that will be logged.

### Conversion attempts

1. CPU, native VQ, three jobs: failed with an impossible MXFP4 LUT index
   (`137438953480` for a 16-entry table).
2. CPU, native VQ, three jobs: failed again with another impossible index.
3. CUDA, native VQ disabled, one job: failed with the same class of impossible
   LUT index, exonerating the optional native VQ implementation.
4. CUDA PyTorch VQ, native disabled, one job, arithmetic E2M1 decoder:
   published layers 1-3 in 330, 315, and 313 seconds. It failed before layer 4
   because `model.safetensors.index.json` had acquired seven one-bit changes
   and could no longer be decoded as UTF-8.
5. A guarded resume was written but not launched.

The arithmetic E2M1 decoder in the preserved patch was exhaustively checked
against the original lookup-table decoder for all 256 possible packed byte
values, including signed zero. The outputs were bit-identical. A real-expert
self-test also passed. This patch avoids an unsafe advanced-index operation;
it does not repair faulty hardware.

### Source verification and repair

The corrupt index was preserved as:

```text
/mnt/data/k3/model.safetensors.index.json.corrupt-20260731-attempt4
```

A fresh revision-pinned index was installed with the SHA-256 listed above.
A complete Hub verification then found exactly three bad model shards among
the 96. Optional evaluation YAML files and `assets/kimi-logo.png` were absent
locally but are not conversion inputs.

| Shard | Expected/current SHA-256 | Old corrupt SHA-256 |
|---|---|---|
| 6 | `b1d4805767471a9721cd087d2047843ab9262d4f9bbe0d4a306e72c07179f939` | `e3c8c63c099e9f54db6ec1461b98bc64d9f011528af3b58131cfc35133c79701` |
| 70 | `ab53464148721a209d4941dc3ab6f698574f913179e729726d00ed6232144893` | `11efbe60bfdb25689c58f83556b4f2e1cfde2b38ee8558249e71e907379e7acd` |
| 92 | `359848294be55e1c8949f0a4c92098bc73276895289fd1123f10769ba685e248` | `57381e0cdd123eb01526d3950a539d0c9de3ee55d1dfee776465f03279491098` |

The replacement files matched the expected hashes through `O_DIRECT` reads.
The originals matched their old corrupt hashes through `O_DIRECT`, proving
that the old disk files really were bad. The replacements were installed with
same-filesystem atomic renames. The old files remain as:

```text
/mnt/data/k3/model-00006-of-000096.safetensors.corrupt-20260731-pre-repair
/mnt/data/k3/model-00070-of-000096.safetensors.corrupt-20260731-pre-repair
/mnt/data/k3/model-00092-of-000096.safetensors.corrupt-20260731-pre-repair
```

Together those backups consume about 50 GB. The repair directory retains
about 4.2 GB of resumable Xet fragments. Delete them only after the machine is
stable and a direct-I/O verification has passed. They are redownloadable, but
they are intentionally not touched by anything in this repository.

### Hardware/kernel failure

The decisive observation was that disk and memory disagreed:

- `O_DIRECT` hashing repeatedly returned the correct Hub hash;
- after evicting clean cache pages, the first ordinary buffered hash was
  correct;
- eight seconds later, the same clean cached file returned a different hash;
- subsequent reads returned additional different hashes;
- byte comparison showed single-bit changes in clean cached pages.

The first full verifier on kernel 6.17 produced repeatable Oopses in
`filemap_get_read_batch`. After upgrading to kernel 7.0 and NVIDIA 595.84, a
second full buffered verifier reached about 1.288 TB (82.5%) before the host
hard-locked.

The previous-boot kernel journal showed the actual failure sequence:

1. `kswapd0` reclaimed the large file cache.
2. The kernel detected an LRU-list pointer whose value changed from
   `ffff895a49423490` to `ffff895e49423490` -- a one-bit change.
3. A second list corruption caused a NULL-pointer dereference in the memory
   reclaim path.
4. CPU 16, running `hf`, hard-locked; RCU, TLB flush, and memory compaction
   then stalled other CPUs until the machine stopped servicing userspace.

This was not GPU OOM or host OOM:

- the verifier did not use CUDA;
- the GPU retained about 27 GiB free;
- the failed boot recorded zero NVIDIA Xids;
- the failed boot recorded zero OOM kills;
- low `free` memory was reclaimable file cache, while `available` memory was
  still about 102 GiB.

Both NVMes reported zero media errors and zero NVMe error-log entries. K3 is
on `nvme1`, behind PCIe root port `00:1a.0`. Repeated correctable PCIe Rx
warnings came from `00:06.0`, which serves the other drive (`nvme0`). Those
warnings deserve separate maintenance, but they do not explain the stable
direct hashes and mutating RAM copies from `nvme1`.

The leading diagnosis is faulty/unstable DDR4 or the i9's integrated memory
controller/platform. A kernel memory-management bug remains possible, and
rogue DMA cannot be absolutely excluded. Non-ECC DDR4 need not produce an MCE
when it flips a bit.

## WASTE patch

The checkout on Smarty is intentionally dirty at WASTE v0.6.0. To reconstruct
it from a fresh clone:

```bash
git clone https://github.com/sqliteai/waste.git
cd waste
git checkout ce96e38e573cb1befd45623d0213027d09dce8a5
git apply --unidiff-zero \
  /path/to/models.server/kimi-k3-waste/patches/waste-v0.6.0-kimi-k3.patch
git diff --check
```

The patch makes two changes:

- `WASTE_DISABLE_NATIVE_VQ=1` forces the portable PyTorch VQ path;
- MXFP4 E2M1 values are decoded arithmetically instead of through fragile
  integer advanced indexing.

The guarded launcher already exports `WASTE_DISABLE_NATIVE_VQ=1`.

## Preserved scripts

- `scripts/run-attempt5.sh` checks the source verification marker, index hash,
  kernel version, free VRAM, partial-layer sizes, duplicate processes, and
  kernel/NVIDIA errors. It launches conversion plus GPU/progress/post-run
  watchers in `tmux`.
- `scripts/post-conversion.sh` waits for a successful conversion, round-trips
  one expert per layer, builds WASTE, checks a 46 GB memory plan, benchmarks
  4/8/16/24/32 CPU threads, exercises the learned cache, runs deterministic
  generation, and starts a localhost OpenAI-compatible server.

These are preservation copies, not production-ready service definitions. The
attempt-5 launcher currently requires `source-verify.log` to contain an `hf`
CLI full-pass marker. That marker does not exist because the verifier crashed,
and the same buffered verifier must not be rerun merely to satisfy the gate.
Update the gate to consume a safe direct-I/O verifier before using it.

## Resume checklist

1. Run MemTest86+ outside Linux for at least four complete passes. If it finds
   an error, reseat and test DIMMs in smaller sets; also use Intel baseline
   settings and keep XMP/MCE-style overclocking disabled.
2. Recheck both NVMes and address the separate `nvme0` correctable PCIe errors.
3. Verify all 96 source shards against the pinned Hub revision using direct
   I/O. Do not refill 100+ GB of Linux page cache with `hf cache verify`.
4. Only after that pass, remove the three `.corrupt-20260731-pre-repair`
   backups and `/mnt/data/k3-repair.160e5P` if space is needed.
5. Reclone WASTE at the pinned commit if necessary and apply the preserved
   patch. Repeat the 256-byte E2M1 equivalence test and a real-expert canary.
6. Treat layers 1-3 as provisional: round-trip them against freshly verified
   source data or regenerate them on stable hardware.
7. Revise the launcher verification gate and its exact-kernel check for the
   repaired machine. Consider dropping clean source-cache pages continuously
   or adding direct source reads so conversion cannot recreate the reclaim
   storm.
8. Record and preserve Smarty's existing service/GPU baseline. Do not start a
   duplicate converter or stop services that were not explicitly authorized.
9. Resume with CUDA PyTorch VQ, one job, and deliberate VRAM headroom. The
   converter is layer-resumable and should begin at layer 4 only after layers
   1-3 are revalidated.
10. Finish with source round-trip verification, record-CRC verification,
    CPU-thread benchmarks, deterministic generation, API smoke tests, and a
    healthy localhost server.

## Logs left on Smarty

All logs and mutable status remain under
`/home/francip/models/k3.waste.runs`. Useful files include:

```text
conversion-attempt1.log
conversion-attempt2.log
conversion-attempt3.log
conversion.log                         # attempt 4
source-verify-kernel6-oops-*.log
source-verify-pre-repair.log
source-verify.log                      # empty: kernel 7 run crashed
repair-download-default-xet.log
repair-download.log
pre-reboot-baseline-*.log
supervisor-attempt*.log
gpu-guard*.log
progress*.log
post-conversion*.log
```

Do not move those logs into Git, and definitely do not teach Git LFS about
the 1.42 TiB checkpoint. Git is a historian, not a storage-unit rental scam.
