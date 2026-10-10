# Paired GLM runtime comparison

Run one serving recipe at a time on Static and Shock. These probes call an
existing endpoint; they do not start, stop or change the model. Use separate
output directories for each immutable candidate and a shared fixture directory.
Keep outputs outside the repository, under `~/Desktop/ai reports/`.

```bash
python3 bench/glm53-tp2/capacity.py \
  --base http://192.168.2.101:2070 --model glm-5.3-flash-2x-dgx \
  --clients 1 --prompt-tokens 2048 --output-tokens 512 --min-tokens 512 \
  --reset-prefix-cache --tag p2k-c1-cold-1 \
  --output-dir '/path/to/evidence/stable' --fixture-dir '/path/to/evidence/fixtures'
```

For the experiment use port 2071 and model `glm-5.3-flash-2x-dgx-experimental`.
Run short canaries before large prompts. Compare 1/4/8 clients at short context,
then single 64K/128K/near-512K prompts and eight independent near-64K histories.
Use distinct tags for repeats. `--reset-prefix-cache` explicitly makes the
request cold; omit it for the immediate replay and verify cache-hit counters.
Record any background work that could affect timing.

The reported stream decode rate uses completion tokens minus one divided by
the first-to-last content interval. Speculative decoding emits bursts, so
compare sustained outputs as well as end-to-end wall time. Forced minimum
output lengths measure synthetic throughput; they are not coding quality or
evidence of model repetition during normal use. A successful HTTP response
alone is not a pass: all retrieval markers, complete SSE termination and
expected prompt-token accounting must pass.

Compare actual RUNNING/WAITING peaks and overlapping decode intervals. The
number of submitted clients and the startup cache token estimate do not prove
active concurrency. Keep bytes reserved, cache blocks consumed and host free
memory separate when reporting a DCP memory saving.
