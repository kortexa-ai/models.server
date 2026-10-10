# Paired GLM runtime comparison

Run one serving recipe at a time on Static and Shock. These probes call an
existing endpoint; they do not start, stop or change the model. Use separate
output directories for each immutable candidate and a shared fixture directory.
Keep outputs outside the repository, under `~/Desktop/ai reports/`.

```bash
python3 bench/glm53-tp2/capacity.py \
  --base http://192.168.2.101:2070 --model glm-5.3-flash-2x-dgx \
  --clients 1 --prompt-tokens 2048 --output-tokens 512 --min-tokens 512 \
  --tag p2k-c1-1 \
  --output-dir '/path/to/evidence/stable' --fixture-dir '/path/to/evidence/fixtures'
```

For the experiment use port 2071 and model `glm-5.3-flash-2x-dgx-experimental`.
Run short canaries before large prompts. Compare 1/4/8 clients at short context,
then single 64K/128K/near-512K prompts and eight independent near-64K histories.
Use distinct tags for repeats. Where the server exposes its development cache
endpoint, `--reset-prefix-cache` explicitly makes the request cold; otherwise
use a fresh fixture and verify zero cache hits. The stable server does not
expose that endpoint. For an immediate replay verify cache-hit counters before
calling it a warm-prefix result.
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

`quality.py --base URL --model ID --output /new/file.json` tests streamed tools,
empty results and one corrected lookup after an error. Add `--replay-file`
only for a private, reconstructed OMP request. It runs one bounded continuation
and never executes generated tools. Keep that payload outside Git.

`images.py` takes the same base/model/output arguments and sends eight generated
2560×960 PNGs with distinct numeric codes. All eight codes must be returned in
order. It uses only the Python standard library.
