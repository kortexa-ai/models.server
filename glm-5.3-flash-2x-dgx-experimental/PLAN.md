# GLM on pinned vLLM main: paired Spark experiment

Keep `glm-5.3-flash-2x-dgx` as the recovery recipe. Its model manifest,
launcher, setup receipt, source checkouts, containers and image must remain
usable. This recipe is the working experiment. Promotion into the stable
recipe requires a later decision based on the measurements.

## Controlled comparison

1. Record both stable ranks, image and source pins, weight hashes, launch
   arguments, cache geometry and available host memory. Run the same bounded
   speed and correctness probes that the candidate will run. Verify that
   stable setup receipts and preserved containers permit a managed restart.
2. Pin vLLM main at `276fbcff2717bd934cfa37c8a2e4c391f3e7237b`.
   Build one immutable ARM64 image, with explicit native-wheel provenance.
   Port only the required mixed-weight loader first. Inventory each old
   overlay; do not mount old model, allocator or scheduler replacements over
   new upstream code. Add performance patches individually after correctness.
3. Use the existing verified NVFP4/mixed-FP8 target and FP8 DFlash2 weights,
   a 524288-token ceiling, eight scheduled sequences, a 6919-token batch budget
   and a 6 GiB cache per rank for the first comparison. Keep image allowance at
   eight. Serve the experiment on its own port, 2071, with bootstrap port 29670.
4. Establish TP2/DCP1 correctness and record cold/warm prefill, decode,
   end-to-end throughput, draft acceptance, cache preemption and actual active
   concurrency. Use identical prompts and output lengths across candidates.
   Separate forced-length synthetic throughput from normal coding responses.
5. Test TP2/DCP2 with cache interleave 4 if the selected backend supports the
   full GLM/FP8/DFlash combination. Record physical allocation and logical
   per-request block consumption. Check which groups shard and which remain
   replicated; recurrent and speculative state is a material cost.
6. Hold pool bytes constant first to measure capacity. Then reduce the pool
   while holding the tested workload constant to measure real memory savings.
   Record both ranks, since rank 0 also runs the API and vision path. Report
   measured memory separately from layout estimates and allocator reservations.
7. Explore controls one at a time: draft lookahead, cache layout/grouping,
   recurrent-state dtype, graph capture and prefill workspace. Identify knobs
   present in the pinned source versus those requiring an unmerged patch.
   Preserve each tested manifest and any applied patch with source provenance.
   Do not attribute a precision or speculation change to DCP.
8. Run short and long retrieval, prefix reuse, mixed-length concurrency,
   streamed tool calls, empty tool results, eight-image histories and bounded
   reconstructed OMP continuation. Generated OMP commands are not executed.
   Fail on malformed streams, repeated calls without progress, lost tool
   results, CUDA faults or memory exhaustion. Extend load only after canaries.

## Recovery

Stop the experimental API rank, then worker, through `ktxsvc`. Start the
preserved stable worker, then API rank. Disable the experimental units and
re-enable the stable units through `ktxsvc` so boot selects only one recipe.
Check both service PIDs, port 2070,
`/health`, `/v1/models` and a real completion. Never run both recipes together.
Keep stable weights, images, source directories and stopped containers.
Use Git for repository delivery and direct Spark-to-Spark transfer for images.
Do not change client defaults or public routing to make a benchmark work.

Record changing execution state and results in models.server issue #60 and
immutable external evidence. A failed candidate is an experimental result,
not permission to weaken correctness or claim an unmeasured speedup.
