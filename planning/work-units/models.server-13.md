# Issue #13: Expand Qwen to two full-context slots

Related API catalog work: https://github.com/kortexa-ai/api.server/issues/40

## Decision

Run Qwen 3.8 27B with two 262,144-token request slots by allocating 524,288
tokens of total llama.cpp context. Advertise the same per-request limit in all
installed local harnesses. Mark LFM2.5 8B-A1B as reasoning-only model metadata
so the API catalog can describe it correctly.

## Validation record

- All 48 model-server tests and JSON validation passed.
- Qwen restarted through `ktxsvc` with `-c 524288 --parallel 2`; `/props` and
  `/slots` report two 262,144-token slots, and a live completion passed.
- Qwen uses 41,190 MiB; the complete 6000 stack uses 61,167 MiB and leaves
  36,075 MiB free.
- OMP, Pi, Prime Agent, and Hermes declarations were updated wherever installed
  on Smarty, Snappy, and Scrappy. Live canaries passed for every runnable
  harness; Scrappy native Hermes YAML validates, while its previously broken
  Python launcher remains unavailable.
- The deployed API catalog advertises Qwen at 262,144 tokens and LFM2.5 8B-A1B
  as text-only reasoning without configurable effort; Gemma is absent.
