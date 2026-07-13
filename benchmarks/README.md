# benchmarks/ — measured numbers, one file per lane

Detailed, reproducible benchmarks for each `GLM_LANE`. Every file lists the exact config,
the numbers, and the `vllm bench serve` command to reproduce them. Numbers are honest —
random-token runs are the worst-case floor; coherent (natural text) runs ~10–15% higher
because MTP draft-acceptance is better on real text.

| lane | file | ctx | streams | headline |
|---|---|---|---|---|
| `dcp2` | [dcp2.md](dcp2.md) | 327K | c=1 | coherent ~24.6, ~21 flat to 300K · 87/100 · flagship (faster-or-equal to dcp4 ≤327K) |
| `dcp4-cc200` | [dcp4-cc200.md](dcp4-cc200.md) | 200K | c=3 | 3×200K fits (666K pool) · ~22/stream at c=3 · 87/100 |
| `dcp4` | [dcp4.md](dcp4.md) | 655K | c=1 | coherent flat ~21 to 655K · 87/100 · the >327K specialist (no decode edge ≤327K) |
| `dcp4-cc128` | [dcp4-cc128.md](dcp4-cc128.md) | 128K | c=5 | 5×128K fits (651K pool) · ~22/stream at c=5 · 89/100 |

## Method notes (apply to every lane)

- **Perf numbers are the *synthetic floor*, not coherent.** The llama-benchy `tg` figures
  (via `tool-eval-bench --perf`) fill the context with synthetic tokens, so the model's
  output is unpredictable → **low MTP draft-acceptance → lower tok/s.** Real coherent
  workloads (natural text/code) accept more drafts and run **~5–10% higher**. Where a file
  shows a **coherent** number (e.g. dcp2 ~24.5 from real coding prompts), that's the
  real-world figure; the synthetic `tg` rows are a pessimistic floor. The *depth* drop
  (decode slowing as context grows) is an attention effect and applies to both.
- **Cold prefill only.** Reusing a fixed seed lets prefix-caching serve a warm KV and
  reports fake-fast prefill — measure with novel prompts.
- **Per-stream t/s = 1000 / median TPOT (ms).** Aggregate is the server's total output rate.
- **Deep-fill matters.** A lane can look fine at 512-token prompts and flop when streams
  actually fill — a concurrency lane must be validated with a *cold, simultaneous* deep fill
  (all streams at full context at once) against both the KV pool (preemption) and the head
  watchdog (a kill). `dcp4-cc128.md` walks through that tuning. Always stress with deep
  prompts, not just toy ones.
- **Only `gpu-memory-utilization` sizes the KV pool** — not `max-num-seqs`, not cudagraph
  capture (both measured to leave it unchanged). See `dcp4-cc128.md` for the gmu→pool table.
