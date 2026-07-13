# Benchmark — `dcp4-cc128` (multi-user, DCP4)

**Config:** 128K ctx · `max-num-seqs 5` · DCP4 · `max-num-batched-tokens 2048` · **`gpu-memory-utilization 0.885`** · cudagraph capture 32
**Role:** multi-user serving — up to 5 concurrent agents at 128K each, tuned to fit *fully* in the KV pool with no preemption.
**Stack:** identical fork stack to `dcp2`; `--decode-context-parallel-size 4`.

> This lane's numbers are the result of a long tuning pass; the reasoning is worth reading
> because it explains what actually controls capacity on GB10 (spoiler: gmu, nothing else).

## The one knob that sizes the pool: `gpu-memory-utilization`

Measured pool size (DCP4, seqs=5) at three gmu values — vLLM reports it at boot as
`GPU KV cache size: N tokens`:

| gmu | KV pool | idle head | head floor under 5× cold 128K prefill | verdict |
|---|---|---|---|---|
| 0.87 | 502K | — | — | pool too small |
| 0.88 | 595K | 4.59 GiB | **2.62 GiB** (plateau, safe) | head-safe, but 640K > 595K → preempts at peak |
| **0.885** | **651K** | 3.93 GiB | **~1.8 GiB** (plateau, no kill) | ✅ **5×128K=640K fits AND head survives** |
| 0.89 | 709K | 3.03 GiB | **1.16 GiB → watchdog kill** | pool fits but head dies |

**`max-num-seqs` and cudagraph `capture` do NOT change the pool** — measured: seqs 6 vs 8 → 604K vs 612K; capture 32 vs 48 → 606K vs 612K (all noise). Only gmu moves it (~+11K tokens per +0.001). So the KV pool is gmu-bound at ~650K here; `seqs` controls *head survival* (fewer = safer), `capture` controls *decode-graph coverage* (`≥ seqs×(1+spec)` = 25), neither touches capacity.

**0.885 is the tuned sweet spot:** the smallest gmu whose pool (651K) fits 5×128K (640K) preempt-free, while the head still plateaus above the 1.5 GiB watchdog under the worst case. It's a deliberate razor: 0.89 crashes the head, 0.88 leaves the pool ~5% short.

## Deep-fill validation — 5 × 128K = 640K, cold, simultaneous (the worst case)

| metric | result |
|---|---|
| Successful | **5 / 5** |
| Preemptions | **0** — 640K fits the 651K pool |
| Head floor (watchdog-log verified) | **~1.8 GiB** — no kill |

Verified against the watchdog log (`/tmp/glm-memwatch.log`), not the mem sampler — the
sampler misses the fast trough (at 0.89 the head fell 3.03→1.16 in ~50 s between samples).

## Concurrency scaling (llama-benchy `tool-eval-bench --perf`, pp8000/tg512, 2026-07-13)

| c | per-stream decode t/s | prefill t/s |
|---|---|---|
| 1 | 23.3 | 571 |
| 2 | 20.8 | 568 |
| 3 | 22.5 | 568 |
| 4 | 22.3 | 566 |
| 5 | 22.7 | 566 |

**Per-stream decode holds ~22 t/s all the way to c=5** — the `seqs=5` config batches its 5
streams with almost no per-stream penalty (aggregate ~113 t/s at c=5). TTFT grows with
concurrency (prefills serialize), but each stream decodes fast once started. The widest
lane: most simultaneous agents, each still ~22 t/s.

Pool: **650,752 tokens** (4.96× at 131K → 5×128K=640K fits). Tool Eval Bench: **89/100 ★★★★**.

## Why DCP4 (and not DCP2)

DCP shards the KV pool, so pool ∝ DCP degree: DCP4 gives ~2× DCP2's pool at the same gmu
(same relationship as the single-stream lanes: dcp2=327K vs dcp4=655K). At gmu 0.885, DCP2
would be ~325K — nowhere near 5×128K=640K. **DCP4's 4-way sharding is what provides the
capacity** for 5 deep streams; the cost is more per-step decode comm (which is why the
single-user flagship stays on DCP2 for speed).

## Tuning it yourself

`GLM_SEQS` sets the concurrency *default*, not a cap — raise or lower it for your workload
(numbers to guide you are above). Raising it is usually fine: real traffic rarely has every
stream at full 128K at once, so the pool usually isn't exhausted, and if it is you get
**graceful preemption** (a re-prefill stall on one stream), not a crash. The one hard edge:
a burst of many *cold deep* prefills at a high `GLM_SEQS` can breach the head watchdog (a
container restart) — that's the boundary the 0.885/seqs=5 default sits safely below.

```bash
GLM_LANE=dcp4-cc128 ./scripts/glm-serve.sh start          # 5×128K, tuned default
GLM_LANE=dcp4-cc128 GLM_SEQS=6 ./scripts/glm-serve.sh start # more agents, at your own margin
```
