# Benchmark — `dcp4-cc200` (multi-user, DCP4)

**Config:** 200K ctx · `max-num-seqs 3` · DCP4 · `max-num-batched-tokens 2048` · **`gpu-memory-utilization 0.885`** · cudagraph capture 16
**Role:** multi-user serving — up to 3 concurrent agents at 200K each, tuned to fit *fully* in the KV pool with no preemption.
**Stack:** identical fork stack to `dcp2`; `--decode-context-parallel-size 4`.

## Why DCP4, not DCP2

The KV pool scales with DCP degree. At gmu 0.885: **DCP2 pool = 327K, DCP4 pool = 651K** (measured). So 3×200K = 600K:
- on **DCP2** → 600K ≫ 327K pool (only ~1.6 streams at 200K fit) → heavy preemption. *This is why the earlier "dcp2-cc200" was retired — DCP2 physically can't hold 3×200K, and no gmu fixes it.*
- on **DCP4** → 600K < 651K pool (**3.25× at 200K**) → all 3 resident, preempt-free. ✅

## Deep-fill validation — 3 × 197K ≈ 590K, cold, simultaneous

| metric | result |
|---|---|
| Successful | **3 / 3** |
| Preemptions | **0** — fits the 651K pool |
| Head floor (watchdog-log verified) | **~2.2 GiB** — no kill |

Head margin here (~0.7 GiB) is more comfortable than `dcp4-cc128`'s (~0.3) — fewer concurrent prefills (3 vs 5) means less rank-0 working set, even though each is deeper.

## Concurrency scaling (llama-benchy `tool-eval-bench --perf`, pp8000/tg512, 2026-07-13)

| c | per-stream decode t/s | prefill t/s |
|---|---|---|
| 1 | 21.9 | 568 |
| 2 | 22.4 | 551 |
| 3 | 21.6 | 563 |

**Per-stream decode holds ~22 t/s all the way to c=3** — DCP4 batches these 3 streams with
almost no per-stream penalty (aggregate ~65 t/s at c=3). TTFT grows with concurrency
(prefills serialize), but once a stream is decoding it stays fast.

Pool: **666,496 tokens** (3.33× at 200K → 3×200K=600K fits, preempt-free). Tool Eval Bench: **87/100 ★★★★**.

## Tuning it yourself

`GLM_SEQS` sets the concurrency *default*, not a cap. At 200K per stream the pool
(651K) backs ~3.25 streams, so `GLM_SEQS=3` is the fit-fully point; going higher works
for mixed-depth traffic (real agents rarely sit at full 200K) but a burst of 4+ cold
200K prefills would preempt (graceful) and eventually pressure the head watchdog. See
[scripts/README](../scripts/README.md#concurrency-lanes-what-actually-controls-capacity)
for the pool↔head reasoning.

```bash
GLM_LANE=dcp4-cc200 ./scripts/glm-serve.sh start          # 3×200K, tuned default
GLM_LANE=dcp4-cc200 GLM_SEQS=2 ./scripts/glm-serve.sh start # 2 agents, deeper headroom
```
