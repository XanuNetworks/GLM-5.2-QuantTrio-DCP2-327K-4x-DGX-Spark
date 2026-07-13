# Benchmark — `dcp4` (max-context specialty)

**Config:** 655K ctx · `max-num-seqs 1` · DCP4 · `max-num-batched-tokens 2048` · cudagraph capture 10
**Role:** single-user, maximum context. DCP4 shards the KV 4 ways → ~655K single-stream.
**Stack:** identical fork stack to `dcp2`; `--decode-context-parallel-size 4`.

## Decode throughput (single stream)

Fresh re-measure (2026-07-13). **Coherent** = real code prompts; synthetic `tg` is a floor.

| context | coherent decode t/s | synthetic floor | prefill |
|---|---|---|---|
| 32K | 20.8 | 23.2 | ~551 |
| 120K | 21.1 | 21.6 | ~542 |
| 300K | 21.0 | — | — |

Decode is **flat ~21 t/s across all depths** — DCP4's 4-way KV sharding keeps the per-token
cost steady out to 300K+.

**But `dcp2` is faster-or-equal at every depth ≤327K** (coherent: 23.7 vs 20.8 at 32K; then
tied ~21 from 120K on). DCP4's extra 4-way comm tax isn't offset by its finer sharding until
very deep context — and even at 300K it only *ties* dcp2, never beats it. So the synthetic
"flatter to depth" edge is a floor artifact; on real text DCP4 has no decode advantage.

**dcp4's real (and only) advantage is the context *ceiling*: 327K → 655K.** Use it when you
genuinely need >327K per stream; at or below 327K, `dcp2` is the faster choice.

Pool: **707,584 tokens** (1.08× at 655K → one full 655K stream). Tool Eval Bench: **87/100 ★★★★**.

## Memory

| node | MemAvailable at serve (idle-with-model) | floor |
|---|---|---|
| head (.11) | ~3.3 GiB | 1.5 GiB watchdog |
| worker | ~5.2 GiB | 1.5 GiB watchdog |

Healthy headroom at c=1 — the single-stream prefill transient stays well clear of the floor.

## How to reproduce

```bash
GLM_LANE=dcp4 ./scripts/glm-serve.sh start
curl -s http://<HEAD>:8000/v1/models   # confirm "max_model_len": 655360
./utils/benchmark.sh
```

## Notes

- Serves at `max_model_len 655360` — verified end-to-end.
- This is a **single-stream** lane (c=1). For concurrency on the DCP4 budget, use
  `dcp4-cc128`, which trades per-stream context for parallel streams.
