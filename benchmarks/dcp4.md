# Benchmark — `dcp4` (max-context specialty)

**Config:** 655K ctx · `max-num-seqs 1` · DCP4 · `max-num-batched-tokens 2048` · cudagraph capture 10
**Role:** single-user, maximum context. DCP4 shards the KV 4 ways → ~655K single-stream.
**Stack:** identical fork stack to `dcp2`; `--decode-context-parallel-size 4`.

## Decode throughput (single stream)

Fresh re-measure (2026-07-13, llama-benchy via `tool-eval-bench --perf`):

| context depth | decode t/s | prefill t/s |
|---|---|---|
| ~16K | **24.0** | 559 |
| 32K | 23.2 | 551 |
| 120K | **21.6** | 542 |

**This lane holds decode to depth better than the flagship** — only ~10% off shallow even
at **120K context** (dcp2 drops ~20% by 32K). DCP4's 4-way attention sharding offsets the
extra cross-rank comm at long context, so decode barely degrades as the context grows.

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
