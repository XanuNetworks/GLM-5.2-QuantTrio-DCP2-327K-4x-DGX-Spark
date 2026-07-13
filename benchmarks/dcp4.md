# Benchmark — `dcp4` (max-context specialty)

**Config:** 655K ctx · `max-num-seqs 1` · DCP4 · `max-num-batched-tokens 2048` · cudagraph capture 10
**Role:** single-user, maximum context. DCP4 shards the KV 4 ways → ~655K single-stream.
**Stack:** identical fork stack to `dcp2`; `--decode-context-parallel-size 4`.

## Decode throughput (single stream)

| prompt type | tok/s | notes |
|---|---|---|
| coherent | **~24** | only ~15% slower than dcp2's 327K despite 2× the context |
| random tokens | ~19 | acceptance floor |

DCP4's finer attention-sharding offsets the extra cross-rank comm at long context, so
the decode cost of doubling max context (327K → 655K) is small.

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
  `dcp4-cc200` / `dcp4-cc128`, which trade per-stream context for parallel streams.
