# Benchmark — `dcp4-cc128` (max-width, DCP4)

**Config:** 128K ctx · `max-num-seqs 8` · DCP4 · `max-num-batched-tokens 2048` · **`gpu-memory-utilization 0.88`** · cudagraph capture 48
**Role:** most simultaneous streams — 8 × 128K ≈ 1M tokens spread across DCP4's pool.
**Stack:** identical fork stack to `dcp2`; `--decode-context-parallel-size 4`.

> `max-num-batched-tokens` is **2048** (not 4096) — 8 concurrent deep prefills would
> otherwise breach the head's 1.5 GiB watchdog, same failure mode documented in
> [`dcp4-cc200.md`](dcp4-cc200.md).

## Status: not yet benchmarked

This lane is defined and boots, but has **not** been run through the concurrency sweep or
deep-fill stress yet. It is the next step after `dcp4-cc200` is confirmed usable under
deep load. The bar is the same: **per-stream throughput must stay usable (≥ ~10 tok/s) at
the full c=8 ceiling, or the lane is worthless.**

<!-- RESULT PENDING:
  - shallow sweep c=1..8 (aggregate + per-stream + TPOT)
  - deep fill 8 x 128K = 1M tokens: per-stream t/s under load, head MemAvailable floor,
    preemption count. 1M tokens is close to DCP4's full pool, so watch for KV-pool
    preemption here (unlike dcp4-cc200's flop, which was activation, not capacity).
-->

## How to reproduce (once being validated)

```bash
GLM_LANE=dcp4-cc128 ./scripts/glm-serve.sh start
for c in 1 4 6 8; do
  vllm bench serve --backend openai-chat --base-url http://<HEAD>:8000 \
    --endpoint /v1/chat/completions --model glm-5.2 \
    --dataset-name random --num-prompts $((c*2)) --max-concurrency $c \
    --random-input-len 512 --random-output-len 512
done
# deep fill:
vllm bench serve ... --num-prompts 8 --max-concurrency 8 \
  --random-input-len 128000 --random-output-len 256
```
