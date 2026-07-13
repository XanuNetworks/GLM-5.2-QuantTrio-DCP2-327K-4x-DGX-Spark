# Benchmark — `dcp2` (flagship)

**Config:** 327K ctx · `max-num-seqs 1` · DCP2 · `max-num-batched-tokens 2048` · cudagraph capture 10
**Role:** single-user, maximum depth + speed. The default lane.
**Stack:** fork (local-inference-lab/vllm @e232d26 + 3 patches + b12x@9cd63a7), MTP k=4, `index_topk_pattern`, dual-rail mesh-NCCL, `kv-cache-dtype fp8_ds_mla`, gmu 0.89.

## Decode throughput (single stream)

| prompt type | tok/s | notes |
|---|---|---|
| coherent (real chat/code) | **~25** | MTP acceptance is higher on natural text |
| random tokens (worst case) | ~23 | acceptance floor; synthetic |
| coherent @ ~32K depth | ~20 | stays flat-ish to depth |

Healthy-cluster decode measured at **22.9 tok/s** (random, cold) after clearing the
node-.12 stuck-clock bug — see the "Is your decode slow?" section in the main README.
Before the fix the same config read **15.6 tok/s** (one GPU wedged at 660 MHz gated all 4).

## Prefill

| metric | value |
|---|---|
| cold prefill throughput | ~720 tok/s (3-sample mean, novel prompts) |

Measure prefill with **novel** prompts — reusing a fixed seed lets prefix-caching
serve a warm KV and reports fake-fast prefill.

## Tool calling

Tool Eval Bench: **88 / 100** (★★★★) — GLM's `glm47` tool-call parser + `enable-auto-tool-choice`.

## How to reproduce

```bash
GLM_LANE=dcp2 ./scripts/glm-serve.sh start
# coherent decode:
./utils/benchmark.sh            # runs coherent + random + cold-prefill legs
```

## Notes / gotchas

- `draft_tensor_parallel_size` is **locked at 4** under DCP2 (vLLM requires 1 or target-TP,
  and divisible by dcp_size=2 → only 4 is legal). CosmicRaisins' "draft tp=1" idea is non-DCP only.
- `clear_thinking=false` keeps the multi-turn prefix stable for cache hits (context cost fine at 327K).
- Coherent > random by ~+10–15% because MTP draft-acceptance is higher on natural text.
