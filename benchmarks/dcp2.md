# Benchmark — `dcp2` (flagship)

**Config:** 327K ctx · `max-num-seqs 1` · DCP2 · `max-num-batched-tokens 2048` · cudagraph capture 10
**Role:** single-user, maximum depth + speed. The default lane.
**Stack:** fork (local-inference-lab/vllm @e232d26 + 3 patches + b12x@9cd63a7), MTP k=4, `index_topk_pattern`, dual-rail mesh-NCCL, `kv-cache-dtype fp8_ds_mla`, gmu 0.89.

## Decode throughput (single stream)

Fresh healthy-cluster re-measure (2026-07-13). **Coherent** = real code prompts (what you'll
actually see); the llama-benchy synthetic `tg` is a pessimistic floor (random filler tanks
MTP draft-acceptance).

| context | coherent decode t/s | synthetic floor |
|---|---|---|
| shallow (~16K) | **24.6** | 23.2 |
| 32K | **23.7** | 18.6 |
| 120K | 21.0 | — |
| 300K | **21.4** | — |

**Decode barely degrades to depth on real workloads** — it eases from ~24.6 to ~21 by 120K,
then *plateaus* (~21 all the way to 300K, near the 327K ceiling — sparse attention caps the
per-token cost). The synthetic "18.6 @32K" is a floor artifact; the coherent figure is 23.7.
`dcp2` is faster-or-equal to `dcp4` at **every depth it reaches** (≤327K) — see [dcp4.md](dcp4.md).

Prefill **~714 t/s**. The healthy-cluster numbers came after clearing a stuck-clock bug (one
GPU wedged at 660 MHz gated all four): the same config read **15.6 tok/s** before the
cold-cycle fix — see [docs/troubleshooting.md](../docs/troubleshooting.md).

## Prefill

| metric | value |
|---|---|
| cold prefill throughput | ~714 tok/s (16K novel ingest, llama-benchy) |

Measure prefill with **novel** prompts — reusing a fixed seed lets prefix-caching
serve a warm KV and reports fake-fast prefill.

## Tool calling

Tool Eval Bench: **87 / 100** (★★★★) — GLM's `glm47` tool-call parser + `enable-auto-tool-choice`. Consistent 87–89 across all lanes (model/parser-level).

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
