# benchmarks/ — measured numbers, one file per lane

Detailed, reproducible benchmarks for each `GLM_LANE`. Every file lists the exact config,
the numbers, and the `vllm bench serve` command to reproduce them. Numbers are honest —
random-token runs are the worst-case floor; coherent (natural text) runs ~10–15% higher
because MTP draft-acceptance is better on real text.

| lane | file | ctx | streams | headline |
|---|---|---|---|---|
| `dcp2` | [dcp2.md](dcp2.md) | 327K | c=1 | ~25 t/s coherent · 88/100 tool-eval · flagship |
| `dcp2-cc200` | [dcp2-cc200.md](dcp2-cc200.md) | 200K | c=3 | ~41 t/s aggregate · ~14 each · TTFT ~1.3 s |
| `dcp4` | [dcp4.md](dcp4.md) | 655K | c=1 | ~24 t/s coherent · max context |
| `dcp4-cc200` | [dcp4-cc200.md](dcp4-cc200.md) | 200K | c=5 | ~47 t/s aggregate · ~10.5 each · gmu 0.88 · deep-fill memory-safe (preempt=0) |
| `dcp4-cc128` | [dcp4-cc128.md](dcp4-cc128.md) | 128K | c=8 | gmu 0.88 · not yet benchmarked |

## Method notes (apply to every lane)

- **Cold prefill only.** Reusing a fixed seed lets prefix-caching serve a warm KV and
  reports fake-fast prefill — measure with novel prompts.
- **Per-stream t/s = 1000 / median TPOT (ms).** Aggregate is the server's total output rate.
- **Deep-fill matters.** A lane can look fine at 512-token prompts and flop when streams
  actually fill — see `dcp4-cc200.md` for a lane that passed shallow and got its head node
  watchdog-killed 8 s into a real 5×150K fill. Always stress the concurrency lanes with
  deep prompts, not just toy ones.
