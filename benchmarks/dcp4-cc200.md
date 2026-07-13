# Benchmark — `dcp4-cc200` (multi-user wide, DCP4)

**Config:** 200K ctx · `max-num-seqs 5` · DCP4 · `max-num-batched-tokens 2048` · **`gpu-memory-utilization 0.88`** · cudagraph capture 32
**Role:** widest multi-user lane — up to 5 streams at 200K each on DCP4's 4× KV budget.
**Stack:** identical fork stack to `dcp2`; `--decode-context-parallel-size 4`.

> **Why gmu 0.88 (not the 0.89 the single-stream lanes use):** 5 concurrent *deep* prefills
> hold ~1 GiB more rank-0 working set than a single stream. At 0.89 that breached the head's
> 1.5 GiB watchdog; 0.88 frees ~1.1 GiB/node and clears it (validated below). It costs ~2% KV.
> **Why batched 2048 (not 4096):** small prefill chunks *interleave* a big single-stream prefill
> with everyone else's decode, so one user's 50K paste dips the others instead of starving them.
> Both are correct choices for a concurrency lane, not compromises.

## Concurrency scaling (random tokens, shallow — 512 in / 512 out)

| concurrency | aggregate t/s | per-stream t/s | median TPOT |
|---|---|---|---|
| c=1 | 23.4 | 24.7 | 40 ms |
| c=3 | 38.7 | 13.6 | 74 ms |
| c=4 | 43.1 | 11.4 | 88 ms |
| c=5 | **47.4** | **10.5** | 95 ms |

Even at the full c=5 ceiling every stream holds **~10.5 tok/s** (~47 t/s aggregate) — past
where `dcp2-cc200` caps out (c=3). `max-num-seqs 5` is the server ceiling; sweep c=1..5 against
one boot with no restart. (At real context *depth*, per-stream decode is somewhat below these
512-token figures — decode slows as context grows.)

## KV pool residency (this sets the real ceiling)

One 197K stream measured at **~18% of the KV pool → pool ≈ 1.1M tokens.** So:
- **5 × 197K ≈ 985K ≈ 89% of the pool** → all five stay **resident** (preempt=0). Their prefixes
  stay cached across turns, so a multi-turn agent re-prefills only each turn's *new* tokens.
- **~5 streams at full depth is the ceiling** (~89% is the edge). A 6th deep stream, or streams
  deeper than ~197K, start evicting — and a victim's next turn then eats a cold re-prefill.

## Deep-fill stress — cold 5 × 197K = ~985K tokens (the worst case, not the normal case)

Filling all 5 streams to near-max depth **cold, simultaneously, 0% prefix cache** — the
pathological worst case (5 users each pasting a full 197K doc at the same instant):

| metric | result |
|---|---|
| Successful | 5 / 5 |
| **Preemptions** | **0** — KV pool held |
| **Head MemAvailable floor** | **2.07 GiB** — never near the 1.5 watchdog |
| Median TTFT | **~18 min** (P99 ~30 min) |
| Per-stream "decode" | ~0.8 t/s |

**Read this correctly.** The 18-min TTFT and 0.8 t/s are *not* the lane's serving speed — they
are 5 cold 197K prefills thrashing a single shared token budget, with decode trickling out
between them. **gmu 0.88 makes it memory-safe** (the headline: preempt=0, floor 2.07 GiB); the
throughput here is the cost of the artificial all-cold-at-once fill, which real traffic never does:

- **Agent working up to depth:** per-turn TTFT = prefill of the *delta* (new user msg + tool
  results), history served from prefix cache → small every turn, never a 197K cold prefill.
- **One user pastes 50K mid-chat:** ~1–2 min prefill on *that one stream*, paid once then cached;
  the other 4 dip (shared compute) but don't stall.
- The recipe's `clear_thinking=false` + `--enable-prefix-caching` exist precisely to keep the
  growing prefix stable so these cache hits actually land.

## Verdict

**Memory-safe and usable for genuine multi-user agent load at gmu 0.88 + batched 2048**, within
the ~5-streams-at-≤197K residency envelope. The only unusable regime is the synthetic
all-cold-simultaneous max-depth fill — avoid needing 5 cold 197K prefills at once and the lane is fine.

## How to reproduce

```bash
GLM_LANE=dcp4-cc200 ./scripts/glm-serve.sh start   # gmu 0.88 baked in
# shallow concurrency sweep (no restart needed for c<=5):
for c in 1 3 4 5; do
  vllm bench serve --backend openai-chat --base-url http://<HEAD>:8000 \
    --endpoint /v1/chat/completions --model glm-5.2 \
    --dataset-name random --num-prompts $((c*2)) --max-concurrency $c \
    --random-input-len 512 --random-output-len 512
done
# cold deep-fill worst case (watch head MemAvailable + preemptions, NOT the TTFT):
vllm bench serve ... --num-prompts 5 --max-concurrency 5 \
  --random-input-len 197000 --random-output-len 128
```

## Notes

- A watchdog kill *during prefill ramp* with KV usage still low = rank-0 working set, fixed by gmu
  (done). Preemptions with KV usage high = pool capacity — the residency ceiling above.
- The head node is the memory-critical one under concurrent deep prefill, not the workers
  (workers held ~5 GiB throughout).
