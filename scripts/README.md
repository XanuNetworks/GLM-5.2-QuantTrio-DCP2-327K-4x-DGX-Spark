# scripts/ — how the launch chain fits together

Three scripts, three **layers** of one launch. You only ever run `glm-serve.sh`.

```
YOU RUN:  GLM_LANE=dcp2 ./glm-serve.sh start
              │
              ▼
  1. glm-serve.sh            ── WRAPPER (orchestration)
     • subcommands: start / stop / status / logs / dry-run
     • preflight: GPU clock-health check, arm memory watchdog,
       apply NIC PFC, drop caches, validate staging
     • picks the lane (GLM_LANE), then calls ↓
              │
              ▼   "$HOME/vllm/glm-node-launch.sh"
  2. glm-node-launch.sh      ── PER-NODE LAUNCHER (internal)
     • builds the vLLM serve flags + env + mounts for the lane
     • issues `docker run` on each node (workers headless, head last)
     • the container's command is ↓
              │
              ▼   docker run … /glm-container-entrypoint.sh vllm serve …
  3. glm-container-entrypoint.sh ── IN-CONTAINER ENTRYPOINT (internal)
     • runs INSIDE each of the 4 containers
     • auto-detects the RoCEv2 GID index, then `exec`s `vllm serve`
```

The optional unified-memory watchdog now lives in [`../utils/glm-memwatch.sh`](../utils/README.md#glm-memwatchsh) — `glm-serve.sh` stages + arms it on each node.

## Lanes (config variants live HERE, not in new scripts)

Config differences are **lanes**, selected with `GLM_LANE`, defined inside
`glm-node-launch.sh`. Do **not** clone `glm-serve.sh` per config.

Naming: `<dcp-size>[-cc<per-stream-K>]`. Bare = single-stream (c=1); `-ccNNN` = multi-user, NNN K per stream.

| `GLM_LANE` | ctx | concurrency | measured | role |
|---|---|---|---|---|
| **`dcp2`** (default) | 327K | c=1 | ~25 t/s coherent | 🏆 flagship — single-user, max depth + speed |
| **`dcp4-cc200`** | 200K | c=3 | 3×200K=600K fits the 651K pool preempt-free · head-safe | multi-user on DCP4 |
| **`dcp4`** | 655K | c=1 | ~24 t/s coherent | max-context specialty (DCP4's 4× KV) |
| **`dcp4-cc128`** | 128K | c=5 | 5×128K=640K fits the 651K pool preempt-free · head-safe | multi-user on DCP4 |

Full per-lane numbers + reproduce commands: [`../benchmarks/`](../benchmarks/README.md).

All lanes run the identical fork stack (image, b12x, MTP k=4, mesh-NCCL, `index_topk_pattern`, `clear_thinking`) — they differ **only** in `max-model-len` / `max-num-seqs` / `max-num-batched-tokens` / cudagraph capture / DCP size / **gpu-memory-utilization**. DCP shards the KV (pool ∝ DCP degree), so it's really one budget you spend on *context* (dcp4 → 655K single-stream) or *width* (dcp4-cc128 → 5×128K streams on DCP4's 2×-bigger pool).

### Concurrency lanes: what actually controls capacity

It comes down to one tension — and two of the knobs you'd *expect* to matter turn out not to. Measured on DCP4/seqs=5, the KV pool is set by **`gpu-memory-utilization` alone**: gmu 0.87→502K, 0.88→595K, 0.885→651K, 0.89→709K tokens (~+11K per +0.001). `max-num-seqs` and cudagraph `capture` do **not** move it (seqs 6 vs 8 → 604K vs 612K; capture 32 vs 48 → 606K vs 612K — noise).

| knob | controls | does *not* control |
|---|---|---|
| **`L_GMU`** | KV pool size **and** head-memory headroom — in opposition | — |
| **`L_SEQS`** | concurrency + head survival (fewer seqs = more headroom) | the pool |
| **`L_CAPTURE`** | decode-graph coverage; set `≥ seqs×(1+spec_tokens)` (=25 at seqs=5) | the pool |
| **`L_DCP`** | pool ∝ DCP degree (DCP4 ≈ 2× DCP2 at the same gmu) | — |

**The gmu tension is the whole game:** higher gmu = bigger pool (fits more resident streams) but *less* head headroom; lower = head-safe but smaller pool. Each lane's gmu is tuned to where its advertised streams **fit the pool preempt-free** *and* the head survives a cold-prefill burst. For `dcp4-cc128` that's **0.885** — 651K pool fits 5×128K=640K, head plateaus ~1.8 GiB (0.89 fits the pool but kills the head; 0.88 is head-safe but ~5% short).

**Read a failure by its signature** (note the two fixes pull gmu in *opposite* directions — that's why the default is a deliberate razor):

| symptom | cause | fix |
|---|---|---|
| head watchdog-kills **during prefill ramp**, KV usage still low | rank-0 prefill working set (scales with *concurrent* deep prefills) | **lower gmu** or **fewer seqs** — *not* batched-tokens, *not* context |
| **preemptions > 0**, per-stream t/s sags mid-decode | resident set exceeds the pool | **higher gmu** (bigger pool) or fewer/shorter streams |

**Don't be scared off by a cold-fill TTFT.** Filling every stream to max depth *cold and simultaneously* (0% prefix cache) gives a pathological TTFT — a stress artifact, not the serving speed. Real multi-turn agents **amortize via prefix cache**: each turn only re-prefills its *delta*, the resident history is a cache hit, so per-turn TTFT stays small. `clear_thinking=false` + `--enable-prefix-caching` (every lane) keep that prefix stable.

### Runtime overrides (optional — most people should just use the lane default)

Each lane's defaults are already tuned; **if in doubt, pick a lane and run it as-is.** The overrides below are only for when you've looked at [`../benchmarks/`](../benchmarks/README.md) and want a different point on the curve for your workload (many shallow agents vs. a few deep ones).

> ⚠️ **Two costs to know before you turn a knob up:**
> - **Higher concurrency → slower per-stream decode.** It's a throughput/latency trade, not free capacity. E.g. `dcp4-cc128`: c=1 and c=5 both hold ~22 tok/s/stream (DCP4 batches well; aggregate scales ~linearly). More agents, each slower.
> - **Higher context / bigger prefill → much longer TTFT.** Loading a deep prompt cold is `O(n²)` attention; a fresh 197K prompt is *minutes* of time-to-first-token, and several deep prompts arriving cold at once multiply that (they share one prefill budget). Steady multi-turn use hides this via prefix cache — a *cold* deep load does not.

Set these env vars at launch to override the lane without touching the file:

| env var | overrides | example |
|---|---|---|
| `GLM_SEQS` | `--max-num-seqs` (concurrency ceiling) | `GLM_LANE=dcp4-cc128 GLM_SEQS=5 ./glm-serve.sh start` |
| `GLM_GMU`  | `--gpu-memory-utilization` (sizes the pool) | `GLM_LANE=dcp4-cc128 GLM_GMU=0.89 ./glm-serve.sh start` |

**`GLM_SEQS` sets a lane's concurrency default — it is not a cap, and there's no enforced maximum.** It does **not** resize the pool (that's `GLM_GMU`); it changes how many streams schedule and how much head headroom you keep. Raising it is usually safe: real traffic rarely has every stream at full context at once, so the pool usually isn't exhausted, and if it is you get **graceful preemption** (a re-prefill stall on one stream), not a crash. The one hard edge: a burst of many *cold deep* prefills at a high `GLM_SEQS` can breach the head watchdog (a container restart) — the boundary each lane's default sits below. (Raising seqs well above the default may also want a bigger `L_CAPTURE` = `seqs×(1+spec_tokens)`, else big decode batches run eager — slower, never a crash.) vLLM prints the actual pool at boot: `GPU KV cache size: N tokens` / `Maximum concurrency for <ctx>/request: X.XX×`.

```bash
GLM_LANE=dcp2 ./glm-serve.sh start        # 327K flagship
./glm-serve.sh start                      # default = dcp2 (flagship)
GLM_LANE=dcp4-cc128 ./glm-serve.sh start  # multi-user (5×128K)
GLM_LANE=dcp2 ./glm-serve.sh dry-run      # print docker commands, run nothing
./glm-serve.sh stop                       # tear down on all nodes
```

## How to add a lane (e.g. a concurrency-tuned config)

Everything is data — add a block, don't add a file:

1. In `glm-node-launch.sh`, extend the `GLM_LANE` `case` with your new value
   (e.g. `dcp4-cc256`): set the lane vars (`L_MAXLEN`, `L_SEQS`, `L_BATCHED`,
   `L_CAPTURE`, `L_DCP`). For a multi-user config that means a higher `L_SEQS`
   (`--max-num-seqs`) on a DCP4 budget — the base recipe is `L_SEQS=1`, single-user max-speed.
2. Add the new value to the validation `case` near the top of the file.
3. Add a row to the lane table above, and a file under `../benchmarks/`.
4. Run it: `GLM_LANE=dcp4-cc256 ./glm-serve.sh start` — then **deep-fill it** before trusting the width.

No new wrapper, no new entrypoint — the outer two layers are config-agnostic.
