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
| **`dcp2-cc200`** | 200K | c=3 | ~41 t/s aggregate · ~14 each · TTFT ~1.3 s | multi-user on DCP2 |
| **`dcp4`** | 655K | c=1 | ~24 t/s coherent | max-context specialty (DCP4's 4× KV) |
| **`dcp4-cc200`** | 200K | up to c=5 | ~47 t/s aggregate · ~10.5 each @ c=5 · gmu 0.88 · deep-fill memory-safe | wide multi-user on DCP4 |
| **`dcp4-cc128`** | 128K | up to c=8 | not yet benchmarked | max-width on DCP4 |

Full per-lane numbers + reproduce commands: [`../benchmarks/`](../benchmarks/README.md).

All lanes run the identical fork stack (image, b12x, MTP k=4, mesh-NCCL, `index_topk_pattern`, `clear_thinking`) — they differ **only** in `max-model-len` / `max-num-seqs` / `max-num-batched-tokens` / cudagraph capture / DCP size / **gpu-memory-utilization**. DCP shards the KV, so it's really one budget you spend on *context* (dcp4 → 655K single-stream) or *width* (dcp4-cc200 → multiple 200K streams).

### Concurrency lanes: gmu, and the two deep-fill failure modes

The `-cc` lanes carry two deliberate deltas from the single-stream lanes, both set per-lane in the `case`:

- **`L_GMU=0.88`** (single-stream lanes use 0.89). Five concurrent *deep* prefills hold ~1 GiB more rank-0 working set than one stream; at 0.89 that breached the head's 1.5 GiB watchdog. `0.88` frees ~1.1 GiB/node and clears it (costs ~2% KV). Override at runtime with `GLM_GMU=…`.
- **`L_BATCHED=2048`** (not 4096). Small prefill chunks *interleave* one stream's big prefill with everyone else's decode, so a user pasting a 50K doc dips the others instead of starving them.

A concurrency lane's `max-num-seqs` is the *shallow* ceiling. **Always deep-fill it before trusting its width**, and read the failure by its signature:

| symptom | cause | lever |
|---|---|---|
| head watchdog-kills **during prefill ramp**, KV usage still ~0% | rank-0 prefill working set (scales with streams × depth) | **lower `L_GMU`** (0.88 → 0.87…) — *not* `L_BATCHED`, *not* context |
| **preemptions > 0** + per-stream t/s sags mid-decode | KV-pool capacity — resident set exceeds the pool | fewer / shorter streams (e.g. `dcp4-cc128`) |

**Don't be scared off by a cold-fill TTFT.** Filling all streams to max depth *cold and simultaneously* (0% prefix cache) gives a pathological TTFT (tens of minutes for 5×197K) — that's a stress artifact, not the serving speed. Real multi-turn agents **amortize via prefix cache**: each turn only re-prefills its *delta* (new message + tool results), the resident history is a cache hit, so per-turn TTFT stays small. `clear_thinking=false` + `--enable-prefix-caching` (already in every lane) are what keep that prefix stable. Usable ceiling ≈ *streams whose combined resident KV fits the pool* (dcp4-cc200: ~5 × ≤197K ≈ 89% of its ~1.1M-token pool).

```bash
GLM_LANE=dcp2 ./glm-serve.sh start        # 327K flagship
./glm-serve.sh start                      # default = dcp2 (flagship)
GLM_LANE=dcp4-cc200 ./glm-serve.sh start  # wide multi-user
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
