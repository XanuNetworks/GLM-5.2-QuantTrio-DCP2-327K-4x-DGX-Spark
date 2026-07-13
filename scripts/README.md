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
| **`dcp4-cc200`** | 200K | up to c=5 | ~47 t/s aggregate · ~10.5 each @ c=5 *(shallow)* | wide multi-user on DCP4 |
| **`dcp4-cc128`** | 128K | up to c=8 | not yet benchmarked | max-width on DCP4 |

Full per-lane numbers + reproduce commands: [`../benchmarks/`](../benchmarks/README.md). **Deep-fill caveat:** the `-cc` lanes' `max-num-seqs` is the *shallow* ceiling; concurrent **deep** streams cost rank-0 prefill working set, so the usable deep-concurrent count is lower than the shallow ceiling (see [`benchmarks/dcp4-cc200.md`](../benchmarks/dcp4-cc200.md) — always deep-fill a concurrency lane before trusting its width).

All lanes run the identical fork stack (image, b12x, MTP k=4, mesh-NCCL, `index_topk_pattern`, `clear_thinking`) — they differ **only** in `max-model-len` / `max-num-seqs` / cudagraph capture / DCP size. DCP shards the KV, so it's really one budget you spend on *context* (dcp4 → 655K single-stream) or *width* (dcp4-cc200 → multiple 200K streams).

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
