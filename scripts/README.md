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

`glm-memwatch.sh` is a 4th helper — the optional unified-memory watchdog that
`glm-serve.sh` arms on each node (see the main README).

## Lanes (config variants live HERE, not in new scripts)

Config differences are **lanes**, selected with `GLM_LANE`, defined inside
`glm-node-launch.sh`. Do **not** clone `glm-serve.sh` per config.

| `GLM_LANE` | ctx | stack | notes |
|---|---|---|---|
| `dcp2` (this recipe) | 327K | fork vLLM + b12x, DCP2, MTP k=4 | the flagship |
| `fast` (default) | 200K | upstream vLLM | proven daily driver, ~25 t/s |
| `dcp` (experimental) | 256K | upstream vLLM + LSE patch | slower (~12 t/s); the "Bonus" path |

```bash
GLM_LANE=dcp2 ./glm-serve.sh start     # 327K flagship
./glm-serve.sh start                   # default = fast lane
GLM_LANE=dcp2 ./glm-serve.sh dry-run   # print docker commands, run nothing
./glm-serve.sh stop                    # tear down on all nodes
```

## How to add a lane (e.g. a concurrency-tuned config)

Everything is data — add a block, don't add a file:

1. In `glm-node-launch.sh`, extend the `GLM_LANE` selector with your new value
   (e.g. `concurrent`): set the lane vars (`LANE_MAXLEN`, `LANE_SEQS`,
   `LANE_KVBYTES`, `LANE_COMPCFG`, `DCP_FLAGS`, …). For a multi-user config that
   means a higher `--max-num-seqs` and more KV budget (trade context or headroom
   for concurrency — the base recipe is `max-num-seqs 1`, single-user max-speed).
2. If the lane needs different env/mounts/image, add a `if [ "$GLM_LANE" = "concurrent" ]`
   override block near the `dcp2` one.
3. Add a row to the lane table above.
4. Run it: `GLM_LANE=concurrent ./glm-serve.sh start`.

No new wrapper, no new entrypoint — the outer two layers are config-agnostic.
