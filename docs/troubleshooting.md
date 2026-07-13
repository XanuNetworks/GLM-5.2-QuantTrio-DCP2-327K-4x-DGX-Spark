# Troubleshooting & gotchas

Hard-won lessons from bringing this up. If something's slow or a boot fails, start here.

## Is your decode slow?

Check the boring stuff before blaming the stack. (We blamed the stack for a while. It wasn't the stack.)

- 🐌 **A GPU stuck at low clock — this is the big one.** A GB10 can silently wedge a single GPU at ~660 MHz: it *reports* `P0` but delivers a quarter-clock, stays cold, and draws ~17 W even under full load. In a synchronized TP cluster **the slowest GPU gates all four**, so one lame node quietly capped us at ~15 t/s for an entire session. Burn each GPU for 5 s and read `nvidia-smi --query-gpu=clocks.current.sm,power.draw` — healthy ≈ **2300–2500 MHz / ~90 W**, wedged ≈ **660 MHz / ~17 W**. A **warm reboot won't fix it** (the GPU firmware holds the wedge on standby power); you need a full **cold power cycle** — shut down, *pull the plug for ~30 s*, power back on. `glm-serve.sh` burn-checks every node in preflight and refuses to launch on a wedged one — or run it on demand: **`utils/check-clocks.sh`**.
- 🌐 **Then the fabric.** Slow or lossy prefill → [Lossless RoCE](lossless-roce.md), and remember the switch QoS does nothing without node-side `NCCL_IB_TC`. Confirm NCCL is on RDMA verbs, not TCP.

## Gotchas (all learned the hard way)

> The optional unified-memory watchdog lives in [`../utils/`](../utils/README.md#glm-memwatchsh) — it's insurance infra, not part of the core recipe.

1. **The head node's ~1.5 GiB memory floor is the constraint that shapes the configs.** Deep prefills (150K+) and concurrent prefills transiently need more activation headroom than the head has. Two levers manage it: `--max-num-batched-tokens 2048` (recipe: 4096) halves the prefill transient, and `gpu-memory-utilization` buys permanent floor (~1.1 GiB/node per −0.01, cost ~2% KV pool). The single-stream lanes run gmu 0.89; the concurrency lanes are tuned per-lane (`dcp4-cc128`/`dcp4-cc200` at 0.885) to the point where their streams fit the KV pool *and* the head survives a cold-prefill burst. See [scripts/README](../scripts/README.md#concurrency-lanes-what-actually-controls-capacity).
2. **RoCE GID index can differ per node** (ours did pre-firmware: 3/3/4/4). `scripts/glm-container-entrypoint.sh` is bind-mounted and auto-detects the RoCEv2 IPv4 GID at container start instead of hardcoding `NCCL_IB_GID_INDEX`.
3. **Do not run periodic drop_caches during weight load** (it starves read-ahead and can stall a rank into the 1800 s Gloo timeout, collapsing the cluster). One drop before launch is right.
4. **MTP k=4** (CosmicRaisins' recipe value). On *random-token* benches k=3/k=4 are a wash — but on real coherent prompts the 4th draft token accepts often enough to help, so k=4 wins where it matters. Requires the PR#72 draft-under-DCP patches (see Stack) or k>1 acceptance collapses under DCP. `draft_tp` is locked at the target TP (=4) under DCP — "draft tp=1" is a non-DCP trick.
5. **Per-node bind-mounts must exist on every node.** The entrypoint is bind-mounted from each node's `$HOME/vllm/`; if it's missing on a worker, Docker silently creates an empty *directory* there and the container exits 126. Stage `glm-container-entrypoint.sh` (and `glm-memwatch.sh`) to **all** nodes.
6. **`docker commit --change 'ENTRYPOINT ...'` quoting**: `\"` inside the change value silently produces a broken `/bin/sh -c` entrypoint. Verify with `docker inspect --format '{{.Config.Entrypoint}}'` after committing.

## Concurrency-lane tuning

The `-cc` lane defaults are already tuned so their advertised streams fit the KV pool with no preemption *and* the head node survives a cold-prefill burst. You only need this section if you're changing `GLM_SEQS` (or curious why the defaults are where they are). Full physics: [scripts/README](../scripts/README.md#concurrency-lanes-what-actually-controls-capacity).

**Why the defaults use gmu 0.885.** The KV pool is set by `gpu-memory-utilization` alone, and it trades against head-node memory: higher gmu = bigger pool (fits more resident streams) but less head headroom. `dcp4-cc128` (5×128K=640K) is the tight case — **0.89** makes the pool fit but *kills the head* under a cold burst; **0.88** is head-safe but leaves the pool ~5% short (occasional preemption); **0.885** is the razor where 640K fits the 651K pool *and* the head holds (~1.8 GiB floor, validated).

**Don't judge a concurrency lane by a cold-fill benchmark.** Filling every stream to max depth *cold and simultaneously* (0% prefix cache) produces a pathological TTFT — tens of minutes for 5×197K — which is a stress artifact, not the serving speed. Real multi-turn agents **amortize via prefix cache**: each turn only re-prefills its *delta* (the resident history is a cache hit — that's what `clear_thinking:false` protects), so per-turn TTFT stays small as context grows. A user pasting a 50K doc pays ~1–2 min on *that one stream*, once, then it's cached.

**Raising `GLM_SEQS` past the default.** It's a default, not a cap — no enforced maximum. Raising it is usually safe: real traffic rarely has every stream at full context at once, so the pool usually isn't exhausted; and if it is, you get **graceful preemption** (a re-prefill stall on one stream), not a crash. The one hard edge: a burst of many *cold, deep* prefills at a high `GLM_SEQS` can breach the head watchdog and restart the container. That's the catastrophic axis — so the rule of thumb is **push concurrency for shallow/mixed-depth agents freely; be conservative if you expect many agents to hit full context cold at the same instant.** If you need more resident deep streams, the lever is a bigger pool: raise gmu (watch the head) or use a higher DCP degree, not more seqs (seqs doesn't size the pool).
