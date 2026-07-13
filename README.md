# GLM-5.2 @ 327K on 4× DGX Spark

**Serve the unpruned [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) (QuantTrio Int4-Int8Mix, all 256 experts) at 327,680-token context across four GB10 Sparks — TP4 + DCP2, MTP speculative decode, fp8 sparse-MLA KV.**

![context](https://img.shields.io/badge/context-327K-1f6feb)
![hardware](https://img.shields.io/badge/hardware-4×_DGX_Spark_(GB10)-76b900)
![decode](https://img.shields.io/badge/decode-~25_tok%2Fs_coherent-brightgreen)
![prefill](https://img.shields.io/badge/prefill-~720_tok%2Fs-blue)
![tool--calling](https://img.shields.io/badge/tool--calling-88%2F100_★★★★-brightgreen)
![driver](https://img.shields.io/badge/driver-580.159.03-brightgreen)
![cutlass](https://img.shields.io/badge/cutlass--dsl-4.6.0-blue)
![license](https://img.shields.io/badge/license-Apache_2.0-lightgrey)

A follower-replicable deployment of [CosmicRaisins' DCP2-320K recipe](https://github.com/CosmicRaisins/glm-5.2-gb10), running a **deliberately newer stack than the reference recipes were written against** — the *current* DGX Spark firmware (driver **580.159.03**) with **cutlass-dsl 4.6.0**, **FlashInfer 0.6.15**, and **torch 2.11.0**. Same unpruned model, same 327K context, **~25 tok/s** coherent decode matching the reference cluster — just on the firmware everyone else hasn't updated to yet (with the one landmine it hides, defused below).

> [!WARNING]
> **Taking the firmware update? Do this first.** On driver 580.159.03, b12x kernels won't compile under `nvidia-cutlass-dsl==4.5.2` (`ValueError: Operation creation failed`) — the version pre-firmware recipes implicitly use. Pin **≥ 4.5.3** (we run 4.6.0). See [Stack](#stack--build).

### What you get

- 🧠 **Full model, no pruning** — all 256 experts, at 327K context on 4 nodes.
- 📏 **Flat to depth** — decode holds within ~8% from 0 → 32K, verified coherent at a 156K-deep retrieval.
- 🔁 **Two lanes, one script** — `GLM_LANE=dcp2` for the tested 327K recipe, or a simpler `fast` fallback (200K, upstream vLLM) for anyone not running the fork.
- 🛡️ **Sane ops** — auto-reapplied lossless-RoCE fabric config, per-node RoCE GID auto-detect, a GPU clock-health preflight, and an optional unified-memory watchdog for tighter configs.

### Benchmarks

Single-stream (matches llama-benchy methodology), all four GPUs at full clock:

| workload | tok/s |
|---|---|
| **Decode — coherent (real agentic coding/reasoning, mean of 5)** | **~25** (21–29) |
| Decode — random-token floor (`tg512`, worst case) | ~23 |
| Decode — @ 32K context depth | ~20 |
| Prefill (cold, ~12–17K-token ingest) | ~720 |

Matches CosmicRaisins' reference (~28 agentic). MTP **k=4**, mean acceptance length ~3.0. Single-stream (`max-num-seqs 1`) — this is a one-user, max-speed config, not a concurrent-serving one. Reproduce this table with **`utils/benchmark.sh`** (coherent + random floor + cold prefill).

> 📏 **Prose beats random by ~10-15%.** Random-token benches (`--dataset-name random`) tank MTP acceptance — real coherent prompts generate predictable, structured output the draft head accepts far more often, so `tg512`-on-random *understates* real-world decode. The **~25 coherent** figure is what you'll actually see; the ~23 is a pessimistic floor. (Watch out for prefix-cache pollution too — repeated random prompts at a fixed seed cache-hit and report fake-fast prefill.)

> 🕵️ **How we got from ~15 → ~25:** it wasn't cutlass, the driver, or NCCL (we suspected all three). One node's GPU was silently wedged at a **quarter of its clock speed**, and in a synchronized TP cluster the slowest GPU gates all four. If your numbers come in low, read [Is your decode slow?](#is-your-decode-slow) before you blame the stack.

### Tool calling

Scored with [Tool Eval Bench](https://github.com/SeraphimSerapis/tool-eval-bench) — 69 scenarios, 14 categories:

**88 / 100 → ★★★★ Good** · 54 pass / 13 partial / 2 fail · **zero safety warnings**.

Perfect (100%) on the fundamentals — Tool Selection, Parameter Precision, Restraint & Refusal, Error Recovery, Structured Reasoning, Code Patterns, Autonomous Planning. It softens only on the hard agentic edges: **Toolset Scale** 62% (disambiguating among many offered tools), nested **Structured Output** 75%, and long-horizon **Context & State** 75%. A "senior engineer" shape — rock-solid core, honest about the corners.

## Stack & build

- **vLLM:** [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm)
  branch `codex/dcp-globaltopk-sharddraft-defaults-20260622` @ `e232d26`,
  plus the three patches vendored in CosmicRaisins' repo (`pr72-1`, `pr72-2`,
  `draft-quant-packed-mapping`) — apply with `patch -p1` against the installed
  tree; all apply cleanly at that ref.
- **b12x:** `pip install --no-deps 'git+https://github.com/lukealonso/b12x@9cd63a7'`
- **cutlass-dsl:** `pip install 'nvidia-cutlass-dsl==4.6.0' 'nvidia-cutlass-dsl-libs-cu13==4.6.0'` — anything **≥ 4.5.3** works on the new firmware; **4.5.2 does not** (see below). 4.5.3 and 4.6.0 measure identically.

> [!WARNING]
> **The firmware landmine.** On driver **580.159.03**, every b12x CuteDSL kernel fails to compile under `nvidia-cutlass-dsl==4.5.2` (what pre-firmware recipes implicitly use):
> ```
> ValueError: Operation creation failed
>   ...nvidia_cutlass_dsl/.../_cute_nvgpu_ops_gen.py: atom_tma_partition
>   via b12x/cute/compiler.py -> cutlass_dsl/cutlass.py: launch
> ```
> Pinning **≥ 4.5.3** (we run **4.6.0**) fixes it. Verified on driver 580.159.03, CUDA 13.0, torch 2.11.0+cu130, FlashInfer 0.6.15.

- **Image build:** [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
  harness. Three temporary Dockerfile edits are needed to build a fork ref
  (revert after building):
  1. point the vLLM clone URL (and, on cache hit, `git remote set-url origin`)
     at the fork;
  2. default `ARG VLLM_APPLY_PRESET_PRS="0"` (the preset upstream PRs
     merge-conflict against the June fork tree);
  3. make the inline `RUN patch -p1` heredocs tolerant
     (`patch -p1 --forward ... || echo skipped`) — they target upstream main
     and don't all apply to the fork.
  Then: `./build-and-copy.sh --vllm-ref e232d26 -t <your-tag>`
- **Extra patch (this repo):** `patches/b12x-prewarm-tolerance.patch` — wraps
  the b12x profile-run prewarm variants in try/except (the pattern from
  tonyd2wild's NF3 notes). With cutlass 4.5.3 the prewarm succeeds anyway;
  kept as insurance since a failed precompile should never kill a boot.

## Weights

```bash
hf download QuantTrio/GLM-5.2-Int4-Int8Mix --local-dir /var/tmp/models/glm52-int4-int8mix
# rsync to every node, then on each node:
mkdir -p /var/tmp/models/hub && ln -sfn ../glm52-int4-int8mix /var/tmp/models/hub/glm52-int4-int8mix
```

Also stage NCCL 2.30.4's `libnccl.so.2` at `/var/tmp/models/hub/nccl-2.30.4/`
(extract from the `nvidia-nccl-cu13==2.30.4` aarch64 wheel) — it is LD_PRELOADed.

## Launch

`scripts/glm-node-launch.sh` (adapted from tonyd2wild's launcher, itself from
CosmicRaisins') + `scripts/glm-serve.sh` wrapper. Edit the marked
CONFIG block (node IPs, user). Lanes via `GLM_LANE`:

```bash
GLM_LANE=dcp2 ./glm-serve.sh start    # this recipe: 327K, DCP2, fork stack
./glm-serve.sh start                  # fallback lane: 200K, upstream vLLM (not benchmarked recently)
```

Key serve flags for the dcp2 lane (full command in the script):

```
--attention-backend B12X_MLA_SPARSE
--decode-context-parallel-size 2 --dcp-kv-cache-interleave-size 1
--max-model-len 327680 --max-num-seqs 1 --max-num-batched-tokens 2048
--gpu-memory-utilization 0.90 --kv-cache-dtype fp8_ds_mla
--speculative-config '{"model":"<weights>","method":"mtp","quantization":"compressed-tensors","draft_attention_backend":"B12X_MLA_SPARSE","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
--hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
--default-chat-template-kwargs '{"clear_thinking":false}'
--compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":10}'
env: VLLM_USE_B12X_SPARSE_INDEXER=1 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_DCP_GLOBAL_TOPK=1 VLLM_DCP_SHARD_DRAFT=1
```

Two flags deserve emphasis (both CosmicRaisins' findings, reproduced here):

- **`index_topk_pattern` is not optional.** GLM-5.2 trains DSA indexer weights
  on only 21/78 layers; QuantTrio ships `index_topk_pattern: null` and vLLM
  then top-k's through uninitialized weights on the other 57 — coherent under
  ~2K tokens, degrading beyond, MTP acceptance collapsing at depth (we measured
  per-position acceptance 0.88/0.46/0.18/0.00 at 120K without it). Derive the
  F/S string from your checkpoint's `indexer_types` (`full`→F, `shared`→S).
- **`clear_thinking:false`** keeps prior turns' thinking in history so the
  prompt prefix stays stable across turns → prefix-cache hits instead of
  re-prefilling the whole conversation every message.

## Gotchas (all learned the hard way)

1. **Optional memory watchdog (scripts/glm-memwatch.sh).** GB10 unified memory
   means a runaway GPU allocation can starve the whole OS — sshd included; an
   early uncontrolled OOM (a 512K autotune, pre-firmware) hard-froze all four
   nodes. As insurance, the watchdog `docker kill`s the vLLM container when
   MemAvailable < 1.5 GiB (1 s poll), and the start wrapper arms it for DCP
   lanes. Full honesty: in the final config (gmu 0.89) it has **never actually
   fired** — the head node sits ~3.7 GiB above the floor. It only earns its keep
   if you run the *tighter* values in Gotcha 2 (gmu 0.90 / 4096 chunks), where
   the floor genuinely gets breached. Harmless to keep, but don't mistake it for
   proven — its behavior against a *real* freeze is untested.
2. **`--max-num-batched-tokens 2048` (recipe: 4096) and `gmu 0.89` (recipe:
   0.90).** Deep prefills (150K+) and long tg benches transiently need more
   activation headroom than the head node has; 4096 chunks and gmu 0.90 both
   drove the head node's MemAvailable down near/through the ~1.5 GiB floor
   (a failed boot/bench). Halving the chunk halves the prefill transient
   (cost: prefill rate); gmu 0.89 buys ~1.1 GiB more permanent floor per node
   (cost: ~2% KV pool). Head now sits ~3.9 GiB idle. If your nodes run leaner,
   try the recipe's values first.
3. **RoCE GID index can differ per node** (ours did pre-firmware: 3/3/4/4).
   `scripts/glm-container-entrypoint.sh` is bind-mounted and auto-detects the RoCEv2
   IPv4 GID at container start instead of hardcoding `NCCL_IB_GID_INDEX`.
4. **Do not run periodic drop_caches during weight load** (it starves
   read-ahead and can stall a rank into the 1800 s Gloo timeout, collapsing
   the cluster). One drop before launch is right.
5. **MTP k=4** (CosmicRaisins' recipe value). On *random-token* benches k=3/k=4
   are a wash — but on real coherent prompts the 4th draft token accepts often
   enough to help, so k=4 wins where it matters. Requires the PR#72 draft-under-
   DCP patches (see Stack) or k>1 acceptance collapses under DCP. `draft_tp` is
   locked at the target TP (=4) under DCP2 — "draft tp=1" is a non-DCP trick.
6. **`docker commit --change 'ENTRYPOINT ...'` quoting**: `\"` inside the
   change value silently produces a broken `/bin/sh -c` entrypoint. Verify
   with `docker inspect --format '{{.Config.Entrypoint}}'` after committing.

## Lossless RoCE (ECN + PFC) — optional, worth it

DCP's per-step syncs are small and latency-bound, so decode never loses packets
— but big prefill all-gathers overrun a plain L2 switch and eat go-back-N
retransmits. If your switch supports RoCE QoS (ECN/DCQCN + PFC on a dedicated
traffic class), configuring it per your vendor's lossless-RoCE guide is worth
it. Verify with the NIC hardware counters
(`/sys/class/infiniband/<hca>/ports/1/hw_counters/{packet_seq_err,out_of_sequence,rp_cnp_handled,np_cnp_sent}`,
diffed around an isolated ~78K-token prefill):

| stage | packet_seq_err / big prefill |
|---|---|
| raw (no QoS) | ~1,000–2,200 per sender |
| + ECN/DCQCN | ~90 (−95%) |
| + PFC | **0 (lossless)** |

The catch that cost us an afternoon: the switch QoS config does **nothing** on
its own — NCCL sends at DSCP 0 by default and sails past every classifier. The
missing key is node-side `NCCL_IB_TC=106` (→ DSCP 26, which the switch then maps
to the RoCE traffic class), already set in `scripts/glm-node-launch.sh`. PFC on the
NICs is `mlnx_qos -i <dev> --trust dscp --pfc 0,0,0,1,0,0,0,0` on both rails per
node — needs root and isn't reboot-persistent, so `glm-serve.sh` reapplies
it in preflight via a privileged container (`--privileged --network host --pid
host`; **`--pid host` is required** or the netlink bind fails "Address already
in use").

Payoff was honest: prefill throughput unchanged (drops were ~0.1% of volume),
but decode picked up ~+10% (queue prioritization trimming latency jitter on the
small sync ops) and the fabric is now clean instead of accidentally-working.

## Is your decode slow?

Check the boring stuff before blaming the stack. (We blamed the stack for a while. It wasn't the stack.)

- 🐌 **A GPU stuck at low clock — this is the big one.** A GB10 can silently wedge a single GPU at ~660 MHz: it *reports* `P0` but delivers a quarter-clock, stays cold, and draws ~17 W even under full load. In a synchronized TP cluster **the slowest GPU gates all four**, so one lame node quietly capped us at ~15 t/s for an entire session. Burn each GPU for 5 s and read `nvidia-smi --query-gpu=clocks.current.sm,power.draw` — healthy ≈ **2300–2500 MHz / ~90 W**, wedged ≈ **660 MHz / ~17 W**. A **warm reboot won't fix it** (the GPU firmware holds the wedge on standby power); you need a full **cold power cycle** — shut down, *pull the plug for ~30 s*, power back on. `glm-serve.sh` burn-checks every node in preflight and refuses to launch on a wedged one — or run it on demand: **`utils/check-clocks.sh`**.
- 🌐 **Then the fabric.** Slow or lossy prefill → [Lossless RoCE](#lossless-roce-ecn--pfc--optional-worth-it), and remember the switch QoS does nothing without node-side `NCCL_IB_TC`. Confirm NCCL is on RDMA verbs, not TCP.

## Credits

None of this is possible without:

- **Z.ai (Zhipu AI)** — GLM-5.2 itself. The in-checkpoint MTP layer and DSA
  sparse attention are why a 4-node cluster can serve an unpruned frontier
  MoE at 327K context at all.
- **QuantTrio** — the unpruned Int4-Int8Mix quantization.
- **CosmicRaisins** — the GLM-5.2 GB10 serving stack, the DCP recipes this
  repo reproduces, and the index_topk_pattern + clear_thinking findings.
- **tonyd2wild** — the QuantTrio recipe lineage, the launch harness, and the
  indexer MTP-overhang fix.
- **lukealonso** — b12x; **m9e / voipmonitor / Zatz** — the DCP vLLM branch
  and the long-context demonstrations; **ciprianveg / back199640 /
  aidendle94** — tuning finds and shared resources (NVIDIA GB10 forum,
  thread 374125); **eugr** — the spark-vllm-docker build harness;
  **the vLLM project** — the engine.

## License

Apache-2.0. See NOTICE for the full lineage — start with CosmicRaisins' and
tonyd2wild's repos; this one exists because theirs did.
