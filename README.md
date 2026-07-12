# GLM-5.2 @ 327K on 4× DGX Spark

**Serve the unpruned [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) (QuantTrio Int4-Int8Mix, all 256 experts) at 327,680-token context across four GB10 Sparks — TP4 + DCP2, MTP speculative decode, fp8 sparse-MLA KV.**

![context](https://img.shields.io/badge/context-327K-1f6feb)
![hardware](https://img.shields.io/badge/hardware-4×_DGX_Spark_(GB10)-76b900)
![decode](https://img.shields.io/badge/decode-~14_tok%2Fs_·_20_peak-blue)
![prefill](https://img.shields.io/badge/prefill-~365_tok%2Fs-blue)
![firmware](https://img.shields.io/badge/driver-580.159.03_✓-orange)
![license](https://img.shields.io/badge/license-Apache_2.0-lightgrey)

A follower-replicable deployment reproducing [CosmicRaisins' DCP2-320K recipe](https://github.com/CosmicRaisins/glm-5.2-gb10) — and the first to document running it on the **current DGX Spark firmware** (driver 580.159.03), which breaks the stack in one specific, fixable way.

> [!WARNING]
> **On the new firmware, b12x kernels won't compile under `nvidia-cutlass-dsl==4.5.2`** (`ValueError: Operation creation failed`). Pin **4.5.3** — see [Build](#stack--build). If you're about to take the firmware update, do this first and save yourself the debugging session.

### What you get

- 🧠 **Full model, no pruning** — all 256 experts, at 327K context on 4 nodes.
- 📏 **Flat to depth** — decode holds within ~8% from 0 → 32K, verified coherent at a 156K-deep retrieval.
- 🔁 **Two lanes, one script** — `GLM_LANE=dcp2` for 327K, or the upstream-vLLM fallback at 200K/~25 t/s.
- 🛡️ **Battle-tested ops** — a unified-memory watchdog (GB10 OOM = frozen box otherwise), auto-reapplied lossless-RoCE fabric config, per-node RoCE GID auto-detect.

### Benchmarks

llama-benchy, recipe methodology (3 runs, coherent corpus; final config — gmu 0.89, lossless RoCE, 2026-07-12):

| test | t/s | peak t/s |
|---|---|---|
| pp2048 | **365–367** | |
| tg64 | 14.3 ± 0.8 | 19.7 |
| tg512 | 12.2 ± 0.6 | 20.0 |
| pp2048 @ d32768 | 318 | |
| tg64 @ d32768 | 12.7 | 18.0 |

MTP acceptance 51–59% (mean accepted length 2.8–3.05). Prefill is chunk-limited by a deliberate memory trade ([Gotcha 2](#gotchas-all-learned-the-hard-way)).

<details>
<summary><b>Open question: mean decode ~12–14 vs the reference ~22 (peaks match)</b></summary>

Every infrastructure suspect has been measured and eliminated: NCCL on RDMA verbs (not TCP), all 8 rails at full 200G with zero PHY errors, single-vs-dual rail, NCHANNELS 4/8, MTP k=3/4, and a hardware-verified lossless fabric ([below](#lossless-roce-ecn--pfc--optional-worth-it)). The residual sits in per-step kernel time on the driver 580.159.03 + cutlass-dsl 4.5.3 combo — a pairing no other cluster runs yet. Discussion in [forum thread 374125](https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125).
</details>

## Stack & build

- **vLLM:** [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm)
  branch `codex/dcp-globaltopk-sharddraft-defaults-20260622` @ `e232d26`,
  plus the three patches vendored in CosmicRaisins' repo (`pr72-1`, `pr72-2`,
  `draft-quant-packed-mapping`) — apply with `patch -p1` against the installed
  tree; all apply cleanly at that ref.
- **b12x:** `pip install --no-deps 'git+https://github.com/lukealonso/b12x@9cd63a7'`
- **cutlass-dsl:** `pip install 'nvidia-cutlass-dsl==4.5.3' 'nvidia-cutlass-dsl-libs-cu13==4.5.3'`

> [!WARNING]
> **The firmware landmine.** On driver **580.159.03**, every b12x CuteDSL kernel fails to compile under `nvidia-cutlass-dsl==4.5.2` (what pre-firmware recipes implicitly use):
> ```
> ValueError: Operation creation failed
>   ...nvidia_cutlass_dsl/.../_cute_nvgpu_ops_gen.py: atom_tma_partition
>   via b12x/cute/compiler.py -> cutlass_dsl/cutlass.py: launch
> ```
> Pinning **4.5.3** (above) fixes it. Verified on: driver 580.159.03, ptxas 13.0, torch 2.11.0+cu130.

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

`scripts/glm-launch.sh` (adapted from tonyd2wild's launcher, itself from
CosmicRaisins') + `scripts/start-glm-5.2.sh` wrapper. Edit the marked
CONFIG block (node IPs, user). Lanes via `GLM_LANE`:

```bash
GLM_LANE=dcp2 ./start-glm-5.2.sh start    # this recipe: 327K, DCP2, fork stack
./start-glm-5.2.sh start                  # fallback lane: 200K, upstream vLLM, ~25 t/s
```

Key serve flags for the dcp2 lane (full command in the script):

```
--attention-backend B12X_MLA_SPARSE
--decode-context-parallel-size 2 --dcp-kv-cache-interleave-size 1
--max-model-len 327680 --max-num-seqs 1 --max-num-batched-tokens 2048
--gpu-memory-utilization 0.90 --kv-cache-dtype fp8_ds_mla
--speculative-config '{"model":"<weights>","method":"mtp","quantization":"compressed-tensors","draft_attention_backend":"B12X_MLA_SPARSE","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
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

1. **Memory watchdog (scripts/glm-memwatch.sh).** GB10 unified memory means a
   runaway GPU allocation starves the whole OS — sshd included; our first
   uncontrolled OOM hard-froze all four nodes (power-cycle recovery). The
   watchdog `docker kill`s the vLLM container when MemAvailable < 1.5 GiB
   (1 s poll). Every OOM since has been a clean container kill. The start
   wrapper arms it automatically for DCP lanes.
2. **`--max-num-batched-tokens 2048` (recipe: 4096) and `gmu 0.89` (recipe:
   0.90).** Deep prefills (150K+) and long tg benches transiently need more
   activation headroom than the head node has; 4096 chunks and gmu 0.90 both
   tripped the 1.5 GiB watchdog. Halving the chunk halves the prefill transient
   (cost: prefill rate); gmu 0.89 buys ~1.1 GiB more permanent floor per node
   (cost: ~2% KV pool). Head now sits ~3.9 GiB idle. If your nodes run leaner,
   try the recipe's values first.
3. **RoCE GID index can differ per node** (ours did pre-firmware: 3/3/4/4).
   `scripts/glm52-entrypoint.sh` is bind-mounted and auto-detects the RoCEv2
   IPv4 GID at container start instead of hardcoding `NCCL_IB_GID_INDEX`.
4. **Do not run periodic drop_caches during weight load** (it starves
   read-ahead and can stall a rank into the 1800 s Gloo timeout, collapsing
   the cluster). One drop before launch is right.
5. **MTP k=3 vs k=4 is a wash** — we A/B'd on identical benchmarks: identical
   mean throughput, k=3 gets higher acceptance rate, k=4 slightly higher burst
   peaks. We run k=3 (matches CosmicRaisins' production).
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
to the RoCE traffic class), already set in `scripts/glm-launch.sh`. PFC on the
NICs is `mlnx_qos -i <dev> --trust dscp --pfc 0,0,0,1,0,0,0,0` on both rails per
node — needs root and isn't reboot-persistent, so `start-glm-5.2.sh` reapplies
it in preflight via a privileged container (`--privileged --network host --pid
host`; **`--pid host` is required** or the netlink bind fails "Address already
in use").

Payoff was honest: prefill throughput unchanged (drops were ~0.1% of volume),
but decode picked up ~+10% (queue prioritization trimming latency jitter on the
small sync ops) and the fabric is now clean instead of accidentally-working.

## Bonus: DCP on *upstream* vLLM (experiment)

Before adopting the fork we got DCP4 running on **upstream** vLLM
(`ab666069`, eugr build) with the stock `FLASHINFER_MLA_SPARSE_SM120` backend —
to our knowledge the first DCP run outside the fork. It requires the small
patch in `upstream-lse-experiment/sm120-return-lse.patch`: FlashInfer 0.6.14's
`trtllm_batch_decode_with_kv_cache_mla` already supports `return_lse` (which
DCP's cross-rank merge requires); upstream just never plumbed it through, and
the decode out-buffer must be sized from the all-gathered head count rather
than the TP-local one. It works (256K ctx validated) but decodes at ~12–13 t/s
because that backend's decode can't run under FULL CUDA graphs — the fork's
b12x path is the right answer today. The patch is included in case upstream
wants it; it's two small changes.

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
