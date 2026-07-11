# GLM-5.2 (unpruned QuantTrio) at 327K context on 4× DGX Spark — DCP2 recipe, validated on the new firmware (driver 580.159.03)

A complete, follower-replicable deployment of **GLM-5.2** — the unpruned QuantTrio
`GLM-5.2-Int4-Int8Mix` checkpoint, all 256 experts — across **4× NVIDIA DGX Spark
(GB10)** at **327,680-token context** via TP4 + DCP2, with MTP speculative decode
and fp8 sparse-MLA KV. This reproduces
[CosmicRaisins' DCP2-320K recipe](https://github.com/CosmicRaisins/glm-5.2-gb10)
and documents everything needed to run it on the **current DGX Spark firmware
(driver 580.159.03)** — which no published recipe covers yet, and which breaks
the stack in one specific, fixable way (below).

**Measured (llama-benchy, recipe methodology — pp2048/tg512, 3 runs, coherent corpus):**

| test | t/s | peak t/s |
|---|---|---|
| pp2048 | 355–364 | |
| tg512 | 12.8–13.9 | 20.0–22.7 |
| pp2048 @ d8192 | 326 | |
| tg512 @ d8192 | 13.0–13.6 | 19.7–21.3 |
| pp2048 @ d32768 | 315 | |
| tg512 @ d32768 | 12.7–12.9 | 18.3–20.7 |

Flat-to-depth holds (−8% decode d0→32K); a manual 156K-deep request decoded at
14.7 t/s with correct long-range retrieval. MTP acceptance 51–59% (mean accepted
length 2.8–3.05). Prefill is chunk-limited by a deliberate memory trade
(see Gotcha 2). Mean decode runs below the reference tables (~13 vs ~22) with
peaks matching — under investigation with the community; every infrastructure
suspect (NCCL transport, rails, channels, MTP k) has been measured and
eliminated (see forum thread).

## THE FIRMWARE LANDMINE (read this first)

On driver **580.159.03**, every b12x CuteDSL kernel fails to compile under
`nvidia-cutlass-dsl==4.5.2` (what all pre-firmware recipes implicitly use):

```
ValueError: Operation creation failed
  ...nvidia_cutlass_dsl/.../_cute_nvgpu_ops_gen.py: atom_tma_partition
  via b12x/cute/compiler.py -> cutlass_dsl/cutlass.py: launch
```

Fix — bake into your image:

```bash
pip install 'nvidia-cutlass-dsl==4.5.3' 'nvidia-cutlass-dsl-libs-cu13==4.5.3'
```

Verified on: driver 580.159.03, ptxas 13.0, torch 2.11.0+cu130.

## Stack

- **vLLM:** [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm)
  branch `codex/dcp-globaltopk-sharddraft-defaults-20260622` @ `e232d26`,
  plus the three patches vendored in CosmicRaisins' repo (`pr72-1`, `pr72-2`,
  `draft-quant-packed-mapping`) — apply with `patch -p1` against the installed
  tree; all apply cleanly at that ref.
- **b12x:** `pip install --no-deps 'git+https://github.com/lukealonso/b12x@9cd63a7'`
- **cutlass-dsl:** `==4.5.3` (see landmine above)
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
2. **`--max-num-batched-tokens 2048`, not the recipe's 4096.** Deep prefills
   (150K+) transiently need more activation headroom than the head node has at
   gmu 0.90; 4096-token chunks tripped the watchdog. Halving the chunk halves
   the transient — the cost is prefill rate (~355 vs ~600 t/s). If your nodes
   run leaner than ours, try 4096 first.
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
