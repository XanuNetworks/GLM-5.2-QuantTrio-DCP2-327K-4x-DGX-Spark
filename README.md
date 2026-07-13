# GLM-5.2 (unpruned) on 4× DGX Spark — depth, max context, or multi-user

**Serve the unpruned [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) (QuantTrio Int4-Int8Mix, all 256 experts) across four GB10 Sparks — one recipe, four lanes, one KV budget spent on **depth or width**. TP4 + DCP + MTP speculative decode + fp8 sparse-MLA KV, tuned for the *current* GB10 firmware.**

![context](https://img.shields.io/badge/context-up_to_655K-1f6feb)
![concurrency](https://img.shields.io/badge/multi--user-up_to_5_agents-1f6feb)
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

### Lanes — pick one with `GLM_LANE`

| lane | context | streams | role |
|---|---|---|---|
| **`dcp2`** *(default)* | 327K | 1 | 🏆 flagship — max depth + speed (~25 tok/s coherent) |
| **`dcp4`** | 655K | 1 | max context |
| **`dcp4-cc200`** | 200K | 3 | multi-user |
| **`dcp4-cc128`** | 128K | 5 | multi-user, most agents |

All four run the identical fork stack — they differ only in how the one KV budget is split: **depth** (`dcp2`/`dcp4`) or **width** (the `-cc` lanes). Lane map: [scripts/README](scripts/README.md) · per-lane numbers: [benchmarks/](benchmarks/README.md).

### What you get

- 🧠 **Full model, no pruning** — all 256 experts on 4 nodes, from 128K per stream up to 655K single-user context depending on lane.
- 📏 **Holds to depth** — `dcp4` keeps ~90% of its shallow decode even at **120K context** (21.6 t/s, DCP4's attention sharding); the 327K flagship holds ~80% at 32K.
- 🔁 **One recipe, four lanes** — the same fork stack serves single-user *depth* (`dcp2`/`dcp4`) or multi-user *width* (the `-cc` lanes); switch with one env var (see the table above). Each lane is tuned so its streams fit the KV pool with no preemption.
- 🛡️ **Sane ops** — auto-reapplied lossless-RoCE fabric config, per-node RoCE GID auto-detect, a GPU clock-health preflight, and an optional unified-memory watchdog for tighter configs.

### Benchmarks

Flagship `dcp2`, single-stream, all four GPUs at full clock (llama-benchy via `tool-eval-bench`, healthy-cluster re-measure 2026-07-13):

| workload | tok/s |
|---|---|
| **Decode — coherent (real coding prompts)** | **~24.5** |
| Decode — shallow (~16K ctx) | 23.2 |
| Decode — @ 32K context depth | 18.6 |
| Prefill (cold, ~16K ingest) | ~714 |

MTP **k=4**. Single-stream (`max-num-seqs 1`) — one-user, max-speed, not a concurrent-serving config. The **`dcp4`** lane actually holds decode *flatter* to depth (21.6 t/s even at 120K — see [benchmarks/](benchmarks/README.md)). Reproduce with `tool-eval-bench --perf` or **`utils/benchmark.sh`**.

> 📏 **Prose beats random by ~10-15%.** Random-token benches (`--dataset-name random`) tank MTP acceptance — real coherent prompts generate predictable, structured output the draft head accepts far more often, so `tg512`-on-random *understates* real-world decode. The **~24.5 coherent** figure is what you'll actually see; the ~21 random floor is pessimistic. (Watch out for prefix-cache pollution too — repeated random prompts at a fixed seed cache-hit and report fake-fast prefill.)

> 🕵️ **How we got from ~15 → ~24:** it wasn't cutlass, the driver, or NCCL (we suspected all three). One node's GPU was silently wedged at a **quarter of its clock speed**, and in a synchronized TP cluster the slowest GPU gates all four. If your numbers come in low, read [Is your decode slow?](docs/troubleshooting.md#is-your-decode-slow) before you blame the stack.

### Tool calling

Scored with [Tool Eval Bench](https://github.com/SeraphimSerapis/tool-eval-bench) — 69 scenarios, 14 categories:

**87 / 100 → ★★★★ Good** · **zero safety warnings** · consistent **87–89 across all four lanes** (tool-calling is model/parser-level, so it's the same regardless of context/concurrency config).

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
./glm-serve.sh start                        # default lane = dcp2: 327K, single-user, max-speed
GLM_LANE=dcp4-cc200 ./glm-serve.sh start    # multi-user on DCP4: 3×200K, tuned to fit the pool
GLM_LANE=dcp4 ./glm-serve.sh start          # max context: 655K, single-user (~24 t/s coherent)
GLM_LANE=dcp4-cc128 ./glm-serve.sh start    # multi-user on DCP4: 5×128K, tuned to fit the pool
```

Key serve flags for the dcp2 lane (full command in the script):

```
--attention-backend B12X_MLA_SPARSE
--decode-context-parallel-size 2 --dcp-kv-cache-interleave-size 1
--max-model-len 327680 --max-num-seqs 1 --max-num-batched-tokens 2048
--gpu-memory-utilization 0.89 --kv-cache-dtype fp8_ds_mla   # 0.88 on the -cc lanes
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

### Multi-user: the `-cc` lanes

Serving several agents at once? Two `-cc` lanes trade per-stream context for width, both on
DCP4: **`dcp4-cc200`** (3×200K) and **`dcp4-cc128`** (5×128K). Each is tuned so its streams fit
the KV pool with **no preemption** and the head node stays safe — just pick one and run it.

**Adjusting concurrency.** `GLM_SEQS` raises or lowers the stream count — it's a *default*, not
a hard cap:

```bash
GLM_LANE=dcp4-cc128 GLM_SEQS=6 ./glm-serve.sh start   # 6 agents instead of the default 5
```

The implications: **more streams = higher total throughput but slower per stream**; and if
every stream happens to fill to full context at once you may see brief **graceful stalling**
(one stream re-prefills), never a crash. The single thing to avoid is a burst of many *cold,
deep* prefills at a high `GLM_SEQS` — that's the boundary each lane's default sits below.
Why, and how to tune it safely: [concurrency-lane tuning](docs/troubleshooting.md#concurrency-lane-tuning)
· per-lane numbers: [benchmarks/](benchmarks/README.md).

## Troubleshooting & gotchas

Slow decode, a failed boot, or a fabric issue? The hard-won stuff — the **stuck-GPU-clock** check (the #1 cause of slow decode), the head-node memory-floor tuning, per-node bind-mount traps, MTP-under-DCP notes, and RoCE — lives in **[docs/troubleshooting.md](docs/troubleshooting.md)**.

The one-line version: **if decode is slow, burn-check the GPU clocks (`utils/check-clocks.sh`) before blaming the stack** — a single GB10 wedged at a quarter-clock silently gates the whole TP cluster, and only a cold power cycle clears it.

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
