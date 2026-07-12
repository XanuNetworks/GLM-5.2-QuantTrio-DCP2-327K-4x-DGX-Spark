#!/usr/bin/env bash
#
# launch.sh — start GLM-5.2 (TP=4) across a 4-node GB10 / DGX Spark cluster.
#
# Derived from launch.sh in CosmicRaisins/glm-5.2-gb10
# (https://github.com/CosmicRaisins/glm-5.2-gb10), Copyright (c) CosmicRaisins,
# licensed under the Apache License, Version 2.0. Adaptations for the unpruned
# QuantTrio Int4-Int8Mix checkpoint at 200K context. See NOTICE.
#
# Self-contained: a plain `docker run` per node, no external harness. Multi-node
# is vLLM's NATIVE mechanism (--nnodes/--node-rank/--master-addr/--master-port);
# the workers (rank >= 1) start headless, then the head (rank 0) serves the API.
# There is NO Ray and NO shared-filesystem requirement — each node just needs the
# weights present locally (see WEIGHTS_DIR) and the kernels deployed to
# KERNELS_DIR (see README step "Kernels").
#
# Run from the HEAD node (NODES[0]). Workers are reached over key-based SSH.
#
#   ./launch.sh            # launch
#   ./launch.sh --dry-run  # print the docker commands without running them
#   ./launch.sh --stop     # docker rm -f the container on every node
#
# License: Apache-2.0.
set -uo pipefail

# ============================================================================
# CONFIG — edit these for your cluster
# ============================================================================
# EDIT: RoCE rail IPs, rank 0 (head) FIRST. Run this script from the head node.
# (RoCE fabric on a managed switch, MTU 9000.)
NODES=(192.168.NNN.1 192.168.NNN.2 192.168.NNN.3 192.168.NNN.4)

# EDIT: SSH key the head node uses to reach the workers (key-based, no prompt).
# On this cluster, developer's default ssh setup already reaches all four nodes,
# so leave SSH_KEY empty and skip the -i flag.
SSH_KEY=""

# EDIT: how to derive the SSH username for a given node IP. On this cluster,
# every node uses the developer user.
node_user() { echo "YOURUSER"; }

# NOTE: the -dcp image variant (probe-modded + CosmicRaisins draft-quant patch)
# is BROKEN — the patch fuzz-landed in load_weights() causing
# "NameError: quant_config is not defined" in the V2 spec-decode path.
# Our draft demonstrably loads quantized fine without it (MTP accept 2.83).
IMAGE="vllm-node-tf5-glm52-b12x:probe-modded"
NAME="vllm_glm52"                        # container name on every node
PORT=8000                                # matches Qwen/DeepSeek convention on this cluster
MASTER_PORT=29501                        # vLLM cross-node rendezvous port

# Weights. A directory THAT EXISTS ON EVERY NODE, holding the HF hub layout:
#   $WEIGHTS_DIR/hub/glm52-int4-int8mix       (QuantTrio Int4-Int8Mix weights,
#                                              symlink -> ../glm52-int4-int8mix)
#   $WEIGHTS_DIR/hub/nccl-2.30.4/libnccl.so.2 (LD_PRELOADed NCCL, see README)
# Mounted writable at /cache/huggingface (HF_HOME) so JIT/Triton caches stay local.
# How the weights get onto each node is YOUR choice — per-node copy, rsync, or a
# shared mount pointed here. This script does not assume any of them.
WEIGHTS_DIR="/var/tmp/models"

# Per-node directory holding the 10 Triton sparse-MLA kernel .py files from
# CosmicRaisins/glm-5.2-gb10 kernels/. Bound file-by-file over the vLLM tree,
# read-only.
KERNELS_DIR="$HOME/glm-triton"

# ---------------------------------------------------------------------------
# LANE SELECTOR — GLM_LANE=fast (default) or GLM_LANE=dcp
#
#   fast: the proven daily driver. 200K ctx, no DCP, FULL graphs, KV 10.95G,
#         25 tok/s decode, stable through long sessions.
#   dcp:  the 2026-07-10 context lane (11-boot bring-up). DCP4 shards KV
#         across ranks: 256K ctx (pool ~512K logical). Measured 12.6 tok/s
#         shallow / ~7.5 tok/s at 120K depth — decode attention still runs
#         eager (vLLM downgrades FULL for the SM120 sparse backend) and DCP
#         adds per-step ag_rs comm. Deep prefills (120K+) can still breach
#         the memory floor: arm watchdogs (start-glm-5.2.sh does) and treat
#         as experimental until upstream lands graph-capturable sparse decode.
# ---------------------------------------------------------------------------
GLM_LANE="${GLM_LANE:-fast}"
if [ "$GLM_LANE" = "dcp" ]; then
  DCP_FLAGS="--decode-context-parallel-size 4 --dcp-comm-backend ag_rs"
  LANE_MAXLEN=256000
  LANE_SEQS=4
  LANE_KERNELCFG='--kernel-config {"enable_flashinfer_autotune":false}'
  LANE_KVBYTES=7000000000
  LANE_COMPCFG='{"cudagraph_mode":"FULL","max_cudagraph_capture_size":32}'
else
  DCP_FLAGS=""
  LANE_MAXLEN=200000
  LANE_SEQS=6
  LANE_KERNELCFG=""
  LANE_KVBYTES=10950000000
  LANE_COMPCFG='{"cudagraph_mode":"FULL"}'
fi
# ============================================================================

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

DRYRUN=0; STOP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRYRUN=1 ;;
    --stop)    STOP=1 ;;
    *) die "unknown arg: $a (use --dry-run or --stop)" ;;
  esac
done

[ "${#NODES[@]}" -ge 1 ] || die "NODES is empty"
NNODES="${#NODES[@]}"
HEAD="${NODES[0]}"

if [ "$STOP" = 1 ]; then
  say "stopping '$NAME' on all ${NNODES} nodes"
  for ip in "${NODES[@]}"; do
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$(node_user "$ip")@$ip" "docker rm -f $NAME 2>/dev/null" \
      && printf '   stopped on %s\n' "$ip"
  done
  exit 0
fi

# ----------------------------------------------------------------------------
# Container env. The NCCL_IB_HCA / *_SOCKET_IFNAME values are RoCE-fabric-
# specific: set them for YOUR cluster (HCAs via `ibdev2netdev`, interfaces via
# `ip link`). Marked EDIT.
ENVV=(
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "LD_PRELOAD=/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2"
  -e "HF_HOME=/cache/huggingface"
  -e "TRITON_CACHE_DIR=/cache/huggingface/.tritoncache"
  -e "HF_HUB_OFFLINE=1"
  -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  -e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256"
  -e "GLM52_BIND_HOST_TRITON=1"
  -e "GLM52_MQA_LOGITS_TRITON=1"
  -e "GLM52_PAGED_MQA_TRITON=1"
  -e "GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192"
  -e "GLM52_B12X_MLA=1"
  -e "TORCH_CUDA_ARCH_LIST=12.1a"
  -e "NCCL_NET=IB"
  -e "NCCL_IB_DISABLE=0"
  -e "NCCL_IB_HCA=rocep1s0f0"          # EDIT: your RoCE HCA (`ibdev2netdev`)
  -e "NCCL_SOCKET_IFNAME=enp1s0f0np0"  # EDIT: your fabric interface (`ip link`)
  -e "GLOO_SOCKET_IFNAME=enp1s0f0np0"  # EDIT: same fabric interface
  # NCCL_IB_GID_INDEX is NOT set here — it differs per-node on this cluster
  # (.11/.12 use idx=3, .13/.14 use idx=4). Instead, the entrypoint wrapper
  # bind-mounted below auto-detects the RoCEv2 IPv4 GID at container start.
  # Bumped 4 -> 8 per forum #145 (Veghes): "50% increase at 2 parallel reqs
  # gen speed and a 2-3% increase at single request"
  -e "NCCL_MAX_NCHANNELS=8"
  -e "NCCL_MIN_NCHANNELS=8"
  -e "NCCL_CROSS_NIC=1"
  -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1"
  -e "NCCL_DEBUG=WARN"
  # DSCP 26 marking (IB TC 106 >> 2 = 26): lets the switch classify RoCE into
  # its ECN-marked queue (switch RoCE QoS). Counter evidence
  # 2026-07-12: prefill all-gathers drop packets on the lossy fabric
  # (packet_seq_err ~1K/sender per 40K-token prefill); decode is loss-free.
  -e "NCCL_IB_TC=106"
)
# V2 model runner: DCP lane ONLY. Needed there for the draft-under-DCP path
# (fork parity, thread 375416). On the fast lane it silently switches autotune
# to the V2 mixed dummy-run, which ground all 4 nodes down to ~1 GiB and wedged
# a previously-reliable boot (2026-07-10). Fast lane = V1 runner, as validated.
if [ "$GLM_LANE" = "dcp" ]; then
  ENVV+=(-e "VLLM_USE_V2_MODEL_RUNNER=1")
fi

# Triton sparse-MLA kernels, bound read-only over the vLLM tree (matches
# GLM52_BIND_HOST_TRITON=1). Paths are inside the image's vLLM install.
MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops"
LAYERS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
MODELS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"
KMOUNTS=(
  -v "$KERNELS_DIR/sparse_mla_kernels.py:$MLA/sparse_mla_kernels.py:ro"
  -v "$KERNELS_DIR/sparse_mla_env.py:$MLA/sparse_mla_env.py:ro"
  -v "$KERNELS_DIR/sm12x_sparse_mla_attn.py:$MLA/sm12x_sparse_mla_attn.py:ro"
  -v "$KERNELS_DIR/patch_flashmla_ops.py:$MLA/patch_flashmla_ops.py:ro"
  -v "$KERNELS_DIR/flashmla_sparse.py:$MLA/flashmla_sparse.py:ro"
  -v "$KERNELS_DIR/sm12x_deep_gemm_fallbacks.py:$OPS/sm12x_deep_gemm_fallbacks.py:ro"
  -v "$KERNELS_DIR/sm12x_mqa.py:$OPS/sm12x_mqa.py:ro"
  -v "$KERNELS_DIR/b12x_sparse_helpers.py:$OPS/b12x_sparse_helpers.py:ro"
  # upstream vLLM #46862: fused indexer Q rope+fp8-quant (fused_indexer_q_rope_quant)
  -v "$KERNELS_DIR/sparse_attn_indexer.py:$LAYERS/sparse_attn_indexer.py:ro"
  -v "$KERNELS_DIR/deepseek_v2.py:$MODELS/deepseek_v2.py:ro"
  # glm52-dcp: SM120 sparse impl patched to return decode LSE (DCP requirement).
  # FlashInfer 0.6.14's kernel supports return_lse; upstream never plumbed it.
  -v "$HOME/glm-dcp-patches/flashinfer_mla_sparse_sm120.py:$MLA/flashinfer_mla_sparse_sm120.py:ro"
)

# docker run base — IB passthrough is REQUIRED (without --device=/dev/infiniband
# + IPC_LOCK + memlock, NCCL silently drops to TCP: ~12 vs 30+ tok/s).
BASE=(
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --network host --ipc host --shm-size 10gb --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_DIR:/cache/huggingface"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
  # Per-node RoCE GID auto-detect wrapper (adapted from DeepSeek-DSpark).
  -v "$HOME/vllm/glm52-entrypoint.sh:/glm52-entrypoint.sh:ro"
)

# The serving command, with {port} resolved.
# NOTE: --max-num-seqs 6 requires the indexer MTP-overhang patch baked into the
# image (patches/fix-indexer-mtp-overhang.py, README step h) — unpatched vLLM
# crashes at >= 3 concurrent requests with MTP enabled.
SERVE=(
  /glm52-entrypoint.sh
  vllm serve /cache/huggingface/hub/glm52-int4-int8mix
  --served-model-name glm-5.2 --host 0.0.0.0 --port "$PORT"
  --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  # index_topk_pattern (both lanes — model-correctness fix, found in
  # CosmicRaisins' README 2026-07-10): GLM-5.2 trains indexer weights on only
  # 21/78 layers; QuantTrio ships index_topk_pattern:null and vLLM then runs
  # top-k through UNINITIALIZED weights on the 57 'shared' layers — coherent
  # under ~2K ctx, degrades beyond, craters MTP acceptance at depth (we
  # measured per-position acceptance 0.88/0.46/0.18/0.00 at 120K before this).
  # Pattern derived from the checkpoint's indexer_types; matches CosmicRaisins'.
  --hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
  --enable-prefix-caching
  --async-scheduling
  # MTP k=4 with the draft attention backend aligned to the backend our stack
  # actually selects for the main model (FLASHINFER_MLA_SPARSE_SM120, from
  # FlashInfer 0.6.14's native SM120 sparse-MLA support). Tony's original pinned
  # "FLASHMLA_SPARSE" here — on his older FlashInfer that WAS the main backend,
  # but on ours it mismatches: main allocates the 656-byte DSA KV record while
  # the FLASHMLA_SPARSE draft group expects plain-MLA 576 → the
  # "shape [3126,64,576] invalid for input of size 131241984" crash.
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"attention_backend":"FLASHINFER_MLA_SPARSE_SM120"}'
  --tensor-parallel-size 4 --pipeline-parallel-size 1
  # DCP4: shard the KV cache sequence-wise across the 4 TP ranks during decode.
  # With MLA the KV is otherwise replicated per rank, so DCP multiplies effective
  # KV capacity 4x — the same 10.95 GB/node pool supports ~800K tokens; we serve
  # 512K. Forum ref (thread 375416): DCP4+MTP4 held 20-37 tok/s to 640K.
  ${DCP_FLAGS}
  # 256K probe boot (was 512K): the 512K DCP boot froze all 4 nodes when a
  # FlashInfer autotune warmup allocation blew past unified memory (NVRM
  # NV_ERR_NO_MEMORY, 07-10). Autotune is disabled below; step context back
  # up only after a stable boot + depth test.
  --max-model-len ${LANE_MAXLEN} --max-num-seqs ${LANE_SEQS} --max-num-batched-tokens 8192
  # Skip FlashInfer autotune entirely — its mixed prefill/decode dummy run was
  # the allocation that killed the boxes. Cost: slightly untuned decode kernel.
  ${LANE_KERNELCFG}
  # gpu-mem-util 0.91 (tony's value). Pre-firmware our CUDA-visible free was
  # ~108 GB and 0.91's 110.68 GiB gate failed — we ran 0.88. Post-firmware
  # (driver 580.159.03) free is 112.69 GiB, so 0.91 fits with ~2 GiB margin.
  # KV stays pinned below; 0.91 only widens the activation/cudagraph budget.
  # KV shrunk 10.95G → 8.5G for DCP: the DCP4 boot at 10.95G left only
  # 0.5-0.9 GiB available per node (allgather buffers + 64-head decode +
  # extra graphs eat the old headroom) — one request away from another
  # freeze. 8.5G still gives a ~620K-token pool = 2.4x concurrency at 256K.
  # KV 7.0G (stepped down from tony's 10.95 through 8.5/8.0 during the DCP
  # bring-up): at 8.0 the boot SERVED with only 1.7 GiB free on the head and
  # the FIRST REQUEST's init (flashinfer workspace + Triton JIT + request
  # buffers) tripped the 1.5 GiB watchdog. 7.0 gives the head ~2.7 GiB steady.
  # Pool ~512K logical tokens at DCP4 = 2.0x concurrency at 256K.
  --gpu-memory-utilization 0.91 --kv-cache-memory-bytes ${LANE_KVBYTES}
  --kv-cache-dtype fp8_ds_mla
  # FULL graphs restored (PIECEWISE measured 9.8 tok/s — eager decode launch
  # overhead on Grace CPU is ~3x per-step cost; MTP acceptance was healthy at
  # 50%/3.0 so spec decode wasn't the issue). The earlier FULL+capture-32 boot
  # died at 1.6-2.0 GiB during decode capture with KV 8.5G; at KV 7.0G we have
  # +1.5 GiB more floor, so capture should bottom ~3.5 GiB — above the 1.5 GiB
  # watchdog. vLLM auto-downgrades FULL to FULL_AND_PIECEWISE for this backend.
  --distributed-executor-backend mp --compilation-config "${LANE_COMPCFG}"
)

# ---------------------------------------------------------------------------
# GLM_LANE=dcp2 — CosmicRaisins' validated DCP2-320K recipe on the FORK stack
# (recipes/glm52-quanttrio-unpruned-dcp2-320k.yaml, adapted to this cluster).
# Reference: 327K ctx, ~600 t/s prefill, ~22 t/s decode, flat to depth,
# unpruned QuantTrio. Fork image = local-inference-lab/vllm @e232d26 +
# CosmicRaisins' 3 patches + b12x@9cd63a7, baked as :e232d26-modded.
# Overrides the default lane wholesale: fork tree is self-contained, so the
# host Triton kernel mounts and the LSE patch mount MUST be dropped (they'd
# shadow fork files with incompatible upstream-era code).
# ---------------------------------------------------------------------------
if [ "$GLM_LANE" = "dcp2" ]; then
  IMAGE="vllm-node-eldritch-dcp:e232d26-modded"
  KMOUNTS=(
    # glm52-gb10 prewarm-tolerance patch (tony's NF3 fix pattern): sm_121a
    # fails some b12x CuteDSL prewarm shapes with "Operation creation failed";
    # wrap prewarm in try/except so boot survives — runtime compiles on demand.
    -v "$HOME/glm-dcp2-patches/sparse_attn_indexer.py:$LAYERS/sparse_attn_indexer.py:ro"
  )
  ENVV+=(
    -e "VLLM_USE_B12X_SPARSE_INDEXER=1"
    -e "VLLM_USE_V2_MODEL_RUNNER=1"
    -e "VLLM_DCP_GLOBAL_TOPK=1"
    -e "VLLM_DCP_SHARD_DRAFT=1"
    -e "GLM52_BIND_HOST_TRITON=0"
    # Dual-rail NCCL (matches CosmicRaisins' recipe): rail-2 (enP2p1s0f0np0 /
    # roceP2p1s0f0) verified UP, MTU 9000, full jumbo mesh, RoCEv2 IPv4 GID at
    # idx=3 on all nodes (2026-07-10). DCP does per-layer per-step ag_rs — the
    # single-rail config was the prime suspect for 15.7 vs their 22 tok/s.
    -e "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0"
    -e "NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0"
    # Channels 8 -> 4 for DCP (overrides the global pin; docker takes the last
    # -e). The 8-channel tune was Veghes' for tony's NON-DCP stack; ciprianveg's
    # DCP-era finding was 4, and CosmicRaisins doesn't pin at all. DCP decode is
    # small-message latency-bound: fewer channels = less per-op overhead.
    -e "NCCL_MAX_NCHANNELS=4"
    -e "NCCL_MIN_NCHANNELS=4"
    # NOT enabled: VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 (ciprianveg #155,
    # +5-10% at dcp1) — tested 2026-07-11 at DCP2: no decode gain (13.8 vs
    # 12.8-13.9 baseline) and the d8K bench leg tripped the 1.5GiB watchdog
    # (extra workspace from the decode-shaped verify path). Reverted.
  )
  SERVE=(
    /glm52-entrypoint.sh
    vllm serve /cache/huggingface/hub/glm52-int4-int8mix
    --served-model-name glm-5.2 --host 0.0.0.0 --port "$PORT"
    --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
    --enable-prefix-caching
    --hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
    # k=3, not the recipe's k=4: CosmicRaisins revealed (thread tail, Jul 7)
    # his PRODUCTION runs k=3 and the repo's k=4 default was a mistake — his
    # measured ~22 tables are k=3. Our own per-position acceptance showed the
    # 4th draft token accepted ~0% of the time (0.88/0.46/0.18/0.00): k=4 runs
    # a full extra draft pass per step for a token it never keeps.
    --speculative-config '{"model":"/cache/huggingface/hub/glm52-int4-int8mix","method":"mtp","quantization":"compressed-tensors","draft_attention_backend":"B12X_MLA_SPARSE","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
    # clear_thinking=false (CosmicRaisins, thread tail): GLM's template strips
    # prior turns' thinking by default, mutating the prefix every turn → full
    # conversation re-prefill per message. Keeping thinking stabilizes the
    # prefix for cache hits (snappy multi-turn); context cost is fine at 327K.
    --default-chat-template-kwargs '{"clear_thinking":false}'
    --tensor-parallel-size 4 --pipeline-parallel-size 1
    --decode-context-parallel-size 2 --dcp-kv-cache-interleave-size 1
    --attention-backend B12X_MLA_SPARSE
    # batched-tokens 2048 (recipe: 4096): the 150K-deep prefill's activation
    # transient tripped the 1.5 GiB watchdog on the head (1.48 GiB, 18:19).
    # Halving the chunk halves the transient; costs some prefill throughput.
    --max-model-len 327680 --max-num-seqs 1 --max-num-batched-tokens 2048
    # gmu 0.89 (recipe: 0.90): the head rode ~2.1G available and the watchdog
    # (1.5G) killed it during a tg512 bench on 2026-07-12. -0.01 gmu ≈ +1.1G
    # floor per node; KV pool is unaffected (auto-sized from the gmu budget,
    # loses ~2% capacity).
    --gpu-memory-utilization 0.89 --kv-cache-dtype fp8_ds_mla
    --async-scheduling
    --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":10}'
  )
fi

# Build the full `docker run` for a given rank, as a single shell-quoted string.
docker_run_cmd() {
  local rank="$1" headless="$2"
  local cmd=(docker run -d --name "$NAME" "${BASE[@]}" "${ENVV[@]}" "${KMOUNTS[@]}"
             -e "NODE_RANK=$rank" -e "MASTER_ADDR=$HEAD"
             "$IMAGE" "${SERVE[@]}"
             --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD" --master-port "$MASTER_PORT")
  [ "$headless" = 1 ] && cmd+=(--headless)
  # printf %q on each token yields a paste-safe, correctly-quoted command line.
  local out="" t
  for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done
  printf '%s' "${out# }"
}

say "GLM-5.2 launch: ${NNODES} nodes, head=$HEAD:$PORT, image=$IMAGE"
[ "$DRYRUN" = 1 ] && echo "   (dry-run — nothing will be executed)"

# Workers first (rank 1..N-1, headless), then the head (rank 0).
for ((rank=1; rank<NNODES; rank++)); do
  w="${NODES[$rank]}"
  run="$(docker_run_cmd "$rank" 1)"
  shell="docker rm -f $NAME 2>/dev/null; $run"
  if [ "$DRYRUN" = 1 ]; then
    printf '\n# worker %s (rank %d, headless)\nssh %s@%s %q\n' "$w" "$rank" "$(node_user "$w")" "$w" "$shell"
  else
    printf '   worker %s rank=%d (headless)\n' "$w" "$rank"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$(node_user "$w")@$w" "$shell" \
      || die "worker launch failed on $w"
  fi
done

run="$(docker_run_cmd 0 0)"
shell="docker rm -f $NAME 2>/dev/null; $run"
if [ "$DRYRUN" = 1 ]; then
  printf '\n# head %s (rank 0)\n%s\n' "$HEAD" "$shell"
  exit 0
fi
printf '   head %s rank=0\n' "$HEAD"
bash -c "$shell" || die "head launch failed"

say "launched"
echo "   poll:  curl -s http://localhost:$PORT/v1/models"
echo "   logs:  docker logs -f $NAME   (on the head node)"
echo "   stop:  ./launch.sh --stop"
echo "   Ready in ~12 min load + ~10 min cudagraph warmup; serves as 'glm-5.2'."
