#!/usr/bin/env bash
# LAYER 2/3 — PER-NODE LAUNCHER (internal; called by glm-serve.sh). Builds the
# vLLM flags/env/mounts for the chosen GLM_LANE and issues `docker run` on each
# node. Runnable standalone with --dry-run / --stop. See scripts/README.md.
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
# LANE SELECTOR — all lanes run the same fork stack, differing ONLY in
# context / concurrency / DCP size (detail + how to add a lane: scripts/README.md).
# Naming: <dcp-size>[-cc<per-stream-K>]. Bare = single-stream (c=1); -ccNNN =
# multi-user, NNN K per stream. DCP is one KV budget spent on depth OR width:
#   dcp2        — 327K ctx, c=1, DCP2  → flagship (~25 t/s coherent)
#   dcp2-cc200  — 200K ctx, c=3, DCP2  → multi-user (~40 t/s aggregate, ~14 each)
#   dcp4        — 655K ctx, c=1, DCP4  → max-context specialty (~24 t/s coherent)
#   dcp4-cc200  — 200K ctx, c=5, DCP4  → multi-user wide (~47 t/s aggregate, ~10 each)
#   dcp4-cc128  — 128K ctx, c=8, DCP4  → max-width (DCP4's pool spread over 8 streams)
# ---------------------------------------------------------------------------
GLM_LANE="${GLM_LANE:-dcp2}"
case "$GLM_LANE" in
  dcp2)       L_MAXLEN=327680; L_SEQS=1; L_BATCHED=2048; L_CAPTURE=10; L_DCP=2 ;;
  dcp2-cc200) L_MAXLEN=200000; L_SEQS=3; L_BATCHED=4096; L_CAPTURE=16; L_DCP=2 ;;
  dcp4)       L_MAXLEN=655360; L_SEQS=1; L_BATCHED=2048; L_CAPTURE=10; L_DCP=4 ;;
  dcp4-cc200) L_MAXLEN=200000; L_SEQS=5; L_BATCHED=2048; L_CAPTURE=32; L_DCP=4; L_GMU=0.88 ;;  # gmu 0.88: 5 concurrent DEEP prefills need +1.1GiB head headroom (0.89 watchdog-killed). batched 2048 interleaves prefill/decode fairly. Validated: 5x197K preempt=0, head floor 2.07GiB.
  dcp4-cc128) L_MAXLEN=131072; L_SEQS=8; L_BATCHED=2048; L_CAPTURE=48; L_DCP=4; L_GMU=0.88 ;;  # gmu 0.88 too (8 streams). Not yet deep-fill validated.
  *) echo "GLM_LANE must be dcp2 (327K,c=1), dcp2-cc200 (200K,c=3), dcp4 (655K,c=1), dcp4-cc200 (200K,c=5), or dcp4-cc128 (128K,c=8); got '$GLM_LANE'" >&2; exit 1 ;;
esac
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
# (Both lanes set VLLM_USE_V2_MODEL_RUNNER in the fork block below.)

# Triton sparse-MLA kernels, bound read-only over the vLLM tree (matches
# GLM52_BIND_HOST_TRITON=1). Paths are inside the image's vLLM install.
MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops"
LAYERS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
MODELS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"

# docker run base — IB passthrough is REQUIRED (without --device=/dev/infiniband
# + IPC_LOCK + memlock, NCCL silently drops to TCP: ~12 vs 30+ tok/s).
BASE=(
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --network host --ipc host --shm-size 10gb --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_DIR:/cache/huggingface"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
  # Per-node RoCE GID auto-detect wrapper (adapted from DeepSeek-DSpark).
  -v "$HOME/vllm/glm-container-entrypoint.sh:/glm-container-entrypoint.sh:ro"
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
if true; then  # every lane runs the fork stack (the top-of-file case validated GLM_LANE)
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
    # idx=3 on all nodes. DCP does per-layer per-step ag_rs, so both rails help
    # the collective. (For the record, the decode gap that had us chasing rails/
    # channels/cutlass was NOT any of them — it was a GPU stuck at 1/4 clock;
    # see the clock-health preflight in glm-serve.sh.)
    -e "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0"
    -e "NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0"
    # Channels 8 -> 4 for DCP (overrides the global pin; docker takes the last
    # -e). The 8-channel tune was Veghes' for tony's NON-DCP stack; ciprianveg's
    # DCP-era finding was 4, and CosmicRaisins doesn't pin at all. DCP decode is
    # small-message latency-bound: fewer channels = less per-op overhead.
    -e "NCCL_MAX_NCHANNELS=4"
    -e "NCCL_MIN_NCHANNELS=4"
    # Mesh-mode NCCL (from eugr/spark-vllm-docker) — targets DCP2 decode's
    # per-step dual-rail ag_rs. Measured +5% (tg512 median TPOT ~64→~61 ms) and
    # kept: NET_PLUGIN=none forces the built-in IB transport; IB_MERGE_NICS=0
    # keeps the two 200G rails as separate logical devices; SUBNET_AWARE_ROUTING
    # picks paths per rail.
    -e "NCCL_NET_PLUGIN=none"
    -e "NCCL_IB_MERGE_NICS=0"
    -e "NCCL_IB_SUBNET_AWARE_ROUTING=1"
    # Skip vLLM's cudagraph-memory estimation pass (eugr) — frees a little floor.
    -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
    # NOT enabled: VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 (ciprianveg #155,
    # +5-10% at dcp1) — tested 2026-07-11 at DCP2: no decode gain (13.8 vs
    # 12.8-13.9 baseline) and the d8K bench leg hit the ~1.5GiB floor
    # (extra workspace from the decode-shaped verify path). Reverted.
  )
  SERVE=(
    /glm-container-entrypoint.sh
    vllm serve /cache/huggingface/hub/glm52-int4-int8mix
    --served-model-name glm-5.2 --host 0.0.0.0 --port "$PORT"
    --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
    --enable-prefix-caching
    --hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
    # k=4 (matches CosmicRaisins' recipe + the ~25 coherent bench). On random
    # tokens k=3/k=4 are a wash, but on real coherent prompts the 4th draft
    # token accepts often enough to help. draft_tensor_parallel_size is LOCKED
    # at the target TP (=4) under DCP2 — vLLM requires it be 1 or target-TP AND
    # divisible by dcp_size=2, so 4 is the only legal value ("draft tp=1" is a
    # non-DCP trick). NOTE: needs the PR#72 draft-under-DCP patches (in Stack) or
    # k>1 acceptance collapses under DCP.
    --speculative-config '{"model":"/cache/huggingface/hub/glm52-int4-int8mix","method":"mtp","quantization":"compressed-tensors","draft_attention_backend":"B12X_MLA_SPARSE","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}'
    # clear_thinking=false (CosmicRaisins, thread tail): GLM's template strips
    # prior turns' thinking by default, mutating the prefix every turn → full
    # conversation re-prefill per message. Keeping thinking stabilizes the
    # prefix for cache hits (snappy multi-turn); context cost is fine at 327K.
    --default-chat-template-kwargs '{"clear_thinking":false}'
    --tensor-parallel-size 4 --pipeline-parallel-size 1
    --decode-context-parallel-size ${L_DCP} --dcp-kv-cache-interleave-size 1
    --attention-backend B12X_MLA_SPARSE
    # batched-tokens 2048 (recipe: 4096). Single-stream lanes: halves the
    # deep-prefill activation transient. -cc lanes: small chunks INTERLEAVE one
    # stream's big prefill with the others' decode (a 50K paste dips them, doesn't
    # starve them). NB it does NOT bound concurrent-prefill working set — gmu does.
    --max-model-len ${L_MAXLEN} --max-num-seqs ${L_SEQS} --max-num-batched-tokens ${L_BATCHED}
    # gpu-memory-utilization is PER-LANE (L_GMU): 0.89 single-stream, 0.88 for the
    # -cc lanes. 5 concurrent DEEP prefills hold ~1G more rank-0 working set than a
    # single stream and breached the ~1.5G head watchdog at 0.89; -0.01 gmu ≈ +1.1G
    # floor/node (KV pool auto-resizes, loses ~2%). Runtime override: GLM_GMU=…
    # Validated dcp4-cc200 cold 5x197K: preempt=0, head floor 2.07 GiB.
    --gpu-memory-utilization ${GLM_GMU:-${L_GMU:-0.89}} --kv-cache-dtype fp8_ds_mla
    --async-scheduling
    --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":'"${L_CAPTURE}"'}'
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
