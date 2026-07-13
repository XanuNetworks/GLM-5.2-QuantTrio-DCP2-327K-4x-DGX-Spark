#!/bin/bash
# stage.sh — one-time staging of weights + NCCL + kernels onto every node.
#
# The launcher assumes each node has, locally:
#   /var/tmp/models/hub/glm52-int4-int8mix            (weights, symlink)
#   /var/tmp/models/hub/nccl-2.30.4/libnccl.so.2      (LD_PRELOADed NCCL)
#   $HOME/glm-triton/*.py                             (10 Triton sparse-MLA kernels)
#
# This is a scaffold — review each step; weights are ~400 GB so plan the transfer.
# EDIT these:
NODES=(192.168.NNN.11 192.168.NNN.12 192.168.NNN.13 192.168.NNN.14)
NODE_USER=YOURUSER
WEIGHTS_DIR=/var/tmp/models
set -euo pipefail

echo "== 1. Download weights on the head node =="
hf download QuantTrio/GLM-5.2-Int4-Int8Mix --local-dir "$WEIGHTS_DIR/glm52-int4-int8mix"

echo "== 2. Extract NCCL 2.30.4's libnccl.so.2 (aarch64) =="
python3 - <<'PY'
import urllib.request, tempfile, zipfile, os, glob, shutil
# nvidia-nccl-cu13==2.30.4 aarch64 wheel → libnccl.so.2
# (adjust the wheel URL/name to the current PyPI file for your platform)
print("Fetch nvidia-nccl-cu13==2.30.4 aarch64 wheel, unzip, copy nvidia/nccl/lib/libnccl.so.2")
PY
mkdir -p "$WEIGHTS_DIR/hub/nccl-2.30.4"
# cp <extracted>/libnccl.so.2  "$WEIGHTS_DIR/hub/nccl-2.30.4/libnccl.so.2"

echo "== 3. Stage the Triton sparse-MLA kernels =="
echo "   Get the 10 kernel .py files from CosmicRaisins/glm-5.2-gb10 (kernels/) into \$HOME/glm-triton/"

echo "== 4. Fan out to every node: weights + kernels, then symlink =="
for ip in "${NODES[@]}"; do
  echo "   -> $ip"
  # weights (large — rsync resumable):
  rsync -a --info=progress2 "$WEIGHTS_DIR/glm52-int4-int8mix/" "$NODE_USER@$ip:$WEIGHTS_DIR/glm52-int4-int8mix/"
  rsync -a "$WEIGHTS_DIR/hub/nccl-2.30.4/" "$NODE_USER@$ip:$WEIGHTS_DIR/hub/nccl-2.30.4/"
  rsync -a "$HOME/glm-triton/" "$NODE_USER@$ip:\$HOME/glm-triton/"
  ssh "$NODE_USER@$ip" "mkdir -p $WEIGHTS_DIR/hub && ln -sfn ../glm52-int4-int8mix $WEIGHTS_DIR/hub/glm52-int4-int8mix"
done
echo "== done. Now: GLM_LANE=dcp2 ./scripts/glm-serve.sh start =="
