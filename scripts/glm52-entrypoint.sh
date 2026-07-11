#!/bin/bash
# glm52-entrypoint.sh — auto-detect the RoCEv2 IPv4 GID index before exec.
# Per-node GID index for the same-named HCA differs across our Sparks (.11/.12
# use idx=3, .13/.14 use idx=4 depending on firmware/init order), so hardcoding
# NCCL_IB_GID_INDEX in the docker env crashes ibv_modify_qp on the odd node.
# Mirror the DeepSeek-DSpark stack's approach: probe /sys at container start,
# pick the RoCE v2 GID whose value maps to an IPv4 address, then exec the real
# command. Bind-mounted read-only from the host.
set -eu

HCA="${NCCL_IB_HCA%%,*}"
if [ -n "${HCA:-}" ] && [ -d "/sys/class/infiniband/$HCA/ports/1" ]; then
  for i in $(seq 0 15); do
    t=$(cat "/sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i" 2>/dev/null || true)
    g=$(cat "/sys/class/infiniband/$HCA/ports/1/gids/$i" 2>/dev/null || true)
    case "$t" in
      *"RoCE v2"*)
        case "$g" in
          *"0000:0000:0000:0000:0000:ffff:"*)
            export NCCL_IB_GID_INDEX=$i
            echo "[glm52-entrypoint] HCA=$HCA NCCL_IB_GID_INDEX=$i gid=$g"
            break
            ;;
        esac
        ;;
    esac
  done
fi
if [ -z "${NCCL_IB_GID_INDEX:-}" ]; then
  echo "[glm52-entrypoint] WARNING: no RoCEv2 IPv4 GID found for HCA=$HCA; NCCL will auto-select" >&2
fi

exec "$@"
