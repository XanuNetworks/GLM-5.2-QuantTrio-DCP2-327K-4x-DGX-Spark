#!/bin/bash
# Start GLM-5.2-Int4-Int8Mix (TP=4) across all 4 Sparks (.11 head, .12/.13/.14 workers).
#
# Wrapper around ~/vllm/glm-launch.sh — tony's launch.sh adapted for this cluster
# (NODES, users, IPs, image name). See project_glm_5_2_cluster.md for full details.
#
# Endpoint:  http://192.168.NNN.1:8000/v1  (served-model-name: glm-5.2)
# Max ctx:   200000
# Cold boot: ~12 min weight load + ~10 min cudagraph warmup = ~22 min to serve
#
# Usage:
#   ./start-glm-5.2.sh           # start
#   ./start-glm-5.2.sh stop      # stop containers on all 4 nodes
#   ./start-glm-5.2.sh status    # /v1/models probe + container ps
#   ./start-glm-5.2.sh logs      # tail head-node container logs
#   ./start-glm-5.2.sh dry-run   # print the docker commands without running

set -e

HEAD_IP="192.168.NNN.1"
PORT=8000
CONTAINER="vllm_glm52"
LAUNCH="$HOME/vllm/glm-launch.sh"

case "${1:-start}" in
  stop)
    "$LAUNCH" --stop
    exit 0
    ;;
  status)
    echo "--- container state on all 4 nodes ---"
    for ip in 11 12 13 14; do
      echo -n ".${ip}: "
      ssh -o ConnectTimeout=3 "YOURUSER@192.168.NNN.${ip}" "docker ps --filter name=$CONTAINER --format '{{.Status}}'" 2>/dev/null || echo "unreachable"
    done
    echo
    echo "--- /v1/models on $HEAD_IP:$PORT ---"
    curl -s --max-time 5 "http://${HEAD_IP}:${PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(API not ready yet)"
    exit 0
    ;;
  logs)
    docker logs -f "$CONTAINER"
    exit 0
    ;;
  dry-run)
    "$LAUNCH" --dry-run
    exit 0
    ;;
  start)
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs|dry-run}"
    exit 1
    ;;
esac

# Memory watchdogs: armed ONLY for the DCP lane. The fast lane is proven
# stable without them, and its autotune phase transiently dips below the
# 1.5 GiB line — a watchdog would kill a boot that always succeeds. For the
# DCP lane the watchdog is mandatory (its warmup froze all 4 boxes once).
if [ "${GLM_LANE:-fast}" = "dcp" ] || [ "${GLM_LANE:-fast}" = "dcp2" ]; then
  echo "[preflight] arming memory watchdog on all 4 nodes (DCP lane) ..."
  for ip in 11 12 13 14; do
    # Two SEPARATE ssh calls: combining pkill + start in one command line makes
    # pkill match its own wrapper shell (the launch path contains the pattern)
    # and kill the ssh session — watchdogs silently never start. Bit us twice.
    ssh -o ConnectTimeout=3 "YOURUSER@192.168.NNN.${ip}" "pkill -f '[g]lm-memwatch' 2>/dev/null; true"
    ssh -o ConnectTimeout=3 "YOURUSER@192.168.NNN.${ip}" "setsid nohup \$HOME/vllm/glm-memwatch.sh >/dev/null 2>&1 </dev/null & exit 0"
    ssh -o ConnectTimeout=3 "YOURUSER@192.168.NNN.${ip}" "sleep 1; grep -q 'memwatch armed' /tmp/glm-memwatch.log 2>/dev/null" \
      && echo "  .${ip}: watchdog armed (verified)" || echo "  .${ip}: WATCHDOG NOT RUNNING — aborting launch"
  done
else
  echo "[preflight] fast lane — disarming any leftover memory watchdogs ..."
  for ip in 11 12 13 14; do
    ssh -o ConnectTimeout=3 "YOURUSER@192.168.NNN.${ip}" "pkill -f '[g]lm-memwatch' 2>/dev/null; true" && echo "  .${ip}: clear"
  done
fi

# Preflight: (re)apply NIC PFC/DSCP-trust on every node — mlnx_qos settings
# don't survive reboots. Runs via privileged container (mlnx_qos needs
# CAP_NET_ADMIN; --pid host avoids netlink PID collisions). Pairs with the
# the switch's RoCE QoS traffic class; verified 2026-07-12: zero packet loss
# (packet_seq_err +0) across a 78K-token prefill.
echo "[preflight] applying NIC PFC prio-3 + DSCP trust on all nodes ..."
for ip in 11 12 13 14; do
  ok=0
  for dev in enp1s0f0np0 enP2p1s0f0np0; do
    ssh -o ConnectTimeout=3 "YOURUSER@192.168.NNN.${ip}" "docker run --rm --privileged --network host --pid host \
      -v /usr/bin/mlnx_qos:/usr/local/bin/mlnx_qos:ro \
      -v /usr/lib/python3/dist-packages/dcbnetlink.py:/deps/dcbnetlink.py:ro \
      -v /usr/lib/python3/dist-packages/netlink.py:/deps/netlink.py:ro \
      -v /usr/lib/python3/dist-packages/genetlink.py:/deps/genetlink.py:ro \
      -e PYTHONPATH=/deps \
      --entrypoint python3 vllm-node-eldritch-dcp:e232d26-modded /usr/local/bin/mlnx_qos -i ${dev} --trust dscp --pfc 0,0,0,1,0,0,0,0 >/dev/null 2>&1" && ok=$((ok+1))
  done
  echo "  .${ip}: PFC applied on ${ok}/2 rails"
done

# Preflight: drop page cache on every node (tony's README Gotcha 6).
# GB10 kernel-reclaim can stall the model load; freeing 15-30 GB of
# page cache guarantees vLLM sees enough for --gpu-memory-utilization 0.91.
echo "[preflight] dropping page cache on all 4 nodes ..."
for ip in 11 12 13 14; do
  ssh -o ConnectTimeout=3 "YOURUSER@192.168.NNN.${ip}" "docker run --rm --privileged -v /proc:/host_proc alpine sh -c 'sync && echo 3 > /host_proc/sys/vm/drop_caches' >/dev/null 2>&1 && echo '  .${ip}: cache dropped'" || echo "  .${ip}: skip (unreachable or docker error)"
done

# Preflight — every node needs: kernels, NCCL lib, weights symlink
for ip in 11 12 13 14; do
  for f in \
    "\$HOME/glm-triton/sm12x_mqa.py" \
    "/var/tmp/models/hub/nccl-2.30.4/libnccl.so.2" \
    "/var/tmp/models/hub/glm52-int4-int8mix/config.json"; do
    if ! ssh -o ConnectTimeout=3 "YOURUSER@192.168.NNN.${ip}" "test -e $f" 2>/dev/null; then
      echo "Error: missing $f on .${ip}"
      echo "       Run the staging steps first (kernels rsync, NCCL wheel extract, weights symlink)."
      exit 1
    fi
  done
done

# Ensure image is present on all 4 nodes
for ip in 11 12 13 14; do
  if ! ssh -o ConnectTimeout=3 "YOURUSER@192.168.NNN.${ip}" "docker image inspect vllm-node-tf5-glm52-b12x:probe-modded > /dev/null 2>&1"; then
    echo "Error: image vllm-node-tf5-glm52-b12x:probe-modded missing on .${ip}"
    echo "       Build on .11 then docker-save / docker-load to the others."
    exit 1
  fi
done

# Launch — tony's launch.sh handles ssh to each worker and local head start.
"$LAUNCH"

echo
echo "Cold-boot expected ~22 min (12 min weight load + 10 min cudagraph warmup)."
echo "  Poll:    curl -s http://${HEAD_IP}:${PORT}/v1/models"
echo "  Status:  $0 status"
echo "  Logs:    $0 logs"
echo "  Stop:    $0 stop"
