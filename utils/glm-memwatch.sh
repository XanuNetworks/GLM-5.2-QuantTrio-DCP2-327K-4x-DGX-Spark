#!/bin/bash
# glm-memwatch.sh — emergency brake for unified-memory exhaustion on GB10.
#
# The 2026-07-10 DCP boot froze all 4 Sparks: a warmup/autotune allocation ate
# >100 GB in ~7 s, the NVIDIA driver hit NV_ERR_NO_MEMORY, and with unified
# memory that starves the whole OS (sshd included) → hard reboot required.
# This watchdog kills the vLLM container BEFORE MemAvailable hits zero.
#
# Runs for at most 1 hour (covers a full cold boot), then exits.
# Log: /tmp/glm-memwatch.log

# 5 GiB (was 10): the graph-capped DCP boot peaks at 7-10 GiB available during
# cudagraph capture — a controlled transient, not the >100GB/7s runaway this
# watchdog exists for. 10 GiB was killing boots that would have completed.
LIMIT_KB=$((3 * 1024 * 2 * 256))  # 1.5 GiB    # pure anti-freeze backstop
CONTAINER="vllm_glm52"
END=$(( $(date +%s) + 3600 ))

echo "$(date '+%F %T') memwatch armed (limit 1.5GiB, 1h)" >> /tmp/glm-memwatch.log
while [ "$(date +%s)" -lt "$END" ]; do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$avail" -lt "$LIMIT_KB" ]; then
    echo "$(date '+%F %T') memwatch: MemAvailable=${avail}KB < limit — killing $CONTAINER" >> /tmp/glm-memwatch.log
    docker kill "$CONTAINER" >> /tmp/glm-memwatch.log 2>&1
    exit 0
  fi
  sleep 1
done
echo "$(date '+%F %T') memwatch: 1h elapsed, exiting (boot survived)" >> /tmp/glm-memwatch.log
