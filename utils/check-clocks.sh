#!/bin/bash
# check-clocks.sh — detect the GB10 stuck-low-clock bug.
#
# A GB10 GPU can silently wedge at ~1/4 clock (reports perf-state P0 but delivers
# ~660 MHz, stays cold, draws ~17 W under full load, no throttle flag). In a
# synchronized TP cluster the SLOWEST GPU gates all of them, so one wedged node
# caps the whole cluster ~30% and every benchmark understates.
#
# This burns each GPU for 5 s and reads its clock under load:
#   healthy ≈ 2300–2500 MHz / ~90 W    wedged ≈ 660 MHz / ~17 W
#
# FIX for a wedged GPU: a full COLD POWER CYCLE (shut down, pull the plug ~30 s,
# power back on). A warm `reboot` does NOT clear it — the GPU firmware holds the
# wedge on standby power. See the README "Is your decode slow?" section.
#
# EDIT these for your cluster:
NODES=(192.168.NNN.11 192.168.NNN.12 192.168.NNN.13 192.168.NNN.14)
NODE_USER=YOURUSER
IMG="${GLM_DCP2_IMAGE:-vllm-node-eldritch-dcp:e232d26-modded}"   # any image with torch + nvidia-smi

BURN='
import torch,time,subprocess
a=torch.randn(8192,8192,device="cuda",dtype=torch.bfloat16)
t=time.time()
while time.time()-t<5:(a@a).sum().item()
print(subprocess.run(["nvidia-smi","--query-gpu=clocks.current.sm,power.draw,temperature.gpu","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())'

echo "GPU clock-health check (5 s burn per node) ..."
bad=0
for ip in "${NODES[@]}"; do
  out=$(ssh -o ConnectTimeout=6 "$NODE_USER@$ip" "docker run --rm --gpus all --entrypoint python3 $IMG -c '$BURN' 2>/dev/null" 2>/dev/null)
  sm=$(echo "$out" | grep -oE '^[0-9]+')
  if   [ -z "$sm" ];        then echo "  $ip: probe FAILED (unreachable / no GPU?)"
  elif [ "$sm" -lt 1500 ];  then echo "  $ip: 🔴 WEDGED — $out  → cold power-cycle this node"; bad=1
  else                           echo "  $ip: ✅ healthy — $out"; fi
done
if [ "$bad" = 1 ]; then
  echo
  echo "A GPU is stuck at low clock — it gates the whole TP cluster (~30% slower)."
  echo "A warm reboot will NOT fix it: cleanly shut the node down, PULL its power"
  echo "for ~30 s, power back on, and re-run. (start-glm-serve preflight does this check too.)"
  exit 1
fi
echo "✅ all GPUs boost normally under load"
