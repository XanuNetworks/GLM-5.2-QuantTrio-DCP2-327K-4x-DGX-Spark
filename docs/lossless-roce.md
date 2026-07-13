# Lossless RoCE (ECN + PFC) — optional, worth it

DCP's per-step syncs are small and latency-bound, so decode never loses packets —
but big prefill all-gathers overrun a plain L2 switch and eat go-back-N
retransmits. If your switch supports RoCE QoS (ECN/DCQCN + PFC on a dedicated
traffic class), configuring it per your vendor's lossless-RoCE guide is worth it.

Verify with the NIC hardware counters
(`/sys/class/infiniband/<hca>/ports/1/hw_counters/{packet_seq_err,out_of_sequence,rp_cnp_handled,np_cnp_sent}`,
diffed around an isolated ~78K-token prefill):

| stage | packet_seq_err / big prefill |
|---|---|
| raw (no QoS) | ~1,000–2,200 per sender |
| + ECN/DCQCN | ~90 (−95%) |
| + PFC | **0 (lossless)** |

**The catch that cost us an afternoon:** the switch QoS config does **nothing** on
its own — NCCL sends at DSCP 0 by default and sails past every classifier. The
missing key is node-side `NCCL_IB_TC=106` (→ DSCP 26, which the switch then maps
to the RoCE traffic class), already set in `scripts/glm-node-launch.sh`. PFC on the
NICs is `mlnx_qos -i <dev> --trust dscp --pfc 0,0,0,1,0,0,0,0` on both rails per
node — needs root and isn't reboot-persistent, so `glm-serve.sh` reapplies it in
preflight via a privileged container (`--privileged --network host --pid host`;
**`--pid host` is required** or the netlink bind fails "Address already in use").

**Payoff was honest:** prefill throughput unchanged (drops were ~0.1% of volume),
but decode picked up ~+10% (queue prioritization trimming latency jitter on the
small sync ops) and the fabric is now clean instead of accidentally-working.
