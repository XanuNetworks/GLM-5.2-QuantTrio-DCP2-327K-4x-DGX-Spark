# utils/ — operational helpers

Standalone tools that support the recipe. Each is self-contained; **edit the
`NODES` / `NODE_USER` block at the top** for your cluster. (The launch chain
lives in [`scripts/`](../scripts/); these are the extras.)

| script | what it does | when to run it |
|---|---|---|
| **`check-clocks.sh`** | Burns each GPU 5 s and reads its clock under load — catches the GB10 **stuck-low-clock bug** (a wedged GPU gates the whole TP cluster). | Decode is slower than expected; before a long serving session; any time a node "feels" slow. |
| **`benchmark.sh`** | Reproduces the README numbers against a running endpoint: coherent real-world decode, random-token floor, cold prefill. | After bring-up, to confirm you built the recipe right; to compare a config change. |
| **`stage.sh`** | One-time staging of weights + NCCL + kernels onto every node (scaffold — review before running; weights are ~400 GB). | First-time setup, before the first launch. |
| **`glm-memwatch.sh`** | Optional unified-memory watchdog — `docker kill`s the container if MemAvailable drops below ~1.5 GiB. `glm-serve.sh` stages + arms it on each node automatically. | Runs itself; nothing to invoke manually. |

## check-clocks.sh
The single highest-value check. A GB10 can silently wedge one GPU at ~660 MHz
(reports `P0` but delivers a quarter-clock, cold, ~17 W under load) — and in a
synchronized TP cluster the slowest GPU caps all of them. Healthy ≈ 2300–2500 MHz
/ ~90 W; wedged ≈ 660 / ~17. **Fix a wedged GPU with a full cold power cycle**
(pull the plug ~30 s) — a warm reboot won't clear it. `glm-serve.sh` runs this
same check in preflight; this is the on-demand version.

```bash
./utils/check-clocks.sh          # exits non-zero if any GPU is wedged
```

## benchmark.sh
Measures honestly: coherent prompts (the real-world headline), a random-token
floor (pessimistic), and a **cold** prefill on a novel prompt (random-token
benches cache-hit and lie about prefill). Post the coherent number; keep the
random one as an honest floor.

```bash
./utils/benchmark.sh             # endpoint must be serving
# optional tool-calling grade (0-100 → ★…★★★★★):
#   uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git
#   tool-eval-bench --base-url http://<head>:8000
```

## stage.sh
Scaffold for the one-time staging the launcher assumes (per-node weights symlink,
`libnccl.so.2`, Triton kernels). Review each step — the ~400 GB weight transfer
should be planned, not fired blindly.

## glm-memwatch.sh
GB10 uses unified memory, so a runaway allocation can starve the whole OS (sshd
included) — an early uncontrolled OOM once hard-froze all four nodes. As
insurance, this watchdog `docker kill`s the vLLM container when MemAvailable drops
below ~1.5 GiB (1 s poll). `glm-serve.sh` stages it to each node and arms it in
preflight — you never run it by hand. Honest note: in the tuned configs it rarely
fires (the head sits well above the floor); it earns its keep only on tighter
gmu/chunk settings. Keep it as a cheap backstop, not a proven safety net.
