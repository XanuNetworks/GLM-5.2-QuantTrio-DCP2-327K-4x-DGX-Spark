# Benchmark — `dcp2-cc200` (multi-user, DCP2)

**Config:** 200K ctx · `max-num-seqs 3` · DCP2 · `max-num-batched-tokens 4096` · cudagraph capture 16
**Role:** multi-user serving on the DCP2 budget. DCP2's 2× KV fits ~3 streams at 200K.
**Stack:** identical fork stack to `dcp2` — only ctx / concurrency / capture differ.

## Concurrency scaling (random tokens, shallow — 512 in / 512 out)

| concurrency | aggregate t/s | per-stream t/s | TTFT |
|---|---|---|---|
| c=1 | ~18 | ~20 | ~1.3 s |
| c=2 | ~30 | ~15 | ~1.3 s |
| c=3 | **~41** | **~14** | ~1.3 s |

Real batching: aggregate climbs with concurrency and **TTFT stays flat ~1.3 s** — the
server is genuinely running the streams in parallel, not queueing them (which is what
`max-num-seqs 1` lanes do — there extra requests wait and TTFT balloons).

`max-num-seqs 3` is the server ceiling; a client asking for c=1/c=2 simply uses fewer
slots — no reboot needed to sweep concurrency **at or below** the ceiling.

## How to reproduce

```bash
GLM_LANE=dcp2-cc200 ./scripts/glm-serve.sh start
# sweep concurrency against the running server:
for c in 1 2 3; do
  vllm bench serve --backend openai-chat --base-url http://<HEAD>:8000 \
    --endpoint /v1/chat/completions --model glm-5.2 \
    --dataset-name random --num-prompts $((c*2)) --max-concurrency $c \
    --random-input-len 512 --random-output-len 512
done
```

## Notes

- Per-stream ~14 t/s at c=3 is comfortably usable for interactive multi-user.
- For **more** concurrency than 3, use the DCP4 lanes (`dcp4-cc200`, `dcp4-cc128`) —
  DCP4's 4× KV budget supports more simultaneous streams.
