#!/bin/bash
# benchmark.sh — reproduce the README numbers against a running endpoint.
#
# Measures what actually matters, honestly:
#   • coherent real-world decode (5 diverse prompts)  ← the headline number
#   • random-token floor (tg512)                       ← pessimistic worst case
#   • cold prefill (novel uncacheable prompt)          ← no prefix-cache lies
#
# Why coherent AND random: random-token benches tank MTP acceptance and UNDERSTATE
# real decode by ~10-15%. Post the coherent number; keep random as an honest floor.
# Why "cold" prefill: vllm bench defaults to seed=0 → repeated inputs cache-hit →
# fake-fast prefill. This uses a unique prompt each run.
#
# EDIT these:
HEAD=192.168.NNN.11
CONTAINER=vllm_glm52          # the vLLM container name on the head node
MODEL=glm-5.2
TOK=/cache/huggingface/hub/glm52-int4-int8mix
API="http://$HEAD:8000"

echo "### 1/3  Coherent real-world decode (5 prompts, single-stream) ..."
python3 - "$HEAD" "$MODEL" <<'PY'
import urllib.request,json,time,sys
URL=f"http://{sys.argv[1]}:8000/v1/chat/completions"; MODEL=sys.argv[2]
P=[
 "Implement a production-quality async bounded MPMC channel in Rust with backpressure and graceful close; explain each atomic's memory ordering.",
 "Write a thread-safe Python LRU cache with TTL expiry and O(1) get/put, with docstrings; then walk through the eviction logic.",
 "Refactor a callback-based Node.js file pipeline into async/await with proper error propagation and backpressure; explain the tradeoffs.",
 "Explain how the Raft protocol achieves consensus: leader election, log replication, safety. Then sketch the state machine in pseudocode.",
 "Given a list of intervals, write an optimal algorithm to merge overlapping ones; prove correctness and analyze complexity.",
]
def one(p):
    b=json.dumps({"model":MODEL,"messages":[{"role":"user","content":p}],"max_tokens":700,"temperature":0.6}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json"})
    t=time.time(); j=json.loads(urllib.request.urlopen(r,timeout=220).read()); w=time.time()-t
    return j["usage"]["completion_tokens"]/max(w-1.0,0.01)   # subtract ~1s TTFT
xs=[one(p) for p in P]
print(f"    coherent decode: mean ~{sum(xs)/len(xs):.1f} tok/s  (range {min(xs):.1f}-{max(xs):.1f})")
PY

echo "### 2/3  Random-token floor (tg512, single-stream) ..."
docker exec "$CONTAINER" vllm bench serve --backend openai-chat --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions --model "$MODEL" --tokenizer "$TOK" \
  --dataset-name random --num-prompts 4 --max-concurrency 1 --random-input-len 256 --random-output-len 512 \
  2>&1 | awk '/Output token throughput/{printf "    random-token decode: ~%s tok/s\n",$NF}'

echo "### 3/3  Cold prefill (novel prompt, no cache) ..."
python3 - "$HEAD" "$MODEL" <<'PY'
import urllib.request,json,time,random,sys
URL=f"http://{sys.argv[1]}:8000/v1/chat/completions"; MODEL=sys.argv[2]
random.seed(time.time_ns())
words=" ".join(f"rec{random.randint(10000,99999)}" for _ in range(4500))   # ~unique, uncacheable
b=json.dumps({"model":MODEL,"messages":[{"role":"user","content":"Unique log; reply only OK.\n"+words}],"max_tokens":4,"temperature":0.0}).encode()
r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json"})
t=time.time(); j=json.loads(urllib.request.urlopen(r,timeout=120).read()); w=time.time()-t
pt=j["usage"]["prompt_tokens"]; ct=j["usage"]["completion_tokens"]
print(f"    cold prefill: ~{pt/max(w-ct/25.0,0.01):.0f} tok/s  ({pt}-token ingest)")
PY

echo
echo "Tool-calling grade (optional):"
echo "  uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git"
echo "  tool-eval-bench --base-url $API      # 0-100 → ★…★★★★★, auto-detects the model"
