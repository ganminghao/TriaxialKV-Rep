#!/usr/bin/env bash
# Usage: run_e2e.sh <mode> <num_prompts> <concurrency> [in_len] [out_len]
# modes: bf16 | fp8 | tri
set -uo pipefail
MODE=${1:-bf16}; NP=${2:-96}; CONC=${3:-64}; INLEN=${4:-8000}; OUTLEN=${5:-256}
PY=/tmp/envs/sglang0510/bin/python
MODEL=${MODEL:-/data/gmh/model/Qwen3-8B}
PORT=${PORT:-31000}
MEM=${MEM:-0.88}
CTX=${CTX:-32768}
CHUNK=${CHUNK:-8192}
SP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-$SP_DIR/results/e2e}
export CUDA_HOME=/tmp/envs/quancache
export TRITON_CACHE_DIR=/tmp/cache/triton_sglang
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

RUN=$OUT/${MODE}_c${CONC}_i${INLEN}_o${OUTLEN}
mkdir -p "$RUN"

case $MODE in
  bf16) EXTRA="" ;;
  fp8)  EXTRA="--kv-cache-dtype fp8_e4m3" ;;
  tri)  EXTRA="--triaxial-kv --triaxial-int2-fraction ${INT2_FRAC:-0.85} --page-size 1" ;;
  *) echo "bad mode"; exit 1 ;;
esac

echo "=== launching $MODE on port $PORT ==="
$PY -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port $PORT \
  --context-length $CTX --mem-fraction-static $MEM \
  --chunked-prefill-size $CHUNK --max-prefill-tokens $CHUNK \
  --disable-radix-cache \
  $EXTRA > "$RUN/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT

for i in $(seq 1 180); do
  if curl -s -m 2 http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1; then break; fi
  if ! kill -0 $SPID 2>/dev/null; then echo "SERVER DIED"; tail -40 "$RUN/server.log"; exit 1; fi
  sleep 5
done
if ! curl -s -m 5 http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1; then
  echo "TIMEOUT"; tail -40 "$RUN/server.log"; exit 1
fi
echo "=== server up ==="
grep -iE "KV Cache is allocated|max_total_num_tokens|TriAxialKV" "$RUN/server.log" | tail -5 | tee "$RUN/capacity.txt"

$PY -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $PORT \
  --model "$MODEL" --dataset-name random-ids --random-input-len $INLEN \
  --random-output-len $OUTLEN --random-range-ratio 0.9 \
  --num-prompts $NP --max-concurrency $CONC \
  --output-file "$RUN/bench.jsonl" 2>&1 | tee "$RUN/bench.log" | tail -35

grep -oE "#running-req: [0-9]+" "$RUN/server.log" | awk '{s+=$2;n++; if($2>m)m=$2} END{printf "avg running-req %.1f  max %d  (n=%d)\n", s/n, m, n}' | tee -a "$RUN/capacity.txt"
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
echo "=== done $MODE -> $RUN ==="
