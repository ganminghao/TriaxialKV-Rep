#!/usr/bin/env bash
# Latest-vLLM baseline point (gap 3). Uses /data/gmh/workspace/vllm READ-ONLY:
# nothing is written inside that repo; results go to our results/followup_vllm/.
# Usage: vllm_point.sh <bf16|fp8> <name> <in_len> <out_len> <conc> <num_prompts>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/gpu_guard.sh"
KV=$1; NAME=$2; INLEN=$3; OUTLEN=$4; C=$5; NP=$6
MODEL=${MODEL:-/data/gmh/model/Qwen3-8B}
PORT=${VPORT:-32000}
OUT=$HERE/results/followup_vllm/$NAME
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT"; exit 1; fi
mkdir -p "$OUT"; AUDIT=$OUT/gpu_audit.txt
[ -f /tmp/quancache_local.env ] && source /tmp/quancache_local.env
# vLLM's engine JIT-compiles kernels at startup and asserts on a CUDA toolchain; the env file
# does not set one. The quancache env ships nvcc for CUDA 13, matching vLLM's torch cu130 build.
export CUDA_HOME=${CUDA_HOME:-/tmp/envs/quancache}
export PATH=$CUDA_HOME/bin:$PATH
VLLM=/tmp/envs/quancache/bin/vllm
case $KV in
  bf16) KVARGS="" ;;
  fp8)  KVARGS="--kv-cache-dtype fp8" ;;
  *) echo "unknown kv $KV"; exit 2 ;;
esac
wait_gpu_free || exit 1
gpu_snapshot "$AUDIT" before_launch
cd "$OUT"   # keep any stray files out of the vllm repo
# match the SGLang side: same context length, memory fraction, prefill chunk, no prefix cache
$VLLM serve "$MODEL" --host 127.0.0.1 --port $PORT --max-model-len 32768 \
  --gpu-memory-utilization ${MEM:-0.88} --max-num-batched-tokens 8192 --max-num-seqs 256 \
  --no-enable-prefix-caching $KVARGS > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for i in $(seq 1 360); do
  curl -s -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -30 "$OUT/server.log"; exit 1; }
  sleep 5
done
echo "[vllm_point] server up $(date +%T)"
$VLLM --version > "$OUT/vllm_version.txt" 2>&1
gpu_snapshot "$AUDIT" start_bench
$VLLM bench serve --backend vllm --base-url http://127.0.0.1:$PORT --model "$MODEL" \
  --dataset-name random --random-input-len $INLEN --random-output-len $OUTLEN --random-range-ratio 0.0 \
  --num-prompts $NP --max-concurrency $C --ignore-eos \
  --save-result --result-dir "$OUT" --result-filename bench.json > "$OUT/bench.log" 2>&1
gpu_snapshot "$AUDIT" end_bench
grep -E "Successful requests|Output token throughput|Total token throughput|Mean TPOT|Mean TTFT|Total input tokens|Total generated tokens" "$OUT/bench.log" | tee "$OUT/summary.txt"
grep -iE "Using .*(attention|backend)|attn_backend|KV cache|GPU KV cache size|Maximum concurrency|cudagraph|CUDA graph" "$OUT/server.log" | head -12 >> "$OUT/summary.txt"
echo "[vllm_point] done $(date +%T)"; touch "$OUT/DONE"
