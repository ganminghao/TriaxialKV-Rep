#!/usr/bin/env bash
# One server, several benchmark runs, for the engine-baseline comparison (PROFILE_FINDINGS.md section 11).
# Usage: vllm_multi.sh <engine: vllm|sglang> <client: vllm|sglang> <kv: bf16|fp8> <name> <reps>
#   runs W1 (1024/2048, conc 128, 128 req) and W2 (8000/256, conc 32, 64 req) <reps> times each,
#   fixed lengths (vLLM client --random-range-ratio 0.0, SGLang client 1.0)
# vLLM is used READ-ONLY from /data/gmh/workspace/vllm; outputs go to results/followup_vllm/<name>/.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/gpu_guard.sh"
ENGINE=$1; CLIENT=$2; KV=$3; NAME=$4; REPS=${5:-2}
MODEL=${MODEL:-/tmp/models/Qwen3-8B}
PORT=31000
OUT=$HERE/results/followup_vllm/$NAME
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT"; exit 1; fi
mkdir -p "$OUT"; AUDIT=$OUT/gpu_audit.txt
VLLM=/tmp/envs/quancache/bin/vllm
SGPY=/tmp/envs/sglang0510/bin/python
vllm_env() {
  [ -f /tmp/quancache_local.env ] && source /tmp/quancache_local.env
  export CUDA_HOME=/tmp/envs/quancache PATH=/tmp/envs/quancache/bin:$PATH
}
wait_gpu_free || exit 1
gpu_snapshot "$AUDIT" before_launch
cd "$OUT"
if [ "$ENGINE" = vllm ]; then
  ( vllm_env
    KVARGS=""; [ "$KV" = fp8 ] && KVARGS="--kv-cache-dtype fp8"
    exec $VLLM serve "$MODEL" --host 127.0.0.1 --port $PORT --max-model-len 32768 \
      --gpu-memory-utilization 0.88 --max-num-batched-tokens 8192 --max-num-seqs 256 \
      --no-enable-prefix-caching $KVARGS ) > "$OUT/server.log" 2>&1 &
  HEALTH=/health
else
  MODEL="$MODEL" MEM=0.88 "$HERE/serve.sh" "$KV" > "$OUT/server.log" 2>&1 &
  HEALTH=/health_generate
fi
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for i in $(seq 1 360); do
  curl -s -m 2 http://127.0.0.1:$PORT$HEALTH >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -30 "$OUT/server.log"; exit 1; }
  sleep 5
done
echo "[vllm_multi] $ENGINE server up $(date +%T)"
run_bench() {  # <label> <in> <out> <conc> <nreq>
  local L=$1 IN=$2 OU=$3 C=$4 N=$5
  gpu_snapshot "$AUDIT" "start_$L"
  if [ "$CLIENT" = vllm ]; then
    ( vllm_env
      BACKEND=vllm; [ "$ENGINE" = sglang ] && BACKEND=openai
      $VLLM bench serve --backend $BACKEND --base-url http://127.0.0.1:$PORT --endpoint /v1/completions \
        --model "$MODEL" --dataset-name random --random-input-len $IN --random-output-len $OU \
        --random-range-ratio 0.0 --num-prompts $N --max-concurrency $C --ignore-eos \
        --save-result --result-dir "$OUT" --result-filename "$L.json" ) > "$OUT/$L.log" 2>&1
  else
    $SGPY -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $PORT --model "$MODEL" \
      --dataset-name random-ids --random-input-len $IN --random-output-len $OU --random-range-ratio 1.0 \
      --num-prompts $N --max-concurrency $C > "$OUT/$L.log" 2>&1
  fi
  gpu_snapshot "$AUDIT" "end_$L"
  g() { grep -oE "$1 +[0-9.]+" "$OUT/$L.log" | grep -oE "[0-9.]+$" | head -1; }
  printf "%-8s %-6s %-6s %-4s %-8s out=%s tot=%s tpot=%s ttft=%s ok=%s in=%s gen=%s\n" "$ENGINE" "$CLIENT" "$KV" "$L" "" \
    "$(g 'Output token throughput \(tok/s\):')" "$(g 'Total token throughput \(tok/s\):')" "$(g 'Mean TPOT \(ms\):')" \
    "$(g 'Mean TTFT \(ms\):')" "$(g 'Successful requests:')" "$(g 'Total input tokens:')" "$(g 'Total generated tokens:')" | tee -a "$OUT/summary.txt"
}
for r in $(seq 1 $REPS); do
  run_bench "W1_r$r" 1024 2048 128 128
  run_bench "W2_r$r" 8000 256 32 64
done
echo "[vllm_multi] done $(date +%T)"; touch "$OUT/DONE"
