#!/usr/bin/env bash
# One server launch, one or more SGLang benchmark points, timing ranges OFF.
# Usage: bench_point.sh <bf16|fp8|tri> <outdir-name> <in_len> <out_len> <conc:num_prompts>[,<conc:num_prompts>...] [range_ratio]
#   e.g. bench_point.sh bf16 qwen8b_bf16_W1 1024 2048 "128:128" 0.9
# Env: MODEL, MEM, CTX, CHUNK, INT2_FRAC, EXTRA_ARGS, PROFILE_FLAG (default 0)
# Output: results/followup_points/<outdir-name>/  (refuses to overwrite)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/gpu_guard.sh"
MODE=$1; NAME=$2; INLEN=$3; OUTLEN=$4; POINTS=$5; RR=${6:-1.0}
PY=/tmp/envs/sglang0510/bin/python
MODEL=${MODEL:-/data/gmh/model/Qwen3-8B}
PORT=${PORT:-31000}
OUT=$HERE/results/followup_points/$NAME
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT"; exit 1; fi
mkdir -p "$OUT"; AUDIT=$OUT/gpu_audit.txt
wait_gpu_free || exit 1
gpu_snapshot "$AUDIT" before_launch
SGLANG_TRIAXIAL_PROFILE=${PROFILE_FLAG:-0} MODEL="$MODEL" "$HERE/serve.sh" "$MODE" > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for i in $(seq 1 360); do
  curl -s -m 2 http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED at startup"; tail -30 "$OUT/server.log"; echo CRASH_STARTUP > "$OUT/STATUS"; exit 1; }
  sleep 5
done
CAP=$(grep -oE "#tokens: [0-9]+" "$OUT/server.log" | head -1 | grep -oE "[0-9]+")
GRAPH=$(grep -oE "Capture cuda graph end" "$OUT/server.log" | head -1)
echo "mode=$MODE model=$(basename $MODEL) kv_tokens=$CAP cuda_graph=${GRAPH:+yes} $(grep -oE 'TriAxialKV pool: .*' "$OUT/server.log" | head -1)" | tee "$OUT/summary.txt"
printf "%-6s %6s %6s %10s %10s %9s %10s %8s %s\n" mode conc nreq "out_tok/s" "tot_tok/s" "TPOT_ms" "TTFT_ms" run_req status | tee -a "$OUT/summary.txt"
for P in ${POINTS//,/ }; do
  C=${P%%:*}; NP=${P##*:}
  L="$OUT/c${C}.log"
  MARK=$(wc -l < "$OUT/server.log")
  gpu_snapshot "$AUDIT" "start_c$C"
  timeout 7200 $PY -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --dataset-name random-ids --random-input-len $INLEN \
    --random-output-len $OUTLEN --random-range-ratio $RR \
    --num-prompts $NP --max-concurrency $C > "$L" 2>&1
  gpu_snapshot "$AUDIT" "end_c$C"
  g() { grep -oE "$1 +[0-9.]+" "$L" | grep -oE "[0-9.]+$" | head -1; }
  ot=$(g "Output token throughput \(tok/s\):"); tt=$(g "Total token throughput \(tok/s\):")
  tp=$(g "Mean TPOT \(ms\):"); tf=$(g "Mean TTFT \(ms\):")
  rr=$(tail -n +$MARK "$OUT/server.log" | grep "Decode batch" | grep -oE "#running-req: [0-9]+" | awk '{s+=$2;n++} END{if(n)printf "%.1f",s/n; else printf "-"}')
  st=ok
  if [ -z "${ot:-}" ]; then
    st=FAIL
    tail -n +$MARK "$OUT/server.log" | grep -qiE "out of memory|Scheduler hit an exception" && st=CRASH
  fi
  printf "%-6s %6s %6s %10s %10s %9s %10s %8s %s\n" "$MODE" "$C" "$NP" "${ot:--}" "${tt:--}" "${tp:--}" "${tf:--}" "$rr" "$st" | tee -a "$OUT/summary.txt"
  if [ "$st" = CRASH ]; then
    tail -n +$MARK "$OUT/server.log" | grep -iE "out of memory|Error|exception" | head -5 | tee -a "$OUT/summary.txt"
    break
  fi
done
echo "[bench_point] done $(date +%T)"; touch "$OUT/DONE"
