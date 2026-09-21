#!/usr/bin/env bash
# Launch one server, sweep concurrency against it.
# Usage: sweep_conc.sh <mode> <in_len> <out_len> <conc_list csv>
set -uo pipefail
MODE=${1:-bf16}; INLEN=${2:-4096}; OUTLEN=${3:-1024}; CONCS=${4:-16,32,64,128,256}
PY=/tmp/envs/sglang0510/bin/python
MODEL=${MODEL:-/data/gmh/model/Qwen3-8B}
PORT=${PORT:-31000}
SP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${RESULTS:-$SP_DIR/results}/sweep/${MODE}${TAG:-}_i${INLEN}_o${OUTLEN}
mkdir -p "$OUT"

MODEL="$MODEL" INT2_FRAC=${INT2_FRAC:-0.05} "$SP_DIR/serve.sh" $MODE > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT
for i in $(seq 1 90); do
  curl -s -m 2 http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -20 "$OUT/server.log"; exit 1; }
  sleep 5
done
grep -E "KV Cache is allocated|TriAxialKV pool" "$OUT/server.log" | tail -2
CAP=$(grep -oE "#tokens: [0-9]+" "$OUT/server.log" | head -1 | grep -oE "[0-9]+")
echo "mode=$MODE kv_tokens=$CAP  seq_per_req=$((INLEN+OUTLEN))  kv-limited concurrency ~= $((CAP/(INLEN+OUTLEN)))"
printf "%-5s %8s %10s %10s %9s %9s %9s\n" mode conc "out_tok/s" "tot_tok/s" "TPOT_ms" "TTFT_ms" "run_req"
for C in $(echo $CONCS | tr ',' ' '); do
  NP=$((C*2)); [ $NP -gt 256 ] && NP=256
  L="$OUT/c$C.log"
  curl -s -m 10 -X POST http://127.0.0.1:$PORT/flush_cache >/dev/null 2>&1
  MARK=$(wc -l < "$OUT/server.log")
  timeout 1800 $PY -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $PORT \
    --model "$MODEL" --dataset-name random-ids --random-input-len $INLEN \
    --random-output-len $OUTLEN --random-range-ratio 1.0 \
    --num-prompts $NP --max-concurrency $C > "$L" 2>&1
  ot=$(grep -oE "Output token throughput \(tok/s\): +[0-9.]+" "$L" | grep -oE "[0-9.]+$")
  tt=$(grep -oE "Total token throughput \(tok/s\): +[0-9.]+" "$L" | grep -oE "[0-9.]+$")
  tp=$(grep -oE "Mean TPOT \(ms\): +[0-9.]+" "$L" | grep -oE "[0-9.]+$")
  tf=$(grep -oE "Mean TTFT \(ms\): +[0-9.]+" "$L" | grep -oE "[0-9.]+$")
  rr=$(tail -n +$MARK "$OUT/server.log" | grep -oE "#running-req: [0-9]+" | awk '{s+=$2;n++} END{if(n)printf "%.0f",s/n; else printf "-"}')
  if [ -z "${ot:-}" ]; then
    if grep -qiE "out of memory|Scheduler hit an exception" "$OUT/server.log"; then echo "$MODE $C  CRASH(OOM)"; break; fi
    printf "%-5s %8s %10s\n" "$MODE" "$C" "FAIL"; continue
  fi
  printf "%-5s %8s %10s %10s %9s %9s %9s\n" "$MODE" "$C" "$ot" "$tt" "$tp" "$tf" "${rr:--}"
done
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
