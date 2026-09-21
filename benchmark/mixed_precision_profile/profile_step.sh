#!/usr/bin/env bash
# Capture a torch.profiler trace of steady decode (W1) or prefill (W2) for one mode.
# Usage: profile_step.sh <bf16|fp8|tri> <W1|W2>
# Env: MODEL, MEM (0.88 = same as run_e2e.sh / report section 6), INT2_FRAC, EXTRA_ARGS, MID_TOKENS,
#      TAG (suffix for the result dir), STEPS (forward passes to record)
# Output: results/followup_breakdown/<mode><TAG>_<W>/  (refuses to overwrite)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/gpu_guard.sh"
MODE=$1; WL=$2
PY=/tmp/envs/sglang0510/bin/python
MODEL=${MODEL:-/data/gmh/model/Qwen3-8B}
PORT=${PORT:-31000}
case $WL in
  W1) INLEN=1024; OUTLEN=2048; CONC=${CONC:-128}; NP=${NP:-128}; STEPS=${STEPS:-60} ;;
  W2) INLEN=8000; OUTLEN=256;  CONC=${CONC:-32};  NP=${NP:-64};  STEPS=${STEPS:-40} ;;
  # capacity-bound decode (Qwen3-32B scenario): requests keep queueing, prefills interleave
  S4096) INLEN=4096; OUTLEN=1024; CONC=${CONC:-32}; NP=${NP:-64}; STEPS=${STEPS:-120} ;;
  *) echo "unknown workload $WL"; exit 2 ;;
esac
OUT=$HERE/results/followup_breakdown/${MODE}${TAG:-}_${WL}
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT"; exit 1; fi
mkdir -p "$OUT/trace"
AUDIT=$OUT/gpu_audit.txt

wait_gpu_free || exit 1
gpu_snapshot "$AUDIT" before_launch

SGLANG_TRIAXIAL_PROFILE=1 SGLANG_TORCH_PROFILER_DIR="$OUT/trace" MODEL="$MODEL" \
  MEM=${MEM:-0.88} INT2_FRAC=${INT2_FRAC:-0.05} "$HERE/serve.sh" "$MODE" > "$OUT/server.log" 2>&1 &
SPID=$!
BPID=""
cleanup() { [ -n "$BPID" ] && kill $BPID 2>/dev/null; kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 360); do
  curl -s -m 2 http://127.0.0.1:$PORT/health_generate >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER DIED"; tail -30 "$OUT/server.log"; exit 1; }
  sleep 5
done
echo "[profile_step] server up $(date +%T)"

bench() {  # bench <num_prompts> <concurrency> <log>
  $PY -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $PORT --model "$MODEL" \
    --dataset-name random-ids --random-input-len $INLEN --random-output-len $OUTLEN \
    --random-range-ratio 0.9 --num-prompts $1 --max-concurrency $2 > "$3" 2>&1
}
start_profile() {
  curl -s -m 30 -X POST http://127.0.0.1:$PORT/start_profile -H 'Content-Type: application/json' \
    -d "{\"output_dir\": \"$OUT/trace\", \"num_steps\": $STEPS, \"activities\": [\"CPU\", \"GPU\"], \"with_stack\": false, \"record_shapes\": false, \"profile_id\": \"${MODE}_${WL}\"}"
  echo; echo "[profile_step] start_profile sent $(date +%T)"
}

if [ "$WL" = W2 ]; then
  echo "[profile_step] warm-up: 8 prefill-shaped requests"
  bench 8 8 "$OUT/warmup.log"
  sleep 3
  bench $NP $CONC "$OUT/bench.log" & BPID=$!
  # bench_serving first sends one warm-up request and decodes it; start the capture only when
  # the main burst begins, whose first forwards are 8k-token prefills (decodes may interleave;
  # the analyzer keeps prefill windows only)
  until grep -q "Starting main benchmark run" "$OUT/bench.log" 2>/dev/null; do
    kill -0 $BPID 2>/dev/null || break; sleep 0.2
  done
  start_profile
elif [ "$WL" = S4096 ]; then
  bench $NP $CONC "$OUT/bench.log" & BPID=$!
  # KV capacity caps the batch far below CONC, so "running >= CONC-4" never happens.
  # Wait for the first decode, then let the queue settle for SETTLE seconds.
  for i in $(seq 1 1800); do
    grep -q "Decode batch" "$OUT/server.log" && break
    kill -0 $BPID 2>/dev/null || break
    sleep 1
  done
  sleep ${SETTLE:-60}
  echo "[profile_step] settled: $(grep -E 'Decode batch' "$OUT/server.log" | tail -1)"
  start_profile
else
  bench $NP $CONC "$OUT/bench.log" & BPID=$!
  # wait for steady decode: a Decode log line with >= CONC-4 running after the last Prefill line
  for i in $(seq 1 600); do
    last=$(grep -nE "Decode batch|Prefill batch" "$OUT/server.log" | tail -1)
    if echo "$last" | grep -q "Decode batch"; then
      rr=$(echo "$last" | grep -oE "#running-req: [0-9]+" | grep -oE "[0-9]+")
      [ "${rr:-0}" -ge $((CONC-4)) ] && break
    fi
    kill -0 $BPID 2>/dev/null || { echo "bench ended before steady decode"; break; }
    sleep 1
  done
  echo "[profile_step] steady decode reached: $last"
  # MID_TOKENS: keep decoding until the batch holds this many KV tokens, so the capture sees the
  # average KV length of the run (W1: 128 requests x 2048 = 262144) instead of the start of decode
  if [ -n "${MID_TOKENS:-}" ]; then
    for i in $(seq 1 1800); do
      tok=$(grep "Decode batch" "$OUT/server.log" | tail -1 | grep -oE "#token: [0-9]+" | grep -oE "[0-9]+")
      [ "${tok:-0}" -ge "$MID_TOKENS" ] && break
      kill -0 $BPID 2>/dev/null || { echo "bench ended before MID_TOKENS"; break; }
      sleep 1
    done
    echo "[profile_step] mid-decode reached: $(grep 'Decode batch' "$OUT/server.log" | tail -1)"
  fi
  start_profile
fi
gpu_snapshot "$AUDIT" during_profile

for i in $(seq 1 600); do
  ls "$OUT"/trace/*.trace.json.gz >/dev/null 2>&1 && grep -q "Profiling done" "$OUT/server.log" && break
  sleep 1
done
ls -la "$OUT"/trace/ | tail -3
wait $BPID; BPID=""
gpu_snapshot "$AUDIT" after_bench
grep -E "Output token throughput|Mean TPOT|Mean TTFT|Successful requests" "$OUT/bench.log"
echo "[profile_step] done $(date +%T) -> $OUT"
touch "$OUT/DONE"
