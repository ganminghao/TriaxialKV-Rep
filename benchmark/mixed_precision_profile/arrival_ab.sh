#!/usr/bin/env bash
# Does SGLANG_TRIAXIAL_PROFILE=1 change how W1 requests are admitted? Alternating flag off/on,
# fresh server each time; each run stops once steady decode (>=124 running) is reached.
# Output: results/followup_checks/arrival_ab/run<i>_flag<f>/ and summary.tsv
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/gpu_guard.sh"
PY=/tmp/envs/sglang0510/bin/python
MODEL=${MODEL:-/tmp/models/Qwen3-8B}
BASE=$HERE/results/followup_checks/arrival_ab
mkdir -p "$BASE"
[ -f "$BASE/summary.tsv" ] || printf "run\tflag\tprefill_batches\tmax_new_seq\tsec_to_steady\tthrottled_during\tload\n" > "$BASE/summary.tsv"
for i in ${RUNS:-1 2 3 4 5 6}; do
  f=$(( i % 2 ))
  OUT=$BASE/run${i}_flag${f}
  [ -e "$OUT/DONE" ] && continue
  rm -rf "$OUT"; mkdir -p "$OUT"
  wait_gpu_free 3600 || exit 1
  SGLANG_TRIAXIAL_PROFILE=$f MODEL=$MODEL MEM=0.88 "$HERE/serve.sh" bf16 > "$OUT/server.log" 2>&1 &
  SPID=$!
  for k in $(seq 1 360); do curl -s -m 2 http://127.0.0.1:31000/health_generate >/dev/null 2>&1 && break; sleep 5; done
  t0=$(awk '/nr_throttled/{print $2}' /sys/fs/cgroup/cpu.stat)
  $PY -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 31000 --model "$MODEL" \
    --dataset-name random-ids --random-input-len 1024 --random-output-len 2048 --random-range-ratio 0.9 \
    --num-prompts 128 --max-concurrency 128 > "$OUT/bench.log" 2>&1 &
  BP=$!
  until grep -q "Starting main benchmark run" "$OUT/bench.log" 2>/dev/null; do sleep 0.2; done
  s=$(date +%s.%N)
  for k in $(seq 1 600); do
    rr=$(grep "Decode batch" "$OUT/server.log" | tail -1 | grep -oE "#running-req: [0-9]+" | grep -oE "[0-9]+")
    [ "${rr:-0}" -ge 124 ] && break
    sleep 0.5
  done
  e=$(date +%s.%N)
  t1=$(awk '/nr_throttled/{print $2}' /sys/fs/cgroup/cpu.stat)
  kill $BP 2>/dev/null; kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; wait $BP 2>/dev/null
  nb=$(grep 'Prefill batch' "$OUT/server.log" | grep -oE '#new-token: [0-9]+' | awk '$2>=500{n++} END{print n+0}')
  mx=$(grep -oE '#new-seq: [0-9]+' "$OUT/server.log" | awk '{if($2>m)m=$2} END{print m+0}')
  printf "%s\t%s\t%s\t%s\t%.1f\t%s\t%s\n" "$i" "$f" "$nb" "$mx" "$(awk -v a="$s" -v b="$e" 'BEGIN{print b-a}')" "$((t1-t0))" "$(cut -d' ' -f1 /proc/loadavg)" >> "$BASE/summary.tsv"
  touch "$OUT/DONE"
  sleep 10
done
