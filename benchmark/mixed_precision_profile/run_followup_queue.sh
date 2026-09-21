#!/usr/bin/env bash
# Resumable job queue for the PROFILE_FOLLOWUP.md experiments.
# Re-run this script after any interruption: finished jobs (DONE marker) are skipped;
# a half-finished result directory is moved to results/_aborted/ (never overwritten).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=$HERE/results
M8=/tmp/models/Qwen3-8B
M32=/tmp/models/Qwen3-32B
LOG=$R/followup_logs/queue.log
mkdir -p "$R/followup_logs" "$R/_aborted"
log() { echo "$*" | tee -a "$LOG"; }

model_ready() {  # complete safetensors checkout: index present, every shard present, no aria2 partials
  local d=$1
  [ -f "$d/model.safetensors.index.json" ] || return 1
  ls "$d"/*.aria2 >/dev/null 2>&1 && return 1
  python3 - "$d" <<'PY' || return 1
import json, os, sys
d = sys.argv[1]
shards = set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values())
sys.exit(0 if all(os.path.exists(os.path.join(d, s)) for s in shards) else 1)
PY
}

# name | result dir (relative to results/) | env assignments | command
JOBS=(
"flag1_check|followup_points/check_bf16_W1_flag1|MODEL=$M8 MEM=0.88 PROFILE_FLAG=1|./bench_point.sh bf16 check_bf16_W1_flag1 1024 2048 128:128 0.9"
"bd_fp8_W1|followup_breakdown/fp8_W1|MODEL=$M8|./profile_step.sh fp8 W1"
"bd_tri_W1|followup_breakdown/tri_W1|MODEL=$M8 INT2_FRAC=0.05|./profile_step.sh tri W1"
"bd_bf16_W2|followup_breakdown/bf16_W2|MODEL=$M8|./profile_step.sh bf16 W2"
"bd_fp8_W2|followup_breakdown/fp8_W2|MODEL=$M8|./profile_step.sh fp8 W2"
"bd_tri_W2|followup_breakdown/tri_W2|MODEL=$M8 INT2_FRAC=0.05|./profile_step.sh tri W2"
"bd_bf16_W1mid|followup_breakdown/bf16_mid_W1|MODEL=$M8 TAG=_mid MID_TOKENS=262144|./profile_step.sh bf16 W1"
"bd_fp8_W1mid|followup_breakdown/fp8_mid_W1|MODEL=$M8 TAG=_mid MID_TOKENS=262144|./profile_step.sh fp8 W1"
"bd_tri_W1mid|followup_breakdown/tri_mid_W1|MODEL=$M8 TAG=_mid MID_TOKENS=262144 INT2_FRAC=0.05|./profile_step.sh tri W1"
"g3_sgl_bf16_W1|followup_points/g3_sglang_bf16_W1|MODEL=$M8 MEM=0.88|./bench_point.sh bf16 g3_sglang_bf16_W1 1024 2048 128:128 1.0"
"g3_sgl_bf16_W2|followup_points/g3_sglang_bf16_W2|MODEL=$M8 MEM=0.88|./bench_point.sh bf16 g3_sglang_bf16_W2 8000 256 32:64 1.0"
"g3_sgl_fp8_W1|followup_points/g3_sglang_fp8_W1|MODEL=$M8 MEM=0.88|./bench_point.sh fp8 g3_sglang_fp8_W1 1024 2048 128:128 1.0"
"g3_sgl_fp8_W2|followup_points/g3_sglang_fp8_W2|MODEL=$M8 MEM=0.88|./bench_point.sh fp8 g3_sglang_fp8_W2 8000 256 32:64 1.0"
"g3_vllm_bf16_W1|followup_vllm/vllm_bf16_W1|MODEL=$M8 MEM=0.88|./vllm_point.sh bf16 vllm_bf16_W1 1024 2048 128 128"
"g3_vllm_bf16_W2|followup_vllm/vllm_bf16_W2|MODEL=$M8 MEM=0.88|./vllm_point.sh bf16 vllm_bf16_W2 8000 256 32 64"
"g3m_vllm_bf16|followup_vllm/vllm_multi_bf16|MODEL=$M8|./vllm_multi.sh vllm vllm bf16 vllm_multi_bf16 2"
"g3m_vllm_fp8|followup_vllm/vllm_multi_fp8|MODEL=$M8|./vllm_multi.sh vllm vllm fp8 vllm_multi_fp8 2"
"g3m_sgl_bf16|followup_vllm/sglang_multi_bf16|MODEL=$M8|./vllm_multi.sh sglang sglang bf16 sglang_multi_bf16 3"
"g3m_sgl_fp8|followup_vllm/sglang_multi_fp8|MODEL=$M8|./vllm_multi.sh sglang sglang fp8 sglang_multi_fp8 3"
"g3x_sgl_bf16_vclient|followup_vllm/sglang_vclient_bf16|MODEL=$M8|./vllm_multi.sh sglang vllm bf16 sglang_vclient_bf16 2"
"q32_bf16_sweep|followup_points/q32_bf16_sweep|MODEL=$M32 MEM=0.90 EXTRA_ARGS=--cuda-graph-max-bs=64|./bench_point.sh bf16 q32_bf16_sweep 4096 1024 2:4,4:8,8:16,16:32,32:64,64:128 1.0"
"q32_fp8_sweep|followup_points/q32_fp8_sweep|MODEL=$M32 MEM=0.90 EXTRA_ARGS=--cuda-graph-max-bs=64|./bench_point.sh fp8 q32_fp8_sweep 4096 1024 2:4,4:8,8:16,16:32,32:64,64:128 1.0"
"q32_tri005_sweep|followup_points/q32_tri005_sweep|MODEL=$M32 MEM=0.90 INT2_FRAC=0.05 EXTRA_ARGS=--cuda-graph-max-bs=64|./bench_point.sh tri q32_tri005_sweep 4096 1024 2:4,4:8,8:16,16:32,32:64,64:128 1.0"
"q32_tri085_sweep|followup_points/q32_tri085_sweep|MODEL=$M32 MEM=0.90 INT2_FRAC=0.85 EXTRA_ARGS=--cuda-graph-max-bs=64|./bench_point.sh tri q32_tri085_sweep 4096 1024 2:4,4:8,8:16,16:32,32:64,64:128 1.0"
"q32_bf16_W2|followup_points/q32_bf16_W2|MODEL=$M32 MEM=0.90 EXTRA_ARGS=--cuda-graph-max-bs=64|./bench_point.sh bf16 q32_bf16_W2 8000 256 32:64 0.9"
"q32_fp8_W2|followup_points/q32_fp8_W2|MODEL=$M32 MEM=0.90 EXTRA_ARGS=--cuda-graph-max-bs=64|./bench_point.sh fp8 q32_fp8_W2 8000 256 32:64 0.9"
"q32_tri005_W2|followup_points/q32_tri005_W2|MODEL=$M32 MEM=0.90 INT2_FRAC=0.05 EXTRA_ARGS=--cuda-graph-max-bs=64|./bench_point.sh tri q32_tri005_W2 8000 256 32:64 0.9"
"q32_tri085_W2|followup_points/q32_tri085_W2|MODEL=$M32 MEM=0.90 INT2_FRAC=0.85 EXTRA_ARGS=--cuda-graph-max-bs=64|./bench_point.sh tri q32_tri085_W2 8000 256 32:64 0.9"
"q32_bd_bf16|followup_breakdown/bf16_q32_S4096|MODEL=$M32 MEM=0.90 TAG=_q32 EXTRA_ARGS=--cuda-graph-max-bs=64|./profile_step.sh bf16 S4096"
"q32_bd_fp8|followup_breakdown/fp8_q32_S4096|MODEL=$M32 MEM=0.90 TAG=_q32 EXTRA_ARGS=--cuda-graph-max-bs=64|./profile_step.sh fp8 S4096"
"q32_bd_tri|followup_breakdown/tri_q32_S4096|MODEL=$M32 MEM=0.90 INT2_FRAC=0.05 TAG=_q32 EXTRA_ARGS=--cuda-graph-max-bs=64|./profile_step.sh tri S4096"
)

log "[queue] start $(date '+%F %T') pid $$"
cd "$HERE"
for job in "${JOBS[@]}"; do
  IFS='|' read -r name rdir envs cmd <<< "$job"
  dir=$R/$rdir
  if [ -e "$dir/DONE" ]; then log "[queue] skip $name (done)"; continue; fi
  if [ -e "$dir" ]; then
    mv "$dir" "$R/_aborted/$(basename "$dir").$(date +%m%d%H%M%S)"
    log "[queue] moved incomplete $rdir to _aborted"
  fi
  if [[ "$envs" == *"MODEL=$M32"* ]]; then
    until model_ready "$M32"; do log "[queue] $(date +%T) waiting for $M32 download"; sleep 300; done
  fi
  log "[queue] $(date '+%F %T') run $name  (log: results/followup_logs/$name.log)"
  # word-split on purpose: env values and args contain no spaces (EXTRA_ARGS uses --flag=value, points are comma-separated)
  # shellcheck disable=SC2086
  env $envs $cmd > "$R/followup_logs/$name.log" 2>&1 < /dev/null
  rc=$?
  log "[queue] $(date '+%F %T') end $name rc=$rc done=$([ -e "$dir/DONE" ] && echo yes || echo no)"
done
log "[queue] finished $(date '+%F %T')"
/tmp/envs/sglang0510/bin/python "$HERE/summarize_followup.py" > "$R/followup_tables.md" 2> "$R/followup_logs/summarize.err" \
  && log "[queue] tables written to results/followup_tables.md"
