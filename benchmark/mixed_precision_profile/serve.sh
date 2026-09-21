#!/usr/bin/env bash
# Launch one SGLang server. Usage: serve.sh <bf16|fp8|tri>
# Env overrides (defaults reproduce PROFILE_FINDINGS.md sections 3-6):
#   MODEL, PORT, MEM (0.85), CTX (32768), CHUNK (8192), INT2_FRAC (0.5, tri only),
#   EXTRA_ARGS (appended verbatim), SGLANG_TRIAXIAL_PROFILE (timing ranges, default off)
set -u
MODE=$1; PORT=${PORT:-31000}
PY=/tmp/envs/sglang0510/bin/python
export CUDA_HOME=/tmp/envs/quancache TRITON_CACHE_DIR=/tmp/cache/triton_sglang HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case $MODE in
  bf16) MODE_ARGS="" ;;
  fp8)  MODE_ARGS="--kv-cache-dtype fp8_e4m3" ;;
  tri)  MODE_ARGS="--triaxial-kv --triaxial-int2-fraction ${INT2_FRAC:-0.5} --page-size 1" ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
exec $PY -m sglang.launch_server --model-path ${MODEL:-/data/gmh/model/Qwen3-8B} --host 127.0.0.1 --port $PORT \
  --context-length ${CTX:-32768} --mem-fraction-static ${MEM:-0.85} \
  --chunked-prefill-size ${CHUNK:-8192} --max-prefill-tokens ${CHUNK:-8192} \
  --disable-radix-cache $MODE_ARGS ${EXTRA_ARGS:-}
