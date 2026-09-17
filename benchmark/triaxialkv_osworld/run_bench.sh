#!/bin/bash
# TriAxialKV throughput reproduction on the OSWorld replay (SGLang v0.5.10 fork).
#
#   ./run_bench.sh <mode> [num_prompts] [max_concurrency]
#     mode: bf16      SGLang bf16 KV, prefill=flashinfer, decode=triton   (paper's "Triton" baseline)
#           bf16-fi   SGLang bf16 KV, flashinfer for both                (reference)
#           fp8       SGLang fp8_e4m3 KV, flashinfer for both            (reference)
#           tri       TriAxialKV INT2/INT4 (prefill=flashinfer over dequantized prefix, decode=fused triton)
#
# Server and client run on the same GPU. Results go to results/<mode>_<timestamp>/.
# Override MODEL / REPLAY / PORT / MEM_FRAC / CHUNK / INT2_FRAC / POLICY via the environment.
# Never edit this file while a run is live (bash reads scripts lazily).
set -euo pipefail

MODE=${1:?mode}
NUM_PROMPTS=${2:-600}
MAX_CONC=${3:-128}

# Pick up PY / CUDA_HOME / cache dirs written by scripts/triaxial_setup_env.sh.
ENV_PREFIX=${ENV_PREFIX:-/tmp/envs/sglang0510}
if [ -f "$ENV_PREFIX/triaxial_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$ENV_PREFIX/triaxial_env.sh"
fi

PY=${PY:-$ENV_PREFIX/bin/python}
MODEL=${MODEL:-/tmp/models/Qwen3-VL-32B-Instruct}
REPLAY=${REPLAY:-/tmp/datasets/osworld_trajs/replay_15step_h4.jsonl}
PORT=${PORT:-30000}
CTX=${CTX:-32768}
MEM_FRAC=${MEM_FRAC:-0.90}
CHUNK=${CHUNK:-8192}
INT2_FRAC=${INT2_FRAC:-0.85}
POLICY=${POLICY:-default}
HERE=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$HERE/results/${MODE}_${STAMP}
mkdir -p "$OUT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/cache/triton_sglang}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/tmp/cache/flashinfer}
export SGLANG_DISABLE_MARLIN=1

COMMON=(--model-path "$MODEL" --port "$PORT" --host 127.0.0.1
        --context-length "$CTX" --mem-fraction-static "$MEM_FRAC"
        --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$CHUNK" --max-running-requests 256
        --page-size 1 --trust-remote-code --log-level info)

case "$MODE" in
  bf16)    ARGS=(--prefill-attention-backend flashinfer --decode-attention-backend triton) ;;
  bf16-fi) ARGS=(--attention-backend flashinfer) ;;
  fp8)     ARGS=(--attention-backend flashinfer --kv-cache-dtype fp8_e4m3) ;;
  tri)     ARGS=(--triaxial-kv --triaxial-int2-fraction "$INT2_FRAC" --triaxial-policy "$POLICY") ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac

echo "[run_bench] mode=$MODE out=$OUT" | tee "$OUT/info.txt"
echo "server args: ${COMMON[*]} ${ARGS[*]}" | tee -a "$OUT/info.txt"

$PY -m sglang.launch_server "${COMMON[@]}" "${ARGS[@]}" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'echo "[run_bench] stopping server $SERVER_PID"; kill $SERVER_PID 2>/dev/null; sleep 3; kill -9 $SERVER_PID 2>/dev/null || true' EXIT

# wait for readiness (model load + cuda graph capture can take several minutes)
for i in $(seq 1 240); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died, see $OUT/server.log"; exit 1; fi
  sleep 5
done
curl -s "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json" || true
grep -E "KV Cache is allocated|TriAxialKV pool|max_total_num_tokens|Capture cuda graph" "$OUT/server.log" | tee -a "$OUT/info.txt"

$PY -m sglang.bench_serving --backend sglang-oai-chat --host 127.0.0.1 --port "$PORT" \
    --dataset-name openai --dataset-path "$REPLAY" --num-prompts "$NUM_PROMPTS" \
    --request-rate inf --max-concurrency "$MAX_CONC" --warmup-requests 2 \
    --tokenizer "$MODEL" --output-file "$OUT/bench.jsonl" 2>&1 | tee "$OUT/bench.log"

grep -E "Successful requests|Benchmark duration|Total input|Total generated|Request throughput|Input token throughput|Output token throughput|Total token throughput|Concurrency|Mean E2E|Median E2E|Mean TTFT|Mean TPOT" "$OUT/bench.log" | tee -a "$OUT/info.txt"
echo "[run_bench] done: $OUT"
