#!/usr/bin/env bash
# Download the public OSWorld trajectory archive and rebuild the replay request file.
#
#   ./scripts/triaxial_fetch_data.sh
#
# Environment variables:
#   DATA_DIR    where the archive and the replay land (default /tmp/datasets/osworld_trajs)
#   PY          python with `pillow` and `transformers` (default $ENV_PREFIX or /tmp/envs/sglang0510)
#   MODEL       tokenizer dir used for the token statistics (default /tmp/models/Qwen3-VL-32B-Instruct)
#   HF_ENDPOINT hugging face mirror (default https://hf-mirror.com; use https://huggingface.co direct)
#
# Output:
#   $DATA_DIR/qwen2.5-vl-32b-instruct_15step.zip   2.7 GB, 369 tasks, 2494 recorded steps
#   $DATA_DIR/replay_15step_h4.jsonl               ~6 GB, 2473 chat requests
#   $DATA_DIR/replay_15step_h4.stats.json          per-request token statistics
#   $DATA_DIR/replay_smoke_20.jsonl                first 20 requests, for quick checks
#
# The archive is one OSWorld-Verified evaluation run published by the OSWorld authors
# (dataset xlangai/ubuntu_osworld_verified_trajs, MIT licence). It holds one screenshot
# per step plus the model's raw output, which is everything the replay needs.
set -euo pipefail

DATA_DIR=${DATA_DIR:-/tmp/datasets/osworld_trajs}
PY=${PY:-${ENV_PREFIX:-/tmp/envs/sglang0510}/bin/python}
MODEL=${MODEL:-/tmp/models/Qwen3-VL-32B-Instruct}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
REPO=$(cd "$(dirname "$0")/.." && pwd)
BENCH="$REPO/benchmark/triaxialkv_osworld"

ZIP_NAME=qwen2.5-vl-32b-instruct_15step.zip
URL="$HF_ENDPOINT/datasets/xlangai/ubuntu_osworld_verified_trajs/resolve/main/$ZIP_NAME"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# ------------------------------------------------------------------- download
if [ ! -f "$ZIP_NAME" ]; then
  echo "=== downloading $ZIP_NAME from $HF_ENDPOINT"
  if command -v aria2c >/dev/null; then
    aria2c --file-allocation=none -x 10 -s 10 -k 4M -c --max-tries=0 --retry-wait=10 \
           --timeout=60 --console-log-level=warn -o "$ZIP_NAME" "$URL"
  else
    curl -L -C - -o "$ZIP_NAME" "$URL"
  fi
fi

echo "=== verifying archive"
"$PY" - "$ZIP_NAME" <<'PYEOF'
import sys, zipfile
zf = zipfile.ZipFile(sys.argv[1])
names = zf.namelist()
tasks = {n.rsplit("/", 1)[0] for n in names if n.endswith("traj.jsonl")}
print(f"entries {len(names)}, tasks {len(tasks)}")
assert len(tasks) == 369, f"expected 369 tasks, found {len(tasks)}"
PYEOF

# --------------------------------------------------------------- build replay
echo "=== building replay requests"
"$PY" "$BENCH/build_replay.py" \
  --zip "$DATA_DIR/$ZIP_NAME" \
  --instructions "$BENCH/tasks_instructions.json" \
  --out "$DATA_DIR/replay_15step_h4.jsonl" \
  --stats "$DATA_DIR/replay_15step_h4.stats.json" \
  --tokenizer "$MODEL"

head -n 20 "$DATA_DIR/replay_15step_h4.jsonl" > "$DATA_DIR/replay_smoke_20.jsonl"
ls -la "$DATA_DIR"
echo "=== DATA READY"
