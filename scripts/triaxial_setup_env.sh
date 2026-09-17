#!/usr/bin/env bash
# Create the conda environment for this TriAxialKV fork of SGLang v0.5.10.
#
#   ./scripts/triaxial_setup_env.sh [--prefetch] [--with-cuda-toolkit]
#
#   --prefetch           download the large wheels with aria2c first (8 MB/s) instead of
#                        letting pip fetch them (0.2 MB/s behind some proxies)
#   --with-cuda-toolkit  also install a CUDA toolkit into the env, so CUDA_HOME can point
#                        at the env itself (deep_gemm refuses to import without nvcc)
#
# Environment variables:
#   ENV_PREFIX  conda env path              (default /tmp/envs/sglang0510)
#   WHEEL_DIR   local wheel cache           (default /tmp/cache/wheels)
#   CONDA       conda executable            (default: conda on PATH, else /opt/miniforge3/bin/conda)
#   https_proxy / http_proxy  honoured as usual
#
# The env is deliberately separate from any other project env: SGLang v0.5.10 pins
# torch 2.9.1 / transformers 5.3.0 / flashinfer 0.6.7.post2 and will fight with newer stacks.
set -euo pipefail

PREFETCH=0
CUDA_TOOLKIT=0
for arg in "$@"; do
  case "$arg" in
    --prefetch) PREFETCH=1 ;;
    --with-cuda-toolkit) CUDA_TOOLKIT=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 1 ;;
  esac
done

ENV_PREFIX=${ENV_PREFIX:-/tmp/envs/sglang0510}
WHEEL_DIR=${WHEEL_DIR:-/tmp/cache/wheels}
CONDA=${CONDA:-$(command -v conda || echo /opt/miniforge3/bin/conda)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY="$ENV_PREFIX/bin/python"

export PIP_NO_CACHE_DIR=1          # pip's HTTP cache makes big-wheel downloads crawl
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-/tmp/cache/conda_pkgs}

echo "=== env      : $ENV_PREFIX"
echo "=== repo     : $REPO"
echo "=== wheel dir: $WHEEL_DIR"

# ---------------------------------------------------------------- 1. conda env
if [ ! -x "$PY" ]; then
  "$CONDA" create -y -p "$ENV_PREFIX" python=3.11 pip
fi

# libnuma.so.1 is dlopen'ed by sgl_kernel; without it common_ops fails to load.
"$CONDA" install -y -q -p "$ENV_PREFIX" -c conda-forge numactl

if [ "$CUDA_TOOLKIT" = 1 ]; then
  # ~2 GB. Afterwards CUDA_HOME="$ENV_PREFIX" works.
  "$CONDA" install -y -q -p "$ENV_PREFIX" -c nvidia cuda-toolkit=12.8
fi

"$PY" -m pip install -q --upgrade pip wheel setuptools setuptools-scm
"$PY" -m pip install -q pytest

# ------------------------------------------------------- 2. optional prefetch
if [ "$PREFETCH" = 1 ]; then
  command -v aria2c >/dev/null || { echo "aria2c not found; drop --prefetch" >&2; exit 1; }
  mkdir -p "$WHEEL_DIR"
  "$PY" "$REPO/scripts/triaxial_prefetch_wheels.py" --out-dir "$WHEEL_DIR"
  if [ -s "$WHEEL_DIR/urls.txt" ]; then
    ( cd "$WHEEL_DIR" && aria2c --file-allocation=none -x 8 -s 8 -j 4 -k 8M -c \
        --max-tries=0 --retry-wait=5 --timeout=60 --console-log-level=warn -i urls.txt )
  fi
fi

# ------------------------------------------------------------- 3. install deps
FIND_LINKS=()
if compgen -G "$WHEEL_DIR/*.whl" >/dev/null; then
  FIND_LINKS=(--find-links "$WHEEL_DIR")
  # Install the prefetched wheels without dependency resolution first: pip would
  # otherwise re-download them while resolving.
  "$PY" -m pip install --no-index --no-deps "$WHEEL_DIR"/*.whl
fi

"$PY" -m pip install "${FIND_LINKS[@]}" --prefer-binary -e "$REPO/python"

# --------------------------------------------------------------- 4. env script
CUDA_HOME_GUESS=""
for c in "$ENV_PREFIX" /usr/local/cuda /tmp/envs/quancache; do
  [ -x "$c/bin/nvcc" ] && { CUDA_HOME_GUESS=$c; break; }
done
if [ -z "$CUDA_HOME_GUESS" ] && command -v nvcc >/dev/null; then
  CUDA_HOME_GUESS=$(dirname "$(dirname "$(command -v nvcc)")")
fi

cat > "$ENV_PREFIX/triaxial_env.sh" <<EOF
# Source this before launching a server or running the benchmarks.
# Every value yields to one already set in the environment.
export PY=\${PY:-$PY}
export CUDA_HOME=\${CUDA_HOME:-${CUDA_HOME_GUESS:-/usr/local/cuda}}
export TRITON_CACHE_DIR=\${TRITON_CACHE_DIR:-/tmp/cache/triton_sglang}
export FLASHINFER_WORKSPACE_BASE=\${FLASHINFER_WORKSPACE_BASE:-/tmp/cache/flashinfer}
export HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=\${TRANSFORMERS_OFFLINE:-1}
EOF
echo "=== wrote $ENV_PREFIX/triaxial_env.sh (CUDA_HOME=${CUDA_HOME_GUESS:-NOT FOUND})"
if [ -z "$CUDA_HOME_GUESS" ]; then
  echo "!!! no nvcc found. deep_gemm asserts on import without a valid CUDA_HOME."
  echo "!!! rerun with --with-cuda-toolkit, or edit $ENV_PREFIX/triaxial_env.sh by hand."
fi

# ------------------------------------------------------------------ 5. verify
CUDA_HOME=${CUDA_HOME_GUESS:-/usr/local/cuda} "$PY" - <<'PYEOF'
import torch, sglang, flashinfer, sgl_kernel
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| sglang", sglang.__version__, "| flashinfer", flashinfer.__version__)
print("gpu", torch.cuda.get_device_name() if torch.cuda.is_available() else "NONE")
import sglang.srt.mem_cache.triaxial_pool          # noqa: F401
import sglang.srt.managers.triaxial_tagger         # noqa: F401
import sglang.srt.layers.attention.triaxial_backend  # noqa: F401
print("triaxial modules import OK")
PYEOF
echo "=== SETUP DONE"
