# Sourced helper: make sure nobody else (e.g. the DiffKV profile) is on the GPU.
# nvidia-smi cannot see other containers' PIDs here, so we check used memory and
# known server command lines instead of the compute-apps list.
gpu_mem_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
# Processes that can hold the GPU: inference servers and any python in the DiffKV env,
# except package installs / extension builds (those never touch the GPU).
foreign_servers() {
  ps -eo pid,args --no-headers \
    | grep -E "sglang[.]launch_server|vllm[.]entrypoints|vllm serve|/tmp/envs/diffkv/bin/python" \
    | grep -vE "grep|pip |setup[.]py|ninja|-m build|shell-snapshots"
}
# wait_gpu_free [max_seconds]: returns 0 when <1 GiB is in use and no server process exists.
wait_gpu_free() {
  local max=${1:-21600} waited=0
  while :; do
    local used; used=$(gpu_mem_used)
    if [ "${used:-99999}" -lt 1024 ] && [ -z "$(foreign_servers)" ]; then
      echo "[gpu_guard] $(date +%T) GPU free (used ${used} MiB)"; return 0
    fi
    if [ $waited -eq 0 ] || [ $((waited % 300)) -eq 0 ]; then
      echo "[gpu_guard] $(date +%T) GPU busy (used ${used} MiB); waiting. procs:"; foreign_servers | cut -c1-160
    fi
    [ $waited -ge $max ] && { echo "[gpu_guard] gave up after ${max}s"; return 1; }
    sleep 30; waited=$((waited+30))
  done
}
# snapshot <file> <label>: record GPU memory + visible server processes for later audit
# The container has an 8-CPU cgroup quota shared with other agents, so CPU-side numbers
# (scheduler time, GPU idle, TTFT) depend on contention. Log the throttle counters too.
cpu_throttle() { awk '/nr_throttled|throttled_usec/{printf "%s=%s ", $1, $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null; }
gpu_snapshot() {
  { echo "== $2 $(date +%T) used_MiB=$(gpu_mem_used) $(cpu_throttle) load=$(cut -d' ' -f1 /proc/loadavg)"
    foreign_servers | cut -c1-200; } >> "$1"
}
