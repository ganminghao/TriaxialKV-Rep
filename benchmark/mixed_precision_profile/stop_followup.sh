#!/usr/bin/env bash
# Stop the follow-up queue and any server/benchmark it started. Matches exact process prefixes,
# so it never kills the calling shell. Safe to run when nothing is running.
kill_where() { ps -eo pid,args --no-headers | awk "$1 {print \$1}" | xargs -r kill 2>/dev/null; }
kill_where '$2=="bash" && ($3=="./run_followup_queue.sh" || $3=="./bench_point.sh" || $3=="./profile_step.sh" || $3=="./vllm_multi.sh" || $3=="./vllm_point.sh")'
sleep 2
kill_where '($3=="-m" && ($4=="sglang.bench_serving" || $4=="sglang.launch_server")) || $3=="/tmp/envs/quancache/bin/vllm" || $2=="VLLM::EngineCore"'
for i in $(seq 1 30); do
  n=$(ps -eo args --no-headers | awk '$3=="-m" && $4=="sglang.launch_server"' | wc -l)
  [ "$n" -eq 0 ] && break; sleep 1
done
echo "stopped; GPU memory used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
