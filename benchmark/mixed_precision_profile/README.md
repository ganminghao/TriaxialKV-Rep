# Mixed-precision KV cache profiling

Scripts behind `../../PROFILE_FINDINGS.md`. See section 8 of that document for the
exact commands. `MODEL` defaults to `/data/gmh/model/Qwen3-8B` and can be overridden.

| file | what it measures |
| --- | --- |
| `bench_kv_decode_roofline.py` | decode attention across KV formats (fa3-bf16 / fa3-fp8 / triton-bf16 / TriAxialKV), us/call, effective GB/s, % of HBM peak |
| `kernel_anatomy.py` | compiles both stage-1 kernels into a fresh Triton cache so `cuobjdump -res-usage` / `nvdisasm` can report registers and SASS instruction mix |
| `bench_prefill_path.py` | `dequant_gather` vs bf16 gather vs the quant write path, plus chunked-prefill re-dequantisation amplification |
| `serve.sh` | launches one SGLang server in `bf16` / `fp8` / `tri` mode |
| `sweep_conc.sh` | one server launch, sweeps concurrency against it (throughput-vs-capacity curve) |
| `run_e2e.sh` | single-point end-to-end serving benchmark, one server per call |
| `roofline.json` | raw output of the full `bench_kv_decode_roofline.py` sweep |

## Follow-up experiments (report sections 9-13)

Run everything with the resumable queue (rerun the same command after any interruption):

```bash
./stop_followup.sh && ./run_followup_queue.sh
```

| file | what it does |
| --- | --- |
| `run_followup_queue.sh` | job queue; skips finished jobs, moves partial ones to `results/_aborted/` |
| `profile_step.sh` | server with `SGLANG_TRIAXIAL_PROFILE=1`, load, `/start_profile` capture (W1 decode, W2 prefill, S4096 capacity-bound decode) |
| `analyze_trace.py` | splits one step into P1g/P2/P3/P4/K1/M/I (rules in report section 9.2) |
| `bench_point.sh` | throughput points with the timing ranges off |
| `vllm_multi.sh` | several W1/W2 runs on one server; server and client each SGLang or vLLM |
| `vllm_point.sh` | single vLLM point |
| `arrival_ab.sh` | checks that the timing flag does not change request admission |
| `gpu_guard.sh` | waits for a free GPU, logs GPU memory and cgroup CPU throttling |
| `stop_followup.sh` | stops the queue and every server it started |
| `summarize_followup.py` | writes `results/followup_tables.md` |
