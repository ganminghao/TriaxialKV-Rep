"""Split one inference step into P1..I from a torch.profiler trace (PROFILE_FINDINGS.md section 9).

Attribution of every GPU event (kernel / memcpy / memset):
  1. Follow its CUPTI correlation id to the CPU call that launched it, and take the innermost
     enclosing SGLANG_TRIAXIAL_PROFILE range on that thread:
       P1:: -> P1g (GPU work launched by CPU-side preparation: metadata, H2D/D2H copies)
       P2:: -> P2   P3:: -> P3   P4:: -> P4   K1:: -> K1
  2. If the innermost range is STEP:: (e.g. everything replayed from a CUDA graph, where no
     Python runs), fall back to kernel-name rules (NAME_RULES).
  3. Everything else is M (linear layers, norms, RoPE, activation, sampling).

Step windows are cut on the GPU timeline: step i runs from the first GPU event launched inside
CPU range STEP::<mode> #i to the first one of #i+1. Idle I = window length - union of busy
intervals. P1 is reported as CPU milliseconds per step (sum of top-level P1:: ranges plus the
P1:: ranges nested in STEP::), and is not added to the wall-clock view because with SGLang's
overlap scheduler the CPU prepares step i+1 while the GPU runs step i.
"""
import argparse, bisect, collections, glob, gzip, json, os, re, statistics, sys

NAME_RULES = [
    # attention compute
    ("K1", re.compile(r"flash::|FlashAttn|_fwd_grouped_kernel_stage1|_fwd_kernel_stage2|"
                      r"BatchPrefill|BatchDecode|flashinfer::.*(prefill|decode)|prefill_.*kernel")),
    # KV write / quantisation
    ("P3", re.compile(r"store_kvcache|_quant_int4_tokens_kernel|_quant_int2_v_tokens_kernel|"
                      r"_quant_int2_k_pages_kernel")),
    # dequantise into a bf16 scratch
    ("P4", re.compile(r"_dequant_gather_kernel")),
]
# Kernels replayed from a CUDA graph have no Python range. Where a generic kernel name is shared
# between categories, split it by per-layer counts measured against the bf16 trace
# (see PROFILE_FINDINGS.md section 9.2):
#  fp8: float8_copy_kernel runs 3x per layer = casts of k, v (KV write) and q (FA3 input)
#  tri: set_kv_buffer_int4 adds loc-offset (add<long>), clamp_min, and one extra .contiguous()
#       copy per layer on top of the one copy per layer that bf16 also has
GRAPH_SPLIT = {
    "fp8": [(re.compile(r"float8_copy_kernel"), {"P3": 2 / 3, "K1": 1 / 3})],
    "tri": [(re.compile(r"CUDAFunctorOnSelf_add<long>|launch_clamp_scalar"), {"P3": 1.0}),
            (re.compile(r"elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel"),
             {"P3": 0.5, "M": 0.5})],
}
TAG = {"P1": "P1g", "P2": "P2", "P3": "P3", "P4": "P4", "K1": "K1"}
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
ORDER = ["P1g", "P2", "P3", "P4", "K1", "M"]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    return json.load(op(path))["traceEvents"]


def analyze(path, mode, skip_first=2, kv_mode="bf16"):
    ev = load(path)
    ann = collections.defaultdict(list)   # (pid,tid) -> [(ts, end, name)]
    runtime = {}                          # correlation -> (pid, tid, ts)
    syncs = collections.defaultdict(list) # (pid,tid) -> [(ts, end)] CPU blocked waiting for the GPU
    gpu = []
    for e in ev:
        c = e.get("cat")
        if c == "cuda_runtime" and "Synchronize" in e["name"]:
            syncs[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"]))
        if c == "user_annotation" and re.match(r"(P[1-4]|K1|STEP)::", e["name"]):
            ann[(e["pid"], e["tid"])].append((e["ts"], e["ts"] + e["dur"], e["name"]))
        elif c in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            runtime[e["args"]["correlation"]] = (e["pid"], e["tid"], e["ts"])
        elif c in GPU_CATS:
            gpu.append(e)
    for k in ann:
        ann[k].sort()

    def enclosing(pid, tid, ts):
        """innermost annotation containing ts, and the enclosing STEP (index into steps)"""
        lst = ann.get((pid, tid), [])
        i = bisect.bisect_right(lst, (ts, float("inf"), "~"))
        inner, step = None, None
        for j in range(i - 1, -1, -1):
            s, t, n = lst[j]
            if s <= ts <= t:
                if inner is None or s >= inner[0]:
                    inner = inner or (s, t, n)
                    if s > inner[0]:
                        inner = (s, t, n)
                if n.startswith("STEP::") and step is None:
                    step = (s, t, n)
            if t < ts - 120e6:  # annotations are short; stop scanning far back
                break
        # innermost = the containing range with the latest start
        cands = [x for x in lst[max(0, i - 400):i] if x[0] <= ts <= x[1]]
        inner = max(cands, key=lambda x: x[0]) if cands else None
        steps = [x for x in cands if x[2].startswith("STEP::")]
        return inner, (steps[0] if steps else None)

    step_name = f"STEP::{mode}"
    cpu_steps = sorted({x for lst in ann.values() for x in lst if x[2] == step_name})
    step_start_gpu = {}
    rows = []
    unattributed = 0
    name_hits = collections.Counter()
    for e in gpu:
        corr = e.get("args", {}).get("correlation")
        r = runtime.get(corr)
        cat, st = "M", None
        split = None
        if r is None:
            unattributed += 1
        else:
            inner, st = enclosing(*r)
            if inner is not None and not inner[2].startswith("STEP::"):
                cat = TAG[inner[2][:2]]
            else:
                for c, rx in NAME_RULES:
                    if rx.search(e["name"]):
                        cat = c
                        break
                else:
                    for rx, w in GRAPH_SPLIT.get(kv_mode, []):
                        if rx.search(e["name"]):
                            split = w
                            break
        if st is not None:
            step_start_gpu[st] = min(step_start_gpu.get(st, float("inf")), e["ts"])
        if split is None:
            split = {cat: 1.0}
        rows.append((e["ts"], e["ts"] + e["dur"], split, e["name"], st[2] if st is not None else None))
    rows.sort()

    # every step of any kind, in GPU order; a window for <mode> ends where the next step of
    # any kind starts, so a prefill squeezed between two decodes is not charged to the decode
    all_steps = sorted((t, st[2]) for st, t in step_start_gpu.items())
    windows = [(a[0], b[0]) for a, b in zip(all_steps[:-1], all_steps[1:]) if a[1] == step_name]
    res_other_steps = sum(1 for x in all_steps if x[1] != step_name)
    windows = windows[skip_first:]
    per_step = []
    dropped = 0
    for a, b in windows:
        i0 = bisect.bisect_left(rows, (a,))
        i1 = bisect.bisect_left(rows, (b,))
        # a window is clean only if every GPU event in it was launched by a step of this kind
        # (with the overlap scheduler, kernels of a neighbouring prefill can land inside a decode window)
        if any(r[4] is not None and r[4] != step_name for r in rows[i0:i1]):
            dropped += 1
            continue
        acc = collections.Counter()
        busy, cur_s, cur_e = 0.0, None, None
        for s, t, split, nm, _k in rows[i0:i1]:
            t = min(t, b)
            for c, w in split.items():
                acc[c] += (t - s) * w
                name_hits[(c, nm[:90])] += (t - s) * w
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = s, t
            else:
                cur_e = max(cur_e, t)
        if cur_e is not None:
            busy += cur_e - cur_s
        acc["_window"] = b - a
        acc["_busy"] = busy
        acc["I"] = (b - a) - busy
        per_step.append(acc)

    # CPU-side P1 per CPU step window
    cpu_p1 = []
    main = max(ann, key=lambda k: sum(1 for x in ann[k] if x[2] == step_name)) if ann else None
    if main is not None:
        lst = ann[main]
        steps = sorted(x for x in lst if x[2].startswith("STEP::"))
        pairs = [(a, b) for a, b in zip(steps[:-1], steps[1:]) if a[2] == step_name]
        for (s0, e0, _), (s1, _, _) in pairs[skip_first:]:
            top = [x for x in lst if s0 <= x[0] < s1 and x[2].startswith("P1::")]
            # count only P1 ranges not nested inside another P1 range
            tot = 0.0
            wait = 0.0
            comp = collections.Counter()
            sl = syncs.get(main, [])
            for x in top:
                nested = any(y is not x and y[0] <= x[0] and x[1] <= y[1] and y[2].startswith("P1::") for y in top)
                if not nested:
                    w = sum(min(t, x[1]) - max(q, x[0]) for q, t in sl if q < x[1] and t > x[0])
                    tot += x[1] - x[0] - w
                    wait += w
                    comp[x[2]] += x[1] - x[0] - w
            launch = (e0 - s0) - sum(x[1] - x[0] for x in top if s0 <= x[0] and x[1] <= e0)
            comp["STEP launch (CPU, excl. nested P1)"] = launch
            comp["_cpu_window"] = s1 - s0
            comp["_p1_total"] = tot
            comp["_p1_gpu_wait"] = wait
            cpu_p1.append(comp)

    def mean(key, lst):
        return statistics.mean(x.get(key, 0.0) for x in lst) / 1e3 if lst else float("nan")

    res = {"trace": path, "mode": mode, "steps": len(per_step), "other_steps_in_trace": res_other_steps,
           "dropped_mixed_windows": dropped,
           "unattributed_gpu_events": unattributed,
           "ms": {k: mean(k, per_step) for k in ORDER + ["I", "_window", "_busy"]},
           "cpu_ms": {k: mean(k, cpu_p1) for k in sorted({k for c in cpu_p1 for k in c})},
           "top_kernels": [(c, n, d / max(1, len(per_step)) / 1e3) for (c, n), d in
                           sorted(name_hits.items(), key=lambda x: -x[1])[:25]]}
    return res


def report(res):
    ms = res["ms"]
    gpu_sum = sum(ms[k] for k in ORDER)
    print(f"\n### {res.get('name', '')} {res['mode']}  {os.path.relpath(res['trace'])}")
    print(f"steps analysed: {res['steps']} (other-kind steps in trace: {res['other_steps_in_trace']}, "
          f"windows dropped because another step's kernels overlapped: {res['dropped_mixed_windows']})   "
          f"GPU events without a launching CPU call: {res['unattributed_gpu_events']}")
    print(f"step window {ms['_window']:.2f} ms   GPU busy (union) {ms['_busy']:.2f} ms   "
          f"sum of GPU event time {gpu_sum:.2f} ms")
    print(f"{'category':<40}{'ms/step':>9}{'GPU %':>8}{'wall %':>8}")
    labels = {"P1g": "P1g GPU work launched by CPU prep", "P2": "P2 GPU memory management",
              "P3": "P3 KV write + quantisation", "P4": "P4 dequant into bf16 scratch",
              "K1": "K1 attention compute", "M": "M rest of model", "I": "I GPU idle"}
    wall = gpu_sum + ms["I"]
    for k in ORDER + ["I"]:
        g = f"{100 * ms[k] / gpu_sum:7.1f}%" if k != "I" else "      -"
        print(f"{labels[k]:<40}{ms[k]:9.2f}{g}{100 * ms[k] / wall:7.1f}%")
    c = res["cpu_ms"]
    print(f"CPU: P1 CPU work {c.get('_p1_total', float('nan')):.2f} ms per step, plus "
          f"{c.get('_p1_gpu_wait', float('nan')):.2f} ms blocked waiting for the GPU inside P1 ranges "
          f"(CPU step period {c.get('_cpu_window', float('nan')):.2f} ms)")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        if not k.startswith("_"):
            print(f"    {k:<45}{v:8.2f} ms (CPU work, GPU waits removed)")
    print("top GPU events per step:")
    for cat, n, d in res["top_kernels"][:14]:
        print(f"    {cat:<4}{d:8.3f} ms  {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dirs", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = []
    for d in a.result_dirs:
        name = os.path.basename(d.rstrip("/"))
        wl_mode = "extend" if name.split("_")[-1].startswith("W2") else "decode"
        for f in sorted(glob.glob(os.path.join(d, "trace", "*.trace.json.gz"))):
            kv_mode = name.split("_")[0]
            r = analyze(f, wl_mode, kv_mode=kv_mode)
            r["name"] = name
            report(r)
            out.append(r)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
