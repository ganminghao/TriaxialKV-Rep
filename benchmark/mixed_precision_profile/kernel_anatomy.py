"""Compile-level anatomy of the TriAxialKV mixed decode kernel vs SGLang's bf16 kernel.

Registers, spills, shared memory, and SASS instruction mix -- all obtainable without
GPU performance counters.
"""
import collections, math, re, sys, torch
sys.path.insert(0, "/data/gmh/workspace/sglang-triaxialkv/python")
from sglang.srt.layers.attention.triton_ops import triaxial_kv as tk
from sglang.srt.layers.attention.triton_ops import decode_attention as da

G = 32
dev = "cuda"
bs, S, Hkv, D, group = 8, 8192, 8, 128, 8
Hq = Hkv * group
total = bs * S
f = 0.85
n2 = (int(S * f) // G) * G
n4 = S - n2
size2 = bs * n2 + G
size4 = bs * n4 + 1
gen = torch.Generator(device=dev).manual_seed(0)

q = torch.randn(bs, Hq, D, generator=gen, device=dev).to(torch.bfloat16)
o = torch.empty_like(q)
sm_scale = 1.0 / math.sqrt(D)
mks = 8
kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
kv_indptr[1:] = torch.cumsum(torch.full((bs,), S, device=dev), 0)
nks = torch.full((bs,), mks, dtype=torch.int32, device=dev)
attn_logits = torch.empty(bs, Hq, mks, D, dtype=torch.float32, device=dev)
attn_lse = torch.empty(bs, Hq, mks, dtype=torch.float32, device=dev)

kv = tk.TriaxialLayerKV(
    offset=size2,
    k2_data=torch.zeros(size2 // G, Hkv, D, 8, dtype=torch.uint8, device=dev),
    k2_meta=torch.ones(size2 // G, Hkv, D, 2, dtype=torch.float16, device=dev),
    v2_data=torch.zeros(size2, Hkv, D // 4, dtype=torch.uint8, device=dev),
    v2_meta=torch.ones(size2, Hkv, D // G, 2, dtype=torch.float16, device=dev),
    k4_data=torch.zeros(size4, Hkv, D // 2, dtype=torch.uint8, device=dev),
    k4_meta=torch.ones(size4, Hkv, D // G, 2, dtype=torch.float16, device=dev),
    v4_data=torch.zeros(size4, Hkv, D // 2, dtype=torch.uint8, device=dev),
    v4_meta=torch.ones(size4, Hkv, D // G, 2, dtype=torch.float16, device=dev),
)
idx = []
for b in range(bs):
    idx.append(torch.cat([G + b * n2 + torch.arange(n2, device=dev),
                          size2 + 1 + b * n4 + torch.arange(n4, device=dev)]))
kvi = torch.cat(idx).to(torch.int64)

tk.decode_attention_fwd_mixed(q, kv, o, kv_indptr, kvi, attn_logits, attn_lse, nks, mks, sm_scale)

kb = torch.zeros(total + 1, Hkv, D, dtype=torch.bfloat16, device=dev)
vb = torch.zeros_like(kb)
kvi_b = (torch.arange(total, device=dev) + 1).to(torch.int64)
da.decode_attention_fwd(q, kb, vb, o, kv_indptr, kvi_b, attn_logits, attn_lse, nks, mks, sm_scale, 1.0, 1.0)
torch.cuda.synchronize()


CATS = [
    ("global load  (LDG)", r"^LDG"),
    ("shared ld/st (LDS/STS)", r"^(LDS|STS)"),
    ("local  ld/st (LDL/STL = SPILL)", r"^(LDL|STL)"),
    ("tensor core  (HMMA/QMMA)", r"^(HMMA|QMMA|IMMA)"),
    ("fp32 fma/add/mul", r"^(FFMA|FADD|FMUL|FSEL|FSETP|MUFU)"),
    ("fp16x2 math", r"^(HFMA2|HADD2|HMUL2|HSET)"),
    ("int arith (IADD/IMAD)", r"^(IADD3|IMAD|IABS|ISETP)"),
    ("bit ops (SHF/LOP/PRMT/BMSK)", r"^(SHF|LOP3|PRMT|BMSK|POPC|FLO)"),
    ("convert (I2F/F2F/F2I)", r"^(I2F|F2F|F2I|I2I)"),
    ("select/move (SEL/MOV)", r"^(SEL|MOV|IMNMX|FMNMX)"),
    ("branch/sync", r"^(BRA|BSSY|BSYNC|BAR|EXIT|NOP|RET|CALL)"),
]


def mix(sass):
    ops = re.findall(r"^\s+/\*[0-9a-f]+\*/\s+@?!?P?\d?\s*([A-Z0-9_.]+)", sass, re.M)
    if not ops:
        ops = re.findall(r"\b([A-Z][A-Z0-9_]{2,})(?:\.[A-Z0-9_.]+)?\s", sass)
    ops = [o.split(".")[0] for o in ops]
    c = collections.Counter(ops)
    total_i = sum(c.values())
    rows = []
    seen = set()
    for label, pat in CATS:
        n = sum(v for k, v in c.items() if re.match(pat, k))
        for k in c:
            if re.match(pat, k):
                seen.add(k)
        rows.append((label, n))
    rows.append(("other", sum(v for k, v in c.items() if k not in seen)))
    return total_i, rows, c


def report(name, jitfn):
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    for key, k in jitfn.cache[0].items():
        md = k.metadata
        print(f"  num_warps={md.num_warps} num_stages={md.num_stages} "
              f"regs={md.num_regs} spills={md.num_spills} shared={md.shared}B "
              f"threads/CTA={md.num_warps*32}")
        occ_ctas = min(65536 // max(md.num_regs * md.num_warps * 32, 1), 32)
        print(f"  reg-limited CTAs/SM = {occ_ctas}  -> warps/SM = {occ_ctas*md.num_warps} "
              f"(max 64) = {100*occ_ctas*md.num_warps/64:.0f}% occupancy")
        try:
            sass = k.asm["sass"]
        except Exception as e:
            print("  (no sass:", e, ")"); return
        ti, rows, c = mix(sass)
        print(f"  SASS instructions in kernel body: {ti}")
        for label, n in rows:
            if n:
                print(f"      {label:34s} {n:6d}  {100*n/ti:5.1f}%")
        print("  top-15 opcodes:", ", ".join(f"{k}:{v}" for k, v in c.most_common(15)))
        break


report("TriAxialKV  _fwd_grouped_kernel_stage1_mixed (INT2/INT4)",
       tk._fwd_grouped_kernel_stage1_mixed)
report("SGLang      _fwd_grouped_kernel_stage1 (bf16)",
       da._fwd_grouped_kernel_stage1)
