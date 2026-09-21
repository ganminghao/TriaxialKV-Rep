"""Decode-attention roofline across KV formats on one H100.

Compares, at identical (bs, seq_len, head geometry):
  fi-bf16      FlashInfer paged decode, bf16 KV            (production baseline)
  fi-fp8       FlashInfer paged decode, fp8_e4m3 KV        (production 2x baseline)
  triton-bf16  SGLang Triton decode_attention_fwd, bf16 KV (TriAxialKV's own baseline)
  tri-mixed    TriAxialKV fused INT2/INT4 kernel at int2_fraction f
  tri-int4     same kernel, all slots INT4
  tri-int2     same kernel, all slots INT2

Reports us/call, KV payload bytes, effective GB/s, and % of H100 HBM peak.
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/data/gmh/workspace/sglang-triaxialkv/python")
from sglang.srt.layers.attention.triton_ops import triaxial_kv as tk
from sglang.srt.layers.attention.triton_ops.decode_attention import decode_attention_fwd
from sgl_kernel.flash_attn import flash_attn_with_kvcache

HBM_PEAK_GBS = 3350.0  # H100 SXM HBM3
G = 32


def num_kv_splits_like_sglang(seq_lens, num_head, num_kv_head, max_kv_splits, core_count):
    seq_lens = seq_lens.to(torch.int64)
    num_seq = seq_lens.numel()
    max_seq_len = int(seq_lens.max()); min_seq_len = int(seq_lens.min())
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = min(-(-max_seq_len // min_seq_len), max_kv_splits)
    kv_chunk_size_1 = -(-max_seq_len // max_kv_splits_1)
    ext_seq_len = float(max_seq_len) / 64.0
    ext_device_core_count = int(core_count * max(math.log2(ext_seq_len), 1.0))
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_head
    else:
        block_h = min(block_h, num_kv_group)
        token_grid = num_seq * (-(-num_head // block_h))
    max_kv_splits_2 = min(-(-ext_device_core_count // token_grid), max_kv_splits)
    kv_chunk_size_2 = -(-max_seq_len // max_kv_splits_2)
    n1 = -(-seq_lens // kv_chunk_size_1); n2 = -(-seq_lens // kv_chunk_size_2)
    return torch.maximum(n1, n2).to(torch.int32)


def make_mixed_layer(H, D, size2, size4, gen, dev):
    num_pages2 = max(size2 // G, 1)
    kv = tk.TriaxialLayerKV(
        offset=size2,
        k2_data=torch.zeros(num_pages2, H, D, 8, dtype=torch.uint8, device=dev),
        k2_meta=torch.ones(num_pages2, H, D, 2, dtype=torch.float16, device=dev),
        v2_data=torch.zeros(max(size2, 1), H, D // 4, dtype=torch.uint8, device=dev),
        v2_meta=torch.ones(max(size2, 1), H, D // G, 2, dtype=torch.float16, device=dev),
        k4_data=torch.zeros(max(size4, 1), H, D // 2, dtype=torch.uint8, device=dev),
        k4_meta=torch.ones(max(size4, 1), H, D // G, 2, dtype=torch.float16, device=dev),
        v4_data=torch.zeros(max(size4, 1), H, D // 2, dtype=torch.uint8, device=dev),
        v4_meta=torch.ones(max(size4, 1), H, D // G, 2, dtype=torch.float16, device=dev),
    )
    # fill with pseudo-random bytes; exact values don't matter for timing
    for t in (kv.k2_data, kv.v2_data, kv.k4_data, kv.v4_data):
        t.random_(0, 256, generator=gen)
    for t in (kv.k2_meta, kv.v2_meta, kv.k4_meta, kv.v4_meta):
        t.uniform_(0.01, 0.1, generator=gen)
    return kv


def bench(fn, n_layers, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1e3 / n_layers)
    return best  # us per layer-call


def run_case(bs, S, Hkv, D, group, int2_frac, n_pools, n_layers, dev, gen, want):
    Hq = Hkv * group
    total = bs * S
    core_count = torch.cuda.get_device_properties(0).multi_processor_count
    seq_lens = torch.full((bs,), S, device=dev)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    kv_indptr[1:] = torch.cumsum(torch.full((bs,), S, device=dev), 0)
    max_kv_splits = 8
    nks = num_kv_splits_like_sglang(seq_lens, Hq, Hkv, max_kv_splits, core_count)
    q = torch.randn(bs, Hq, D, generator=gen, device=dev).to(torch.bfloat16)
    o = torch.empty_like(q)
    attn_logits = torch.empty(bs, Hq, max_kv_splits, D, dtype=torch.float32, device=dev)
    attn_lse = torch.empty(bs, Hq, max_kv_splits, dtype=torch.float32, device=dev)
    sm_scale = 1.0 / math.sqrt(D)
    out = {}

    free0, _ = torch.cuda.mem_get_info()

    # ---------- FA3 (sgl_kernel flash_attn, page_size=1) bf16 / fp8 ----------
    for name, dt, bpe in (("fa3-bf16", torch.bfloat16, 2), ("fa3-fp8", torch.float8_e4m3fn, 1)):
        if name not in want:
            continue
        per_layer = 2 * (total + 1) * Hkv * D * bpe
        free, _ = torch.cuda.mem_get_info()
        np_ = max(1, min(n_pools, int(free * 0.55 // per_layer)))
        try:
            ks = [torch.zeros(total + 1, 1, Hkv, D, dtype=dt, device=dev) for _ in range(np_)]
            vs = [torch.zeros(total + 1, 1, Hkv, D, dtype=dt, device=dev) for _ in range(np_)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        page_table = (torch.arange(total, dtype=torch.int32, device=dev) + 1).view(bs, S)
        cache_seqlens = torch.full((bs,), S, dtype=torch.int32, device=dev)
        q3 = q.unsqueeze(1)  # [bs, 1, Hq, D]
        desc = None
        if bpe == 1:  # fp8 KV: SGLang casts q to fp8 too and passes descales
            q3 = q3.to(torch.float8_e4m3fn)
            desc = torch.ones(bs, Hkv, dtype=torch.float32, device=dev)
        try:
            def fn(np_=np_, ks=ks, vs=vs, desc=desc):
                for i in range(n_layers):
                    flash_attn_with_kvcache(
                        q3, ks[i % np_], vs[i % np_], page_table=page_table,
                        cache_seqlens=cache_seqlens, softmax_scale=sm_scale, causal=True,
                        k_descale=desc, v_descale=desc)
            us = bench(fn, n_layers)
            payload = total * Hkv * D * 2 * bpe
            out[name] = dict(us=us, bytes=payload, gbs=payload / us / 1e3, pools=np_)
        except Exception as ex:
            out[name] = dict(err=repr(ex)[:200])
        del ks, vs
        torch.cuda.empty_cache()

    # ---------- SGLang Triton bf16 ----------
    if "triton-bf16" in want:
        per_layer = 2 * (total + 1) * Hkv * D * 2
        free, _ = torch.cuda.mem_get_info()
        np_ = max(1, min(n_pools, int(free * 0.55 // per_layer)))
        ks = [torch.randn(total + 1, Hkv, D, generator=gen, device=dev).to(torch.bfloat16) for _ in range(np_)]
        vs = [torch.randn(total + 1, Hkv, D, generator=gen, device=dev).to(torch.bfloat16) for _ in range(np_)]
        kvi = (torch.arange(total, device=dev) + 1).to(torch.int64)
        fn = lambda: [decode_attention_fwd(q, ks[i % np_], vs[i % np_], o, kv_indptr, kvi,
                                           attn_logits, attn_lse, nks, max_kv_splits, sm_scale, 1.0, 1.0)
                      for i in range(n_layers)]
        us = bench(fn, n_layers)
        payload = total * Hkv * D * 2 * 2
        out["triton-bf16"] = dict(us=us, bytes=payload, gbs=payload / us / 1e3, pools=np_)
        del ks, vs; torch.cuda.empty_cache()

    # ---------- TriAxialKV mixed / int4-only / int2-only ----------
    for name, f in (("tri-int4", 0.0), ("tri-mixed", int2_frac), ("tri-int2", 1.0)):
        if name not in want:
            continue
        n2 = (int(S * f) // G) * G
        n4 = S - n2
        size2 = bs * n2 + G
        size4 = bs * n4 + 1
        per_layer = (size2 // G) * Hkv * D * 8 + (size2 // G) * Hkv * D * 4 \
            + size2 * Hkv * (D // 4) + size2 * Hkv * (D // G) * 4 \
            + 2 * (size4 * Hkv * (D // 2) + size4 * Hkv * (D // G) * 4)
        free, _ = torch.cuda.mem_get_info()
        np_ = max(1, min(n_pools, int(free * 0.55 // max(per_layer, 1))))
        try:
            layers = [make_mixed_layer(Hkv, D, size2, size4, gen, dev) for _ in range(np_)]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); continue
        idx = []
        for b in range(bs):
            parts = []
            if n2: parts.append(G + b * n2 + torch.arange(n2, device=dev))
            if n4: parts.append(size2 + 1 + b * n4 + torch.arange(n4, device=dev))
            idx.append(torch.cat(parts))
        kvi = torch.cat(idx).to(torch.int64)
        fn = lambda: [tk.decode_attention_fwd_mixed(q, layers[i % np_], o, kv_indptr, kvi,
                                                    attn_logits, attn_lse, nks, max_kv_splits, sm_scale)
                      for i in range(n_layers)]
        us = bench(fn, n_layers)
        payload = bs * n2 * Hkv * 96 + bs * n4 * Hkv * 160
        out[name] = dict(us=us, bytes=payload, gbs=payload / us / 1e3, pools=np_)
        del layers; torch.cuda.empty_cache()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Hkv", type=int, default=8)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--int2-fraction", type=float, default=0.85)
    ap.add_argument("--bs-list", type=str, default="1,4,16,32,64")
    ap.add_argument("--seq-list", type=str, default="4096,16384")
    ap.add_argument("--pools", type=int, default=8)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--variants", type=str,
                    default="fa3-bf16,fa3-fp8,triton-bf16,tri-int4,tri-mixed,tri-int2")
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()
    dev = "cuda"; torch.cuda.set_device(0)
    gen = torch.Generator(device=dev).manual_seed(0)
    want = set(a.variants.split(","))
    print(f"# {torch.cuda.get_device_name()}  Hkv={a.Hkv} D={a.D} gqa_group={a.group} "
          f"int2_frac={a.int2_fraction}  HBM peak {HBM_PEAK_GBS} GB/s")
    hdr = f"{'bs':>4} {'seq':>7} {'variant':>12} {'us/call':>9} {'MB/call':>9} {'GB/s':>8} {'%peak':>7} {'vs fa3-bf16':>11}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for S in [int(x) for x in a.seq_list.split(",")]:
        for bs in [int(x) for x in a.bs_list.split(",")]:
            if bs * S > 1_600_000:
                continue
            r = run_case(bs, S, a.Hkv, a.D, a.group, a.int2_fraction, a.pools, a.layers, dev, gen, want)
            base = r.get("fa3-bf16", {}).get("us")
            for k in ("fa3-bf16", "fa3-fp8", "triton-bf16", "tri-int4", "tri-mixed", "tri-int2"):
                if k not in r: continue
                v = r[k]
                if "err" in v:
                    print(f"{bs:4d} {S:7d} {k:>12} {v['err']}"); continue
                sp = f"{base / v['us']:.2f}x" if base else "-"
                print(f"{bs:4d} {S:7d} {k:>12} {v['us']:9.1f} {v['bytes']/1e6:9.1f} "
                      f"{v['gbs']:8.1f} {100*v['gbs']/HBM_PEAK_GBS:6.1f}% {sp:>11}")
                rows.append(dict(bs=bs, seq=S, variant=k, **v))
            print()
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
