"""Micro-benchmark: decode_attention_fwd_mixed (INT2/INT4 TriAxialKV) vs SGLang's bf16
decode_attention_fwd, at the Qwen3-VL-32B decode geometry.

    python python/sglang/test/bench_triaxial_decode.py [--bs 12] [--seq-len 11000]
        [--int2-fraction 0.75] [--layers 64] [--distinct-layers 8]

Each "layer" call reads a distinct pool (cycling over --distinct-layers pools, each larger
than the H100 L2) so the numbers reflect HBM traffic. Effective bytes/s counts the KV
payload only (INT2 token-head = 96 B, INT4 = 160 B, bf16 = 512 B; kv_indices excluded).
The bf16 baseline needs decode_attention.py, which imports sglang.srt.utils; if the sglang
package is not importable the file is loaded by path with a stub for is_hip(). If neither
works the baseline is skipped.
"""

import argparse
import importlib.util
import math
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_OPS = os.path.join(_PY_ROOT, "sglang", "srt", "layers", "attention", "triton_ops")


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_triaxial():
    try:
        from sglang.srt.layers.attention.triton_ops import triaxial_kv as tk

        return tk
    except Exception:
        return _load_by_path("triaxial_kv", os.path.join(_OPS, "triaxial_kv.py"))


def load_sglang_decode():
    try:
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )

        return decode_attention_fwd, "sglang package"
    except Exception as e:  # noqa: BLE001
        first_err = e
    try:
        if "sglang.srt.utils" not in sys.modules:
            stub = types.ModuleType("sglang.srt.utils")
            stub.is_hip = lambda: False
            sys.modules["sglang.srt.utils"] = stub
        mod = _load_by_path("decode_attention", os.path.join(_OPS, "decode_attention.py"))
        return mod.decode_attention_fwd, "decode_attention.py loaded by path (is_hip stub)"
    except Exception as e:  # noqa: BLE001
        print(f"[bench] bf16 baseline unavailable: {first_err!r} / {e!r}")
        return None, None


def num_kv_splits_like_sglang(seq_lens, num_head, num_kv_head, max_kv_splits, core_count):
    """Port of triton_backend.get_num_kv_splits_triton (num_group = 1)."""
    seq_lens = seq_lens.to(torch.int64)
    num_seq = seq_lens.numel()
    max_seq_len = int(seq_lens.max())
    min_seq_len = int(seq_lens.min())
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
    n1 = -(-seq_lens // kv_chunk_size_1)
    n2 = -(-seq_lens // kv_chunk_size_2)
    return torch.maximum(n1, n2).to(torch.int32)


def make_mixed_layer(tk, H, D, size2, size4, gen, dev):
    """Random pool filled through the quant kernels (from random bf16 K/V)."""
    G = 32
    num_pages2 = size2 // G
    kv = tk.TriaxialLayerKV(
        offset=size2,
        k2_data=torch.empty(num_pages2, H, D, 8, dtype=torch.uint8, device=dev),
        k2_meta=torch.empty(num_pages2, H, D, 2, dtype=torch.float16, device=dev),
        v2_data=torch.empty(size2, H, D // 4, dtype=torch.uint8, device=dev),
        v2_meta=torch.empty(size2, H, D // G, 2, dtype=torch.float16, device=dev),
        k4_data=torch.empty(size4, H, D // 2, dtype=torch.uint8, device=dev),
        k4_meta=torch.empty(size4, H, D // G, 2, dtype=torch.float16, device=dev),
        v4_data=torch.empty(size4, H, D // 2, dtype=torch.uint8, device=dev),
        v4_meta=torch.empty(size4, H, D // G, 2, dtype=torch.float16, device=dev),
    )
    chunk = 16384
    for start in range(0, size4, chunk):
        n = min(chunk, size4 - start)
        rows = torch.arange(start, start + n, device=dev)
        x = torch.randn(n, H, D, generator=gen, device=dev).to(torch.bfloat16)
        tk.quant_int4_tokens(x, rows, kv.k4_data, kv.k4_meta)
        x = torch.randn(n, H, D, generator=gen, device=dev).to(torch.bfloat16)
        tk.quant_int4_tokens(x, rows, kv.v4_data, kv.v4_meta)
    for start in range(0, size2, chunk):
        n = min(chunk, size2 - start)
        rows = torch.arange(start, start + n, device=dev)
        x = torch.randn(n, H, D, generator=gen, device=dev).to(torch.bfloat16)
        tk.quant_int2_v_tokens(x, rows, kv.v2_data, kv.v2_meta)
        x = torch.randn(n, H, D, generator=gen, device=dev).to(torch.bfloat16)
        pages = torch.arange(start // G, (start + n) // G, device=dev)
        tok_rows = torch.arange(n, device=dev, dtype=torch.int32).view(-1, G)
        tk.quant_int2_k_pages(x, tok_rows, pages, kv.k2_data, kv.k2_meta)
    return kv


def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters  # us per fn() call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=11000)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--kv-group-num", type=int, default=8)
    ap.add_argument("--int2-fraction", type=float, default=0.75)
    ap.add_argument("--layers", type=int, default=64, help="calls per timed forward")
    ap.add_argument("--distinct-layers", type=int, default=8, help="distinct pools cycled over")
    ap.add_argument("--max-kv-splits", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=5, help="timed forwards")
    ap.add_argument("--skip-bf16", action="store_true")
    args = ap.parse_args()

    dev = "cuda"
    torch.cuda.set_device(0)
    tk = load_triaxial()
    G = 32
    bs, S, H, D = args.bs, args.seq_len, args.H, args.D
    Hq = H * args.kv_group_num
    total = bs * S
    n2_per_req = (int(S * args.int2_fraction) // G) * G  # whole INT2 pages per request
    n4_per_req = S - n2_per_req
    size2 = bs * n2_per_req + G  # + reserved page 0
    size4 = bs * n4_per_req + 1  # + reserved row 0
    gen = torch.Generator(device=dev).manual_seed(0)
    core_count = torch.cuda.get_device_properties(0).multi_processor_count

    print(
        f"[bench] {torch.cuda.get_device_name()} torch {torch.__version__} | bs={bs} seq_len={S} "
        f"H={H} D={D} q_heads={Hq} int2_fraction={n2_per_req / S:.3f} layers={args.layers} "
        f"distinct_pools={args.distinct_layers}"
    )

    # slot lists: per request, INT2 prefix (whole pages) then INT4 suffix
    idx = []
    for b in range(bs):
        s2 = G + b * n2_per_req + torch.arange(n2_per_req, device=dev)
        s4 = size2 + 1 + b * n4_per_req + torch.arange(n4_per_req, device=dev)
        idx.append(torch.cat([s2, s4]))
    kv_indices = torch.cat(idx).to(torch.int64)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    kv_indptr[1:] = torch.cumsum(torch.full((bs,), S, device=dev), 0)
    seq_lens = torch.full((bs,), S, device=dev)
    num_kv_splits = num_kv_splits_like_sglang(seq_lens, Hq, H, args.max_kv_splits, core_count)
    print(f"[bench] num_kv_splits={num_kv_splits.tolist()} max_kv_splits={args.max_kv_splits}")

    q = torch.randn(bs, Hq, D, generator=gen, device=dev).to(torch.bfloat16)
    o = torch.empty_like(q)
    attn_logits = torch.empty(bs, Hq, args.max_kv_splits, D, dtype=torch.float32, device=dev)
    attn_lse = torch.empty(bs, Hq, args.max_kv_splits, dtype=torch.float32, device=dev)
    sm_scale = 1.0 / math.sqrt(D)

    # ---- mixed INT2/INT4
    layers = [make_mixed_layer(tk, H, D, size2, size4, gen, dev) for _ in range(args.distinct_layers)]
    mixed_bytes = (bs * n2_per_req * H * 96 + bs * n4_per_req * H * 160)

    def fwd_mixed():
        for i in range(args.layers):
            tk.decode_attention_fwd_mixed(
                q, layers[i % len(layers)], o, kv_indptr, kv_indices, attn_logits, attn_lse,
                num_kv_splits, args.max_kv_splits, sm_scale,
            )

    us = min(bench(fwd_mixed, 1, warmup=2) for _ in range(args.repeats)) / args.layers
    print(
        f"[bench] mixed INT2/INT4 : {us:9.1f} us/call  ({us * args.layers / 1e3:7.2f} ms per {args.layers} layers)  "
        f"KV payload {mixed_bytes / 1e6:8.1f} MB/call -> {mixed_bytes / us / 1e3:7.1f} GB/s effective"
    )
    o_mixed = o.clone()
    del layers
    torch.cuda.empty_cache()

    # ---- bf16 baseline (SGLang decode_attention_fwd, page_size = 1 buffers)
    if args.skip_bf16:
        return
    decode_attention_fwd, how = load_sglang_decode()
    if decode_attention_fwd is None:
        return
    print(f"[bench] bf16 baseline from: {how}")
    n_bf16 = args.distinct_layers
    free, _ = torch.cuda.mem_get_info()
    per_layer = 2 * (total + 1) * H * D * 2
    n_bf16 = max(1, min(n_bf16, int(free * 0.6 // per_layer)))
    k_bufs = [torch.randn(total + 1, H, D, generator=gen, device=dev).to(torch.bfloat16) for _ in range(n_bf16)]
    v_bufs = [torch.randn(total + 1, H, D, generator=gen, device=dev).to(torch.bfloat16) for _ in range(n_bf16)]
    kv_indices_bf16 = (torch.arange(total, device=dev) + 1).to(torch.int64)
    bf16_bytes = total * H * 512

    def fwd_bf16():
        for i in range(args.layers):
            decode_attention_fwd(
                q, k_bufs[i % n_bf16], v_bufs[i % n_bf16], o, kv_indptr, kv_indices_bf16,
                attn_logits, attn_lse, num_kv_splits, args.max_kv_splits, sm_scale, 1.0, 1.0,
            )

    us_bf16 = min(bench(fwd_bf16, 1, warmup=2) for _ in range(args.repeats)) / args.layers
    print(
        f"[bench] bf16 (sglang)   : {us_bf16:9.1f} us/call  ({us_bf16 * args.layers / 1e3:7.2f} ms per {args.layers} layers)  "
        f"KV payload {bf16_bytes / 1e6:8.1f} MB/call -> {bf16_bytes / us_bf16 / 1e3:7.1f} GB/s effective  "
        f"[{n_bf16} distinct pools]"
    )
    print(f"[bench] speedup mixed vs bf16: {us_bf16 / us:.2f}x  (bytes ratio {bf16_bytes / mixed_bytes:.2f}x)")
    assert torch.isfinite(o_mixed).all() and torch.isfinite(o).all()


if __name__ == "__main__":
    main()
