"""Cost of the TriAxialKV prefill / write path, per layer.

TriaxialFlashInferBackend.forward_extend materialises a bf16 scratch of the WHOLE
attended window every layer:  dequant_gather(prefix slots) -> scratch, copy new rows,
then run the paged FA kernel over the scratch.  So per prefill layer it pays

    dequant_gather(P tokens)  +  FA-prefill over (P+N)      [TriAxialKV]
vs  FA-prefill over (P+N) reading the paged pool directly   [bf16 baseline]

and the write path pays quant kernels instead of a plain copy.
Measured here in isolation, plus the chunked-prefill re-dequantisation amplification.
"""
import argparse, math, sys, torch
sys.path.insert(0, "/data/gmh/workspace/sglang-triaxialkv/python")
from sglang.srt.layers.attention.triton_ops import triaxial_kv as tk
from sgl_kernel.flash_attn import flash_attn_with_kvcache

G = 32
dev = "cuda"


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1e3 / iters)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Hkv", type=int, default=8)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--int2-fraction", type=float, default=0.85)
    ap.add_argument("--P-list", type=str, default="2048,8192,16384,32768")
    a = ap.parse_args()
    H, D, f = a.Hkv, a.D, a.int2_fraction
    gen = torch.Generator(device=dev).manual_seed(0)

    print(f"# per-layer prefill-path cost, H={H} D={D} int2_frac={f}")
    hdr = (f"{'P (prefix tok)':>15} {'dequant_gather':>15} {'bf16 gather':>12} {'quant-write':>12} "
           f"{'FA3 over P':>11} {'deq/FA3':>8} {'deq GB/s':>9}")
    print(hdr); print("-" * len(hdr))

    for P in [int(x) for x in a.P_list.split(",")]:
        n2 = (int(P * f) // G) * G; n4 = P - n2
        size2 = n2 + G; size4 = n4 + 1
        kv = tk.TriaxialLayerKV(
            offset=size2,
            k2_data=torch.zeros(size2 // G, H, D, 8, dtype=torch.uint8, device=dev),
            k2_meta=torch.ones(size2 // G, H, D, 2, dtype=torch.float16, device=dev),
            v2_data=torch.zeros(size2, H, D // 4, dtype=torch.uint8, device=dev),
            v2_meta=torch.ones(size2, H, D // G, 2, dtype=torch.float16, device=dev),
            k4_data=torch.zeros(size4, H, D // 2, dtype=torch.uint8, device=dev),
            k4_meta=torch.ones(size4, H, D // G, 2, dtype=torch.float16, device=dev),
            v4_data=torch.zeros(size4, H, D // 2, dtype=torch.uint8, device=dev),
            v4_meta=torch.ones(size4, H, D // G, 2, dtype=torch.float16, device=dev),
        )
        slots = torch.cat([G + torch.arange(n2, device=dev),
                           size2 + 1 + torch.arange(n4, device=dev)]).to(torch.int64)
        ok = torch.empty(P, H, D, dtype=torch.bfloat16, device=dev)
        ov = torch.empty_like(ok)
        t_deq = bench(lambda: tk.dequant_gather(slots, kv, ok, ov))

        # bf16 baseline materialisation (what you'd pay if the pool were already bf16)
        kb = torch.zeros(P + 1, H, D, dtype=torch.bfloat16, device=dev)
        vb = torch.zeros_like(kb)
        sl = (torch.arange(P, device=dev) + 1)
        t_gather = bench(lambda: (torch.index_select(kb, 0, sl, out=ok),
                                  torch.index_select(vb, 0, sl, out=ov)))

        # write path: quantise P new tokens into the pool (what set_kv_buffer does)
        x = torch.randn(P, H, D, generator=gen, device=dev).to(torch.bfloat16)
        rows4 = torch.arange(min(n4, size4 - 1), device=dev)
        rows2 = torch.arange(n2, device=dev)
        pages = torch.arange(n2 // G, device=dev)
        tokrows = torch.arange(n2, device=dev, dtype=torch.int32).view(-1, G)

        def write():
            if n4: tk.quant_int4_tokens(x[:rows4.numel()], rows4, kv.k4_data, kv.k4_meta)
            if n4: tk.quant_int4_tokens(x[:rows4.numel()], rows4, kv.v4_data, kv.v4_meta)
            if n2: tk.quant_int2_v_tokens(x[:n2], rows2, kv.v2_data, kv.v2_meta)
            if n2: tk.quant_int2_k_pages(x[:n2], tokrows, pages, kv.k2_data, kv.k2_meta)
        t_write = bench(write)

        # FA3 prefill over the same window (q = P tokens, causal)
        Hq = H * a.group
        qp = torch.randn(1, P, Hq, D, generator=gen, device=dev).to(torch.bfloat16)
        kc = torch.zeros(P + 1, 1, H, D, dtype=torch.bfloat16, device=dev)
        vc = torch.zeros_like(kc)
        pt = (torch.arange(P, dtype=torch.int32, device=dev) + 1).view(1, P)
        cs = torch.tensor([P], dtype=torch.int32, device=dev)
        cu = torch.tensor([0, P], dtype=torch.int32, device=dev)
        try:
            t_fa = bench(lambda: flash_attn_with_kvcache(
                qp.view(P, Hq, D), kc, vc, page_table=pt, cache_seqlens=cs,
                cu_seqlens_q=cu, max_seqlen_q=P, causal=True,
                softmax_scale=1.0 / math.sqrt(D)), iters=5, warmup=2)
        except Exception as ex:
            t_fa = float("nan"); print("  FA3 prefill err:", repr(ex)[:120])

        deq_bytes = n2 * H * 96 + n4 * H * 160 + P * H * D * 2 * 2  # read packed + write bf16
        print(f"{P:15d} {t_deq:13.1f}us {t_gather:10.1f}us {t_write:10.1f}us "
              f"{t_fa:9.1f}us {t_deq/t_fa:7.2f}x {deq_bytes/t_deq/1e3:8.0f}")
        del kv, kb, vb, kc, vc, qp
        torch.cuda.empty_cache()

    print("\n# chunked-prefill re-dequantisation amplification")
    print(f"{'ctx len':>9} {'chunk':>7} {'chunks':>7} {'tok-deq total':>14} {'vs bf16 (x)':>12}")
    for N in (8192, 32768, 131072):
        for C in (4096, 8192):
            nc = -(-N // C)
            deq = sum(i * C for i in range(nc))  # prefix re-read per chunk
            print(f"{N:9d} {C:7d} {nc:7d} {deq:14d} {deq/N:11.1f}x")


if __name__ == "__main__":
    main()
