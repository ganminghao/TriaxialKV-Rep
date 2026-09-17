"""GPU tests for the TriAxialKV kernels (python/sglang/srt/layers/attention/triton_ops/triaxial_kv.py).

Run with:  python -m pytest python/sglang/test/test_triaxial_kv.py -q
The kernel module imports only torch/triton and is loaded here by file path, so the tests
themselves do not need the sglang server dependencies. (pytest still imports the enclosing
`sglang` package when the file sits inside it; in an env without those deps, copy this
file elsewhere and set TRIAXIAL_KV_MODULE=/path/to/triaxial_kv.py.)
"""

import math
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _PY_ROOT not in sys.path:
    sys.path.insert(0, _PY_ROOT)

import importlib.util

# TRIAXIAL_KV_MODULE lets a copy of this file (e.g. in a scratch dir, or an env without the
# sglang package deps) point at the kernel module explicitly.
_MOD_PATH = os.environ.get(
    "TRIAXIAL_KV_MODULE",
    os.path.join(
        _PY_ROOT, "sglang", "srt", "layers", "attention", "triton_ops", "triaxial_kv.py"
    ),
)
_spec = importlib.util.spec_from_file_location("triaxial_kv", _MOD_PATH)
tk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tk)

TriaxialLayerKV = tk.TriaxialLayerKV

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DEV = "cuda"
G = 32
MAX_KV_SPLITS = 8


def _sm_count():
    return torch.cuda.get_device_properties(0).multi_processor_count


# ----------------------------------------------------------------------------- helpers


def make_pool(H, D, size2, size4, dtype=torch.bfloat16, seed=0, scale=1.0):
    """Random pool filled through the quant kernels. Returns (kv, x_k, x_v) where x_k / x_v
    are the original bf16 values indexed by slot ([size2 + size4, H, D])."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    assert size2 % G == 0
    num_pages2 = size2 // G
    total = size2 + size4
    x_k = torch.randn(total, H, D, generator=g, device=DEV, dtype=torch.float32) * scale
    x_v = torch.randn(total, H, D, generator=g, device=DEV, dtype=torch.float32) * scale
    x_k = x_k.to(dtype)
    x_v = x_v.to(dtype)

    kv = TriaxialLayerKV(
        offset=size2,
        k2_data=torch.zeros(num_pages2, H, D, 8, dtype=torch.uint8, device=DEV),
        k2_meta=torch.zeros(num_pages2, H, D, 2, dtype=torch.float16, device=DEV),
        v2_data=torch.zeros(size2, H, D // 4, dtype=torch.uint8, device=DEV),
        v2_meta=torch.zeros(size2, H, D // G, 2, dtype=torch.float16, device=DEV),
        k4_data=torch.zeros(size4, H, D // 2, dtype=torch.uint8, device=DEV),
        k4_meta=torch.zeros(size4, H, D // G, 2, dtype=torch.float16, device=DEV),
        v4_data=torch.zeros(size4, H, D // 2, dtype=torch.uint8, device=DEV),
        v4_meta=torch.zeros(size4, H, D // G, 2, dtype=torch.float16, device=DEV),
    )
    # INT4 rows (row 0 included, it is just data here)
    rows4 = torch.arange(size4, device=DEV, dtype=torch.int64)
    tk.quant_int4_tokens(x_k[size2:], rows4, kv.k4_data, kv.k4_meta)
    tk.quant_int4_tokens(x_v[size2:], rows4, kv.v4_data, kv.v4_meta)
    # INT2 V rows
    rows2 = torch.arange(size2, device=DEV, dtype=torch.int64)
    tk.quant_int2_v_tokens(x_v[:size2], rows2, kv.v2_data, kv.v2_meta)
    # INT2 K pages: page p holds slots 32p..32p+31 in j order
    pages = torch.arange(num_pages2, device=DEV, dtype=torch.int64)
    tok_rows = torch.arange(size2, device=DEV, dtype=torch.int32).view(num_pages2, G)
    tk.quant_int2_k_pages(x_k[:size2], tok_rows, pages, kv.k2_data, kv.k2_meta)
    return kv, x_k, x_v


def random_slot_list(n, size2, size4, gen, mode="mixed"):
    """Slot ids for one request. mode: 'int2' (page-unaligned runs), 'int4', 'mixed'
    (alternating runs of random length, some runs of repeated slots)."""
    slots = []
    while len(slots) < n:
        if mode == "int2":
            kind = 2
        elif mode == "int4":
            kind = 4
        else:
            kind = 2 if torch.rand(1, generator=gen, device=DEV).item() < 0.5 else 4
        run = int(torch.randint(1, 80, (1,), generator=gen, device=DEV).item())
        run = min(run, n - len(slots))
        if kind == 2:
            start = int(torch.randint(0, size2 - run + 1, (1,), generator=gen, device=DEV))
            slots.extend(range(start, start + run))  # page-unaligned contiguous run
        else:
            start = int(torch.randint(0, size4 - run + 1, (1,), generator=gen, device=DEV))
            slots.extend(range(size2 + start, size2 + start + run))
        if torch.rand(1, generator=gen, device=DEV).item() < 0.2 and len(slots) < n:
            slots.append(slots[-1])  # repeated slot
    return torch.tensor(slots[:n], dtype=torch.int64, device=DEV)


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


def build_decode_batch(kv, seq_lens, gen, modes):
    """kv_indices concatenated per request in kv_indptr order."""
    size2 = kv.offset
    size4 = kv.k4_data.shape[0]
    idx = [random_slot_list(int(n), size2, size4, gen, mode) for n, mode in zip(seq_lens, modes)]
    kv_indices = torch.cat(idx)
    kv_indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEV)
    kv_indptr[1:] = torch.cumsum(torch.tensor(seq_lens, device=DEV), 0)
    return kv_indptr, kv_indices


def run_mixed_decode(q, kv, kv_indptr, kv_indices, num_kv_splits, sm_scale, logit_cap=0.0):
    bs, Hq, D = q.shape
    o = torch.empty_like(q)
    attn_logits = torch.empty(bs, Hq, MAX_KV_SPLITS, D, dtype=torch.float32, device=DEV)
    attn_lse = torch.empty(bs, Hq, MAX_KV_SPLITS, dtype=torch.float32, device=DEV)
    tk.decode_attention_fwd_mixed(
        q, kv, o, kv_indptr, kv_indices, attn_logits, attn_lse,
        num_kv_splits, MAX_KV_SPLITS, sm_scale, logit_cap,
    )
    return o


# ----------------------------------------------------------------------------- quant kernels


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quant_int4_tokens_matches_reference(D, dtype):
    torch.manual_seed(1)
    n, H, N = 300, 8, 1024
    x = (torch.randn(n, H, D, device=DEV) * 3).to(dtype)
    # a few degenerate groups: constant rows (max == min) and tiny ranges
    x[0, 0, :G] = 1.25
    x[1, 1] = 0.0
    x[2, 2, G : 2 * G] = torch.tensor(0.5, dtype=dtype) + torch.arange(G, device=DEV).to(dtype) * 1e-5
    rows = torch.randperm(N, device=DEV)[:n]
    data = torch.zeros(N, H, D // 2, dtype=torch.uint8, device=DEV)
    meta = torch.zeros(N, H, D // G, 2, dtype=torch.float16, device=DEV)
    tk.quant_int4_tokens(x, rows, data, meta)

    ref_data, ref_meta = tk.ref_quant_int4_tokens(x)
    torch.testing.assert_close(data[rows], ref_data, rtol=0, atol=0)
    torch.testing.assert_close(meta[rows], ref_meta, rtol=0, atol=0)
    # untouched rows stay zero
    untouched = torch.ones(N, dtype=torch.bool, device=DEV)
    untouched[rows] = False
    assert int(data[untouched].sum()) == 0 and int(meta[untouched].float().abs().sum()) == 0

    # round-trip error bound: |x - deq| <= scale/2 + eps (eps: fp16 rounding of scale/min)
    xh = tk.ref_dequant_tokens(data[rows], meta[rows], 4)
    sc = meta[rows][..., 0].float().repeat_interleave(G, dim=-1)
    eps = 2e-3 * x.float().abs().amax() + 1e-6
    assert ((x.float() - xh).abs() <= sc / 2 + eps).all()
    # constant group -> q = 0, dequant == min
    assert int(data[rows[0], 0, : G // 2].sum()) == 0
    assert torch.allclose(xh[0, 0, :G], torch.full((G,), 1.25, device=DEV))


@pytest.mark.parametrize("D", [64, 128])
def test_quant_int2_v_tokens_matches_reference(D):
    torch.manual_seed(2)
    n, H, N = 257, 8, 700
    x = (torch.randn(n, H, D, device=DEV) * 2).to(torch.bfloat16)
    x[5, 3] = -2.0
    rows = torch.randperm(N, device=DEV)[:n]
    data = torch.zeros(N, H, D // 4, dtype=torch.uint8, device=DEV)
    meta = torch.zeros(N, H, D // G, 2, dtype=torch.float16, device=DEV)
    tk.quant_int2_v_tokens(x, rows, data, meta)

    ref_data, ref_meta = tk.ref_quant_int2_v_tokens(x)
    torch.testing.assert_close(data[rows], ref_data, rtol=0, atol=0)
    torch.testing.assert_close(meta[rows], ref_meta, rtol=0, atol=0)

    xh = tk.ref_dequant_tokens(data[rows], meta[rows], 2)
    sc = meta[rows][..., 0].float().repeat_interleave(G, dim=-1)
    eps = 2e-3 * x.float().abs().amax() + 1e-6
    assert ((x.float() - xh).abs() <= sc / 2 + eps).all()
    assert torch.allclose(xh[5, 3], torch.full((D,), -2.0, device=DEV))


@pytest.mark.parametrize("D", [64, 128])
def test_quant_int2_k_pages_matches_reference(D):
    torch.manual_seed(3)
    H, num_pages, n_extra, NP = 8, 37, 100, 90
    n = num_pages * G + n_extra  # x has more rows than the pages use
    x = torch.randn(n, H, D, device=DEV).to(torch.bfloat16)
    # tok_rows: an arbitrary permutation of rows (tokens of a page need not be contiguous in x)
    perm = torch.randperm(n, device=DEV)[: num_pages * G]
    tok_rows = perm.view(num_pages, G).to(torch.int32).contiguous()
    pages = torch.randperm(NP, device=DEV)[:num_pages]
    data = torch.zeros(NP, H, D, 8, dtype=torch.uint8, device=DEV)
    meta = torch.zeros(NP, H, D, 2, dtype=torch.float16, device=DEV)
    tk.quant_int2_k_pages(x, tok_rows, pages, data, meta)

    x_pages = x[tok_rows.long()]  # [num_pages, 32, H, D]
    ref_data, ref_meta = tk.ref_quant_int2_k_pages(x_pages)
    torch.testing.assert_close(data[pages], ref_data, rtol=0, atol=0)
    torch.testing.assert_close(meta[pages], ref_meta, rtol=0, atol=0)

    xh = tk.ref_dequant_k_pages(data[pages], meta[pages])  # [P,32,H,D]
    sc = meta[pages][..., 0].float()[:, None]  # [P,1,H,D]
    eps = 2e-3 * x.float().abs().amax() + 1e-6
    assert ((x_pages.float() - xh).abs() <= sc / 2 + eps).all()


def test_quant_bit_layout_is_spec_exact():
    """Hand-checked packing: INT4 low nibble = channel 2b; INT2 V bits 2*(d%4); INT2 K bits 2*(j%4)."""
    H, D = 1, 32
    # channels 0..31 hold levels 0..15,0..15 exactly (x = q * 1 + 0 after fp16 rounding)
    x = (torch.arange(D, device=DEV) % 16).float().to(torch.bfloat16)[None, None]  # [1,1,32]
    data = torch.zeros(1, H, D // 2, dtype=torch.uint8, device=DEV)
    meta = torch.zeros(1, H, 1, 2, dtype=torch.float16, device=DEV)
    tk.quant_int4_tokens(x, torch.zeros(1, dtype=torch.int64, device=DEV), data, meta)
    expect = torch.tensor([(2 * b) % 16 | (((2 * b + 1) % 16) << 4) for b in range(16)], dtype=torch.uint8, device=DEV)
    assert torch.equal(data[0, 0], expect)
    assert meta[0, 0, 0, 0].item() == 1.0 and meta[0, 0, 0, 1].item() == 0.0

    x = (torch.arange(D, device=DEV) % 4).float().to(torch.bfloat16)[None, None]
    data = torch.zeros(1, H, D // 4, dtype=torch.uint8, device=DEV)
    tk.quant_int2_v_tokens(x, torch.zeros(1, dtype=torch.int64, device=DEV), data, meta)
    assert torch.equal(data[0, 0], torch.full((8,), 0b11100100, dtype=torch.uint8, device=DEV))

    # INT2 K page: token j has value j % 4 in every channel -> byte = 0b11100100
    x = (torch.arange(G, device=DEV) % 4).float().to(torch.bfloat16)[:, None, None].expand(G, H, D).contiguous()
    data = torch.zeros(1, H, D, 8, dtype=torch.uint8, device=DEV)
    meta = torch.zeros(1, H, D, 2, dtype=torch.float16, device=DEV)
    tok_rows = torch.arange(G, dtype=torch.int32, device=DEV).view(1, G)
    tk.quant_int2_k_pages(x, tok_rows, torch.zeros(1, dtype=torch.int64, device=DEV), data, meta)
    assert torch.equal(data[0, 0], torch.full((D, 8), 0b11100100, dtype=torch.uint8, device=DEV))
    assert (meta[0, 0, :, 0] == 1.0).all() and (meta[0, 0, :, 1] == 0.0).all()


# ----------------------------------------------------------------------------- dequant_gather


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("mode", ["int2", "int4", "mixed"])
def test_dequant_gather_matches_reference(D, mode):
    H, size2, size4 = 8, 64 * G, 1500
    kv, x_k, x_v = make_pool(H, D, size2, size4, seed=10)
    gen = torch.Generator(device=DEV).manual_seed(11)
    slots = random_slot_list(1234, size2, size4, gen, mode)
    # add explicit repeats and a page-unaligned INT2 run crossing a page boundary
    slots = torch.cat([slots, slots[:50], torch.arange(3 * G - 5, 4 * G + 7, device=DEV)])
    m = slots.numel()
    out_k = torch.empty(m, H, D, dtype=torch.bfloat16, device=DEV)
    out_v = torch.empty(m, H, D, dtype=torch.bfloat16, device=DEV)
    tk.dequant_gather(slots, kv, out_k, out_v)
    ref_k, ref_v = tk.ref_dequant_gather(slots, kv)
    torch.testing.assert_close(out_k.float(), ref_k.to(torch.bfloat16).float(), rtol=0, atol=0)
    torch.testing.assert_close(out_v.float(), ref_v.to(torch.bfloat16).float(), rtol=0, atol=0)
    # and the reference itself is close to the original values (sanity on the layout):
    # per-group range <= 2 * amax, so the half-step is <= amax / (2^bits - 1).
    amax = max(x_k.float().abs().amax(), x_v.float().abs().amax())
    sel2 = slots < size2
    for ref, x in ((ref_k, x_k), (ref_v, x_v)):
        err = (ref - x[slots].float()).abs()
        if sel2.any():
            assert err[sel2].amax() < amax / 3 + 0.05
        if (~sel2).any():
            assert err[~sel2].amax() < amax / 15 + 0.05


def test_dequant_gather_empty_and_strided_out():
    H, D, size2, size4 = 4, 64, 8 * G, 100
    kv, _, _ = make_pool(H, D, size2, size4, seed=12)
    slots = torch.tensor([], dtype=torch.int64, device=DEV)
    out_k = torch.empty(0, H, D, dtype=torch.bfloat16, device=DEV)
    tk.dequant_gather(slots, kv, out_k, out_k.clone())
    # output rows that are a slice of a larger [m, 2H, D] scratch (head stride != D)
    slots = torch.arange(0, size2 + size4, 7, device=DEV)
    scratch = torch.zeros(slots.numel(), 2 * H, D, dtype=torch.bfloat16, device=DEV)
    tk.dequant_gather(slots, kv, scratch[:, :H], scratch[:, H:])
    ref_k, ref_v = tk.ref_dequant_gather(slots, kv)
    torch.testing.assert_close(scratch[:, :H].float(), ref_k.to(torch.bfloat16).float(), rtol=0, atol=0)
    torch.testing.assert_close(scratch[:, H:].float(), ref_v.to(torch.bfloat16).float(), rtol=0, atol=0)


# ----------------------------------------------------------------------------- decode attention


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("kv_group_num", [1, 4, 8])
@pytest.mark.parametrize("bs", [1, 4, 8])
def test_decode_attention_fwd_mixed(bs, kv_group_num, D):
    H = 8
    Hq = H * kv_group_num
    size2, size4 = 200 * G, 5000
    kv, _, _ = make_pool(H, D, size2, size4, seed=100 + bs)
    gen = torch.Generator(device=DEV).manual_seed(200 + bs * 10 + kv_group_num)
    seq_lens = [int(torch.randint(1, 4097, (1,), generator=gen, device=DEV)) for _ in range(bs)]
    if bs >= 4:
        seq_lens[0] = 1  # degenerate single-token request
        seq_lens[1] = 4096
    modes = ["int2", "int4", "mixed"] * (bs // 3 + 1)
    modes = modes[:bs]
    kv_indptr, kv_indices = build_decode_batch(kv, seq_lens, gen, modes)
    num_kv_splits = num_kv_splits_like_sglang(
        torch.tensor(seq_lens, device=DEV), Hq, H, MAX_KV_SPLITS, _sm_count()
    )
    assert int(num_kv_splits.max()) <= MAX_KV_SPLITS and int(num_kv_splits.min()) >= 1

    q = torch.randn(bs, Hq, D, generator=gen, device=DEV).to(torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(D)
    o = run_mixed_decode(q, kv, kv_indptr, kv_indices, num_kv_splits, sm_scale)

    ref_k, ref_v = tk.ref_dequant_gather(kv_indices, kv)
    o_ref = tk.ref_attention(q.float(), ref_k, ref_v, kv_indptr, sm_scale)
    torch.testing.assert_close(o.float(), o_ref, atol=2e-2, rtol=2e-2)


def test_decode_attention_fwd_mixed_logit_cap_and_static_splits():
    H, D, kv_group_num, bs = 8, 128, 8, 3
    Hq = H * kv_group_num
    kv, _, _ = make_pool(H, D, 64 * G, 3000, seed=7)
    gen = torch.Generator(device=DEV).manual_seed(8)
    seq_lens = [33, 1000, 2500]
    kv_indptr, kv_indices = build_decode_batch(kv, seq_lens, gen, ["mixed"] * bs)
    num_kv_splits = torch.full((bs,), MAX_KV_SPLITS, dtype=torch.int32, device=DEV)  # static splits
    q = (torch.randn(bs, Hq, D, generator=gen, device=DEV) * 4).to(torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(D)
    for logit_cap in (0.0, 30.0):
        o = run_mixed_decode(q, kv, kv_indptr, kv_indices, num_kv_splits, sm_scale, logit_cap)
        ref_k, ref_v = tk.ref_dequant_gather(kv_indices, kv)
        o_ref = tk.ref_attention(q.float(), ref_k, ref_v, kv_indptr, sm_scale, logit_cap)
        torch.testing.assert_close(o.float(), o_ref, atol=2e-2, rtol=2e-2)


def test_decode_attention_fwd_mixed_fp16_q():
    H, D, kv_group_num, bs = 8, 128, 4, 2
    Hq = H * kv_group_num
    kv, _, _ = make_pool(H, D, 32 * G, 2000, dtype=torch.float16, seed=9)
    gen = torch.Generator(device=DEV).manual_seed(10)
    kv_indptr, kv_indices = build_decode_batch(kv, [700, 1300], gen, ["mixed", "int2"])
    num_kv_splits = num_kv_splits_like_sglang(
        torch.tensor([700, 1300], device=DEV), Hq, H, MAX_KV_SPLITS, _sm_count()
    )
    q = torch.randn(bs, Hq, D, generator=gen, device=DEV).to(torch.float16)
    o = run_mixed_decode(q, kv, kv_indptr, kv_indices, num_kv_splits, 1.0 / math.sqrt(D))
    ref_k, ref_v = tk.ref_dequant_gather(kv_indices, kv)
    o_ref = tk.ref_attention(q.float(), ref_k, ref_v, kv_indptr, 1.0 / math.sqrt(D))
    torch.testing.assert_close(o.float(), o_ref, atol=2e-2, rtol=2e-2)


def test_decode_attention_cuda_graph():
    H, D, kv_group_num, bs = 8, 128, 8, 4
    Hq = H * kv_group_num
    kv, _, _ = make_pool(H, D, 100 * G, 4000, seed=21)
    gen = torch.Generator(device=DEV).manual_seed(22)
    seq_lens = [512, 1, 3000, 77]
    kv_indptr, kv_indices = build_decode_batch(kv, seq_lens, gen, ["mixed", "int4", "mixed", "int2"])
    # static buffers, as the CUDA-graph runner keeps them
    max_total = 4 * 4096
    kv_indices_buf = torch.zeros(max_total, dtype=torch.int64, device=DEV)
    kv_indices_buf[: kv_indices.numel()] = kv_indices
    kv_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32, device=DEV)
    kv_indptr_buf.copy_(kv_indptr)
    num_kv_splits = torch.full((bs,), MAX_KV_SPLITS, dtype=torch.int32, device=DEV)
    q = torch.randn(bs, Hq, D, generator=gen, device=DEV).to(torch.bfloat16)
    o = torch.zeros_like(q)
    attn_logits = torch.zeros(bs, Hq, MAX_KV_SPLITS, D, dtype=torch.float32, device=DEV)
    attn_lse = torch.zeros(bs, Hq, MAX_KV_SPLITS, dtype=torch.float32, device=DEV)
    sm_scale = 1.0 / math.sqrt(D)

    def run():
        tk.decode_attention_fwd_mixed(
            q, kv, o, kv_indptr_buf, kv_indices_buf, attn_logits, attn_lse,
            num_kv_splits, MAX_KV_SPLITS, sm_scale,
        )

    # warm up (compiles) on a side stream, then capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # new inputs (q and a different slot list / lengths), replay, compare with eager reference
    seq_lens2 = [1000, 2048, 5, 4096]
    kv_indptr2, kv_indices2 = build_decode_batch(kv, seq_lens2, gen, ["int2", "mixed", "mixed", "int4"])
    kv_indices_buf[: kv_indices2.numel()] = kv_indices2
    kv_indptr_buf.copy_(kv_indptr2)
    q.copy_(torch.randn(bs, Hq, D, generator=gen, device=DEV).to(torch.bfloat16))
    o.zero_()
    graph.replay()
    torch.cuda.synchronize()
    o_graph = o.clone()

    ref_k, ref_v = tk.ref_dequant_gather(kv_indices2, kv)
    o_ref = tk.ref_attention(q.float(), ref_k, ref_v, kv_indptr2, sm_scale)
    torch.testing.assert_close(o_graph.float(), o_ref, atol=2e-2, rtol=2e-2)
    # eager on the same inputs must give the same result as the replay
    o.zero_()
    run()
    torch.testing.assert_close(o.float(), o_graph.float(), rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-x"] + sys.argv[1:]))
