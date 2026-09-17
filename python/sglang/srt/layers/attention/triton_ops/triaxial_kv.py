# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
TriAxialKV: mixed INT2 / INT4 paged KV cache kernels (see TRIAXIAL_DESIGN.md §2, §3).

Layouts (per layer, G = 32 channels per quantization group, H = kv heads, D = head_dim):

    k2_data: uint8 [num_pages2, H, D, 8]   per-channel INT2 keys; byte b holds tokens
                                            j = 4b..4b+3 of the page, token j at bits 2*(j%4)
    k2_meta: fp16  [num_pages2, H, D, 2]   [scale, min] per (page, head, channel)
    v2_data: uint8 [size2, H, D//4]        per-token INT2 values; byte b holds channels
                                            d = 4b..4b+3, channel d at bits 2*(d%4)
    v2_meta: fp16  [size2, H, D//32, 2]    [scale, min] per (slot, head, 32-channel group)
    k4_data: uint8 [size4, H, D//2]        per-token INT4 keys; byte b holds channel 2b in the
                                            low nibble and 2b+1 in the high nibble
    k4_meta: fp16  [size4, H, D//32, 2]
    v4_data, v4_meta: same as k4_*

Slot address space: INT2 slots are s in [0, offset) with page = s // 32, j = s % 32;
INT4 slots are s in [offset, offset + size4) with row r = s - offset.

Quantization: scale = (max - min) / (2^bits - 1), q = clamp(rint((x - min) / scale), 0, 2^bits - 1),
dequant x = q * scale + min. scale and min are rounded to fp16 *before* quantizing so that
the stored meta is exactly what the quantizer used. A scale that rounds to fp16 zero
(this includes max == min) is replaced by 1.0, giving q = 0 and dequant == min.
Rounding is round-half-to-even (libdevice.rint in Triton, torch.round in the reference)
and the division is IEEE round-to-nearest (libdevice.div_rn), so the Triton kernels and
the torch references produce byte-identical packed data.

Kernel structure (decode stage 1 and dequant_gather share the tile loaders): a block of
BLOCK_N slots is classified as all-INT2 / all-INT4 / mixed by comparing the slot ids with
`offset`; K and V are dequantized to fp32 tiles on the fly and fed to tl.dot as bf16/fp16.
Channels are processed as four interleaved sub-tiles (channel 4b + c) so that per-token
packed bytes are loaded once and unpacked in registers; per-(token, group) meta is loaded
once per group and expanded in registers; INT2 K meta (per page and channel) is loaded as
two per-page vectors when a block touches at most two pages, else gathered per element.

This module deliberately imports only torch and triton (no sglang server code).
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_MIN_BLOCK_KV = 32
_G = 32  # quantization group size (channels), also the INT2 page size (tokens)


@dataclass
class TriaxialLayerKV:
    offset: int  # = size2
    k2_data: torch.Tensor  # [num_pages2, H, D, 8] uint8
    k2_meta: torch.Tensor  # [num_pages2, H, D, 2] fp16
    v2_data: torch.Tensor  # [size2, H, D//4] uint8
    v2_meta: torch.Tensor  # [size2, H, D//32, 2] fp16
    k4_data: torch.Tensor  # [size4, H, D//2] uint8
    k4_meta: torch.Tensor  # [size4, H, D//32, 2] fp16
    v4_data: torch.Tensor
    v4_meta: torch.Tensor

    @property
    def num_kv_heads(self) -> int:
        return self.k4_data.shape[1]

    @property
    def head_dim(self) -> int:
        return self.k4_data.shape[2] * 2


def _check_head_dim(D: int, min_d: int = 64):
    # Power of two so tl.arange tiles need no masking. The attention / gather kernels also
    # need D >= 64 so the D/4-wide channel sub-tiles satisfy tl.dot's minimum size of 16;
    # the quant kernels accept any power of two >= 32.
    assert D % _G == 0 and (D & (D - 1)) == 0 and D >= min_d, (
        f"head_dim must be a power of two >= {min_d}, got {D}"
    )


# =============================================================================
# Quantization kernels
# =============================================================================


@triton.jit
def _scale_min(mx, mn, QMAX: tl.constexpr):
    """fp32 max/min -> fp16-rounded (scale, min) as fp32 values."""
    qmax = tl.zeros_like(mx) + QMAX  # fp32 tensor operand for the IEEE division
    sc = libdevice.div_rn(mx - mn, qmax)
    sc_h = sc.to(tl.float16)
    sc_h = tl.where(sc_h == 0.0, 1.0, sc_h.to(tl.float32)).to(tl.float16)
    mn_h = mn.to(tl.float16)
    return sc_h.to(tl.float32), mn_h.to(tl.float32)


@triton.jit
def _quantize(x, sc, mn, QMAX: tl.constexpr):
    """x fp32 -> int32 level in [0, QMAX], same rounding as torch.round."""
    q = libdevice.rint(libdevice.div_rn(x - mn, sc))
    q = tl.minimum(tl.maximum(q, 0.0), QMAX)
    return q.to(tl.int32)


@triton.jit
def _quant_int4_tokens_kernel(
    X,
    Rows,
    Data,
    Meta,
    stride_xn,
    stride_xh,
    H,
    D: tl.constexpr,
    NG: tl.constexpr,  # D // 32
):
    i = tl.program_id(0)
    h = tl.program_id(1)
    row = tl.load(Rows + i).to(tl.int64)

    g = tl.arange(0, NG)  # group
    b = tl.arange(0, 16)  # byte within group (16 bytes = 32 channels)
    # channel 2b (low nibble) and 2b+1 (high nibble) of group g
    x_base = X + i.to(tl.int64) * stride_xn + h * stride_xh
    off_lo = g[:, None] * 32 + 2 * b[None, :]
    x_lo = tl.load(x_base + off_lo).to(tl.float32)
    x_hi = tl.load(x_base + off_lo + 1).to(tl.float32)

    mx = tl.maximum(tl.max(x_lo, 1), tl.max(x_hi, 1))
    mn = tl.minimum(tl.min(x_lo, 1), tl.min(x_hi, 1))
    sc, mn = _scale_min(mx, mn, 15)

    q_lo = _quantize(x_lo, sc[:, None], mn[:, None], 15)
    q_hi = _quantize(x_hi, sc[:, None], mn[:, None], 15)
    packed = (q_lo | (q_hi << 4)).to(tl.uint8)

    d_base = Data + (row * H + h) * (D // 2)
    tl.store(d_base + g[:, None] * 16 + b[None, :], packed)
    m_base = Meta + (row * H + h) * (NG * 2)
    tl.store(m_base + g * 2, sc.to(tl.float16))
    tl.store(m_base + g * 2 + 1, mn.to(tl.float16))


@triton.jit
def _quant_int2_v_tokens_kernel(
    X,
    Rows,
    Data,
    Meta,
    stride_xn,
    stride_xh,
    H,
    D: tl.constexpr,
    NG: tl.constexpr,  # D // 32
):
    i = tl.program_id(0)
    h = tl.program_id(1)
    row = tl.load(Rows + i).to(tl.int64)

    g = tl.arange(0, NG)
    b = tl.arange(0, 8)  # byte within group (8 bytes = 32 channels)
    x_base = X + i.to(tl.int64) * stride_xn + h * stride_xh
    off0 = g[:, None] * 32 + 4 * b[None, :]
    x0 = tl.load(x_base + off0).to(tl.float32)
    x1 = tl.load(x_base + off0 + 1).to(tl.float32)
    x2 = tl.load(x_base + off0 + 2).to(tl.float32)
    x3 = tl.load(x_base + off0 + 3).to(tl.float32)

    mx = tl.maximum(
        tl.maximum(tl.max(x0, 1), tl.max(x1, 1)),
        tl.maximum(tl.max(x2, 1), tl.max(x3, 1)),
    )
    mn = tl.minimum(
        tl.minimum(tl.min(x0, 1), tl.min(x1, 1)),
        tl.minimum(tl.min(x2, 1), tl.min(x3, 1)),
    )
    sc, mn = _scale_min(mx, mn, 3)

    q0 = _quantize(x0, sc[:, None], mn[:, None], 3)
    q1 = _quantize(x1, sc[:, None], mn[:, None], 3)
    q2 = _quantize(x2, sc[:, None], mn[:, None], 3)
    q3 = _quantize(x3, sc[:, None], mn[:, None], 3)
    packed = (q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)).to(tl.uint8)

    d_base = Data + (row * H + h) * (D // 4)
    tl.store(d_base + g[:, None] * 8 + b[None, :], packed)
    m_base = Meta + (row * H + h) * (NG * 2)
    tl.store(m_base + g * 2, sc.to(tl.float16))
    tl.store(m_base + g * 2 + 1, mn.to(tl.float16))


@triton.jit
def _quant_int2_k_pages_kernel(
    X,
    TokRows,
    Pages,
    Data,
    Meta,
    stride_xn,
    stride_xh,
    H,
    D: tl.constexpr,
):
    p = tl.program_id(0)
    h = tl.program_id(1)
    page = tl.load(Pages + p).to(tl.int64)

    b = tl.arange(0, 8)  # byte index -> tokens j = 4b + c
    d = tl.arange(0, D)
    tr_base = TokRows + p * 32
    r0 = tl.load(tr_base + 4 * b).to(tl.int64)
    r1 = tl.load(tr_base + 4 * b + 1).to(tl.int64)
    r2 = tl.load(tr_base + 4 * b + 2).to(tl.int64)
    r3 = tl.load(tr_base + 4 * b + 3).to(tl.int64)

    x_base = X + h * stride_xh + d[None, :]
    x0 = tl.load(x_base + r0[:, None] * stride_xn).to(tl.float32)  # [8, D]
    x1 = tl.load(x_base + r1[:, None] * stride_xn).to(tl.float32)
    x2 = tl.load(x_base + r2[:, None] * stride_xn).to(tl.float32)
    x3 = tl.load(x_base + r3[:, None] * stride_xn).to(tl.float32)

    mx = tl.maximum(
        tl.maximum(tl.max(x0, 0), tl.max(x1, 0)),
        tl.maximum(tl.max(x2, 0), tl.max(x3, 0)),
    )  # [D]
    mn = tl.minimum(
        tl.minimum(tl.min(x0, 0), tl.min(x1, 0)),
        tl.minimum(tl.min(x2, 0), tl.min(x3, 0)),
    )
    sc, mn = _scale_min(mx, mn, 3)

    q0 = _quantize(x0, sc[None, :], mn[None, :], 3)
    q1 = _quantize(x1, sc[None, :], mn[None, :], 3)
    q2 = _quantize(x2, sc[None, :], mn[None, :], 3)
    q3 = _quantize(x3, sc[None, :], mn[None, :], 3)
    packed = (q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)).to(tl.uint8)  # [8, D]

    d_base = Data + (page * H + h) * (D * 8)
    tl.store(d_base + d[None, :] * 8 + b[:, None], packed)
    m_base = Meta + (page * H + h) * (D * 2)
    tl.store(m_base + d * 2, sc.to(tl.float16))
    tl.store(m_base + d * 2 + 1, mn.to(tl.float16))


def quant_int4_tokens(x, rows, data, meta):
    """x [n,H,D] bf16/fp16; rows int64 [n] (row index into data/meta = slot - offset);
    writes data[rows] (uint8 [.,H,D//2]) and meta[rows] (fp16 [.,H,D//32,2])."""
    n, H, D = x.shape
    if n == 0:
        return
    _check_head_dim(D, 32)
    assert x.stride(2) == 1 and data.is_contiguous() and meta.is_contiguous()
    assert data.shape[1:] == (H, D // 2) and meta.shape[1:] == (H, D // _G, 2)
    _quant_int4_tokens_kernel[(n, H)](
        x,
        rows,
        data,
        meta,
        x.stride(0),
        x.stride(1),
        H,
        D=D,
        NG=D // _G,
        num_warps=1,
    )


def quant_int2_v_tokens(x, rows, data, meta):
    """Same as quant_int4_tokens for INT2 per-token values: data [.,H,D//4], meta [.,H,D//32,2]."""
    n, H, D = x.shape
    if n == 0:
        return
    _check_head_dim(D, 32)
    assert x.stride(2) == 1 and data.is_contiguous() and meta.is_contiguous()
    assert data.shape[1:] == (H, D // 4) and meta.shape[1:] == (H, D // _G, 2)
    _quant_int2_v_tokens_kernel[(n, H)](
        x,
        rows,
        data,
        meta,
        x.stride(0),
        x.stride(1),
        H,
        D=D,
        NG=D // _G,
        num_warps=1,
    )


def quant_int2_k_pages(x, tok_rows, pages, data, meta):
    """x [n,H,D] (all new K of this call); tok_rows int32 [num_pages, 32] = row indices into x
    for the 32 tokens of each page in j order; pages int64 [num_pages] = page ids.
    Per (page, head, channel): min/max over the 32 tokens -> scale/min -> pack 8 bytes.
    Writes data[pages] (uint8 [.,H,D,8]) and meta[pages] (fp16 [.,H,D,2])."""
    n, H, D = x.shape
    num_pages = pages.shape[0]
    if num_pages == 0:
        return
    _check_head_dim(D, 32)
    assert x.stride(2) == 1 and data.is_contiguous() and meta.is_contiguous()
    assert tok_rows.shape == (num_pages, _G) and tok_rows.is_contiguous()
    assert data.shape[1:] == (H, D, 8) and meta.shape[1:] == (H, D, 2)
    _quant_int2_k_pages_kernel[(num_pages, H)](
        x,
        tok_rows,
        pages,
        data,
        meta,
        x.stride(0),
        x.stride(1),
        H,
        D=D,
        num_warps=2,
    )


# =============================================================================
# Dequantizing tile loaders (shared by dequant_gather and the decode kernel)
#
# Channels are handled as four interleaved sub-tiles: sub-tile c holds channels
# d = 4b + c (b in [0, D/4)). This matches the INT2 V byte packing (one byte = four
# consecutive channels) so every byte is loaded once, and it keeps the meta group of a
# sub-tile row independent of c (group(4b + c) = b // 8). Loaders return four fp32 tiles:
# K sub-tiles are [D4, BLOCK_N] (channel-major, as tl.dot(q, k) wants), V sub-tiles are
# [BLOCK_N, D4]. "loc" is a [BLOCK_N] int64 tensor of slot ids; "mask_n" [BLOCK_N] marks
# the lanes to load; masked lanes come back finite (0 for per-token formats) and must be
# excluded by the caller.
# =============================================================================


@triton.jit
def _expand_rows(m, NG: tl.constexpr, R: tl.constexpr, N: tl.constexpr):
    # [NG, N] -> [NG * R, N]; row b takes group b // R
    return tl.reshape(tl.broadcast_to(m[:, None, :], [NG, R, N]), [NG * R, N])


@triton.jit
def _expand_cols(m, N: tl.constexpr, NG: tl.constexpr, R: tl.constexpr):
    # [N, NG] -> [N, NG * R]; column b takes group b // R
    return tl.reshape(tl.broadcast_to(m[:, :, None], [N, NG, R]), [N, NG * R])


@triton.jit
def _k4_tiles(
    K4D, K4M, loc, offset, cur_kv_head, H, mask_n,
    D: tl.constexpr, NG: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # INT4 per-token keys: byte b' of a row holds channels 2b' (low nibble), 2b'+1 (high).
    D4: tl.constexpr = D // 4
    b = tl.arange(0, D4)
    g = tl.arange(0, NG)
    row = (loc - offset) * H + cur_kv_head  # [N]
    base = row[None, :] * (D // 2) + 2 * b[:, None]  # [D4, N]: byte of channels 4b, 4b+1
    m2 = mask_n[None, :] & (b[:, None] >= 0)
    B0 = tl.load(K4D + base, mask=m2, other=0).to(tl.int32)
    B1 = tl.load(K4D + base + 1, mask=m2, other=0).to(tl.int32)
    moff = (row[None, :] * NG + g[:, None]) * 2  # [NG, N]
    mm = mask_n[None, :] & (g[:, None] >= 0)
    sc = _expand_rows(tl.load(K4M + moff, mask=mm, other=0.0).to(tl.float32), NG, 8, BLOCK_N)
    mn = _expand_rows(tl.load(K4M + moff + 1, mask=mm, other=0.0).to(tl.float32), NG, 8, BLOCK_N)
    k0 = (B0 & 15).to(tl.float32) * sc + mn
    k1 = ((B0 >> 4) & 15).to(tl.float32) * sc + mn
    k2 = (B1 & 15).to(tl.float32) * sc + mn
    k3 = ((B1 >> 4) & 15).to(tl.float32) * sc + mn
    return k0, k1, k2, k3


@triton.jit
def _v4_tiles(
    V4D, V4M, loc, offset, cur_kv_head, H, mask_n,
    D: tl.constexpr, NG: tl.constexpr, BLOCK_N: tl.constexpr,
):
    D4: tl.constexpr = D // 4
    b = tl.arange(0, D4)
    g = tl.arange(0, NG)
    row = (loc - offset) * H + cur_kv_head  # [N]
    base = row[:, None] * (D // 2) + 2 * b[None, :]  # [N, D4]
    m2 = mask_n[:, None] & (b[None, :] >= 0)
    B0 = tl.load(V4D + base, mask=m2, other=0).to(tl.int32)
    B1 = tl.load(V4D + base + 1, mask=m2, other=0).to(tl.int32)
    moff = (row[:, None] * NG + g[None, :]) * 2  # [N, NG]
    mm = mask_n[:, None] & (g[None, :] >= 0)
    sc = _expand_cols(tl.load(V4M + moff, mask=mm, other=0.0).to(tl.float32), BLOCK_N, NG, 8)
    mn = _expand_cols(tl.load(V4M + moff + 1, mask=mm, other=0.0).to(tl.float32), BLOCK_N, NG, 8)
    v0 = (B0 & 15).to(tl.float32) * sc + mn
    v1 = ((B0 >> 4) & 15).to(tl.float32) * sc + mn
    v2 = (B1 & 15).to(tl.float32) * sc + mn
    v3 = ((B1 >> 4) & 15).to(tl.float32) * sc + mn
    return v0, v1, v2, v3


@triton.jit
def _v2_tiles(
    V2D, V2M, loc, cur_kv_head, H, mask_n,
    D: tl.constexpr, NG: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # INT2 per-token values: byte b holds channels 4b..4b+3, channel 4b+c at bits 2c.
    D4: tl.constexpr = D // 4
    b = tl.arange(0, D4)
    g = tl.arange(0, NG)
    row = loc * H + cur_kv_head  # [N]
    base = row[:, None] * D4 + b[None, :]  # [N, D4]
    m2 = mask_n[:, None] & (b[None, :] >= 0)
    B = tl.load(V2D + base, mask=m2, other=0).to(tl.int32)
    moff = (row[:, None] * NG + g[None, :]) * 2  # [N, NG]
    mm = mask_n[:, None] & (g[None, :] >= 0)
    sc = _expand_cols(tl.load(V2M + moff, mask=mm, other=0.0).to(tl.float32), BLOCK_N, NG, 8)
    mn = _expand_cols(tl.load(V2M + moff + 1, mask=mm, other=0.0).to(tl.float32), BLOCK_N, NG, 8)
    v0 = (B & 3).to(tl.float32) * sc + mn
    v1 = ((B >> 2) & 3).to(tl.float32) * sc + mn
    v2 = ((B >> 4) & 3).to(tl.float32) * sc + mn
    v3 = ((B >> 6) & 3).to(tl.float32) * sc + mn
    return v0, v1, v2, v3


@triton.jit
def _k2_tiles(
    K2D, K2M, loc, cur_kv_head, H, mask_n,
    D: tl.constexpr, NG: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # INT2 per-channel keys: page = s // 32, j = s % 32; byte k2_data[page, h, d, j // 4],
    # token j at bits 2 * (j % 4); meta per (page, h, d).
    D4: tl.constexpr = D // 4
    b = tl.arange(0, D4)
    page = loc // 32
    j = loc % 32
    pbase = (page * H + cur_kv_head) * D  # [N] int64: index of (page, h, channel 0)
    bptr = K2D + (pbase * 8 + j // 4)[None, :] + (32 * b)[:, None]  # [D4, N]: byte of channel 4b
    shift = (2 * (j % 4)).to(tl.int32)[None, :]
    m2 = mask_n[None, :] & (b[:, None] >= 0)
    q0 = (tl.load(bptr, mask=m2, other=0).to(tl.int32) >> shift) & 3
    q1 = (tl.load(bptr + 8, mask=m2, other=0).to(tl.int32) >> shift) & 3
    q2 = (tl.load(bptr + 16, mask=m2, other=0).to(tl.int32) >> shift) & 3
    q3 = (tl.load(bptr + 24, mask=m2, other=0).to(tl.int32) >> shift) & 3

    # Meta: a block of position-ordered tokens touches at most two pages, so load the
    # [scale, min] vectors of those two pages once and select per token. Fall back to a
    # per-element gather when the block holds tokens from more than two pages.
    p_hi = tl.max(tl.where(mask_n, page, 0))
    p_lo = tl.min(tl.where(mask_n, page, p_hi))
    on_two = tl.where(mask_n, ((page == p_lo) | (page == p_hi)).to(tl.int32), 1)
    if tl.min(on_two) == 1:
        sel = (page == p_lo)[None, :]  # [1, N]
        cb_lo = (((p_lo * H + cur_kv_head) * D) + 4 * b) * 2  # [D4]: meta of channel 4b, page p_lo
        cb_hi = (((p_hi * H + cur_kv_head) * D) + 4 * b) * 2
        sc0 = tl.where(sel, tl.load(K2M + cb_lo).to(tl.float32)[:, None], tl.load(K2M + cb_hi).to(tl.float32)[:, None])
        mn0 = tl.where(sel, tl.load(K2M + cb_lo + 1).to(tl.float32)[:, None], tl.load(K2M + cb_hi + 1).to(tl.float32)[:, None])
        sc1 = tl.where(sel, tl.load(K2M + cb_lo + 2).to(tl.float32)[:, None], tl.load(K2M + cb_hi + 2).to(tl.float32)[:, None])
        mn1 = tl.where(sel, tl.load(K2M + cb_lo + 3).to(tl.float32)[:, None], tl.load(K2M + cb_hi + 3).to(tl.float32)[:, None])
        sc2 = tl.where(sel, tl.load(K2M + cb_lo + 4).to(tl.float32)[:, None], tl.load(K2M + cb_hi + 4).to(tl.float32)[:, None])
        mn2 = tl.where(sel, tl.load(K2M + cb_lo + 5).to(tl.float32)[:, None], tl.load(K2M + cb_hi + 5).to(tl.float32)[:, None])
        sc3 = tl.where(sel, tl.load(K2M + cb_lo + 6).to(tl.float32)[:, None], tl.load(K2M + cb_hi + 6).to(tl.float32)[:, None])
        mn3 = tl.where(sel, tl.load(K2M + cb_lo + 7).to(tl.float32)[:, None], tl.load(K2M + cb_hi + 7).to(tl.float32)[:, None])
    else:
        mptr = K2M + (pbase * 2)[None, :] + (8 * b)[:, None]  # [D4, N]: meta of channel 4b, token n's page
        sc0 = tl.load(mptr, mask=m2, other=0.0).to(tl.float32)
        mn0 = tl.load(mptr + 1, mask=m2, other=0.0).to(tl.float32)
        sc1 = tl.load(mptr + 2, mask=m2, other=0.0).to(tl.float32)
        mn1 = tl.load(mptr + 3, mask=m2, other=0.0).to(tl.float32)
        sc2 = tl.load(mptr + 4, mask=m2, other=0.0).to(tl.float32)
        mn2 = tl.load(mptr + 5, mask=m2, other=0.0).to(tl.float32)
        sc3 = tl.load(mptr + 6, mask=m2, other=0.0).to(tl.float32)
        mn3 = tl.load(mptr + 7, mask=m2, other=0.0).to(tl.float32)
    # Masked lanes carry q = 0 and (in the two-page path) a real page's [scale, min], i.e. a
    # finite value; callers never use them (qk is masked to -inf, the gather masks its store).
    k0 = q0.to(tl.float32) * sc0 + mn0
    k1 = q1.to(tl.float32) * sc1 + mn1
    k2 = q2.to(tl.float32) * sc2 + mn2
    k3 = q3.to(tl.float32) * sc3 + mn3
    return k0, k1, k2, k3


# =============================================================================
# dequant_gather
# =============================================================================


@triton.jit
def _dequant_gather_kernel(
    Slots,
    m,
    K2D, K2M, V2D, V2M, K4D, K4M, V4D, V4M,
    OutK,
    OutV,
    offset,
    H,
    stride_on,
    stride_oh,
    D: tl.constexpr,
    NG: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    D4: tl.constexpr = D // 4
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < m
    loc = tl.load(Slots + offs_n, mask=mask_n, other=0).to(tl.int64)
    b = tl.arange(0, D4)
    is2 = loc < offset
    mask2 = mask_n & is2
    mask4 = mask_n & (~is2)

    k2_0, k2_1, k2_2, k2_3 = _k2_tiles(K2D, K2M, loc, h, H, mask2, D, NG, BLOCK_N)
    k4_0, k4_1, k4_2, k4_3 = _k4_tiles(K4D, K4M, loc, offset, h, H, mask4, D, NG, BLOCK_N)
    v2_0, v2_1, v2_2, v2_3 = _v2_tiles(V2D, V2M, loc, h, H, mask2, D, NG, BLOCK_N)
    v4_0, v4_1, v4_2, v4_3 = _v4_tiles(V4D, V4M, loc, offset, h, H, mask4, D, NG, BLOCK_N)
    s2k = is2[None, :]
    s2v = is2[:, None]

    out_base = OutK + offs_n[:, None] * stride_on + h * stride_oh + 4 * b[None, :]
    outv_base = OutV + offs_n[:, None] * stride_on + h * stride_oh + 4 * b[None, :]
    om = mask_n[:, None] & (b[None, :] >= 0)
    ok = OutK.dtype.element_ty
    tl.store(out_base, tl.trans(tl.where(s2k, k2_0, k4_0)).to(ok), mask=om)
    tl.store(out_base + 1, tl.trans(tl.where(s2k, k2_1, k4_1)).to(ok), mask=om)
    tl.store(out_base + 2, tl.trans(tl.where(s2k, k2_2, k4_2)).to(ok), mask=om)
    tl.store(out_base + 3, tl.trans(tl.where(s2k, k2_3, k4_3)).to(ok), mask=om)
    ov = OutV.dtype.element_ty
    tl.store(outv_base, tl.where(s2v, v2_0, v4_0).to(ov), mask=om)
    tl.store(outv_base + 1, tl.where(s2v, v2_1, v4_1).to(ov), mask=om)
    tl.store(outv_base + 2, tl.where(s2v, v2_2, v4_2).to(ov), mask=om)
    tl.store(outv_base + 3, tl.where(s2v, v2_3, v4_3).to(ov), mask=om)


def dequant_gather(slots, kv: TriaxialLayerKV, out_k, out_v):
    """slots int64 [m] (any mix of INT2/INT4 slots); out_k/out_v bf16 [m,H,D], written in place."""
    m = slots.shape[0]
    if m == 0:
        return
    H, D = kv.num_kv_heads, kv.head_dim
    _check_head_dim(D)
    assert out_k.shape == (m, H, D) and out_v.shape == (m, H, D)
    assert out_k.stride(2) == 1 and out_v.stride(2) == 1
    assert out_k.stride(0) == out_v.stride(0) and out_k.stride(1) == out_v.stride(1)
    BLOCK_N = 32
    grid = (triton.cdiv(m, BLOCK_N), H)
    _dequant_gather_kernel[grid](
        slots,
        m,
        kv.k2_data, kv.k2_meta, kv.v2_data, kv.v2_meta,
        kv.k4_data, kv.k4_meta, kv.v4_data, kv.v4_meta,
        out_k,
        out_v,
        kv.offset,
        H,
        out_k.stride(0),
        out_k.stride(1),
        D=D,
        NG=D // _G,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


# =============================================================================
# Decode attention (fork of sglang decode_attention.py, GQA path, mixed-precision K/V)
# =============================================================================


@triton.jit
def _tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_grouped_kernel_stage1_mixed(
    Q,
    K2D, K2M, V2D, V2M, K4D, K4M, V4D, V4M,
    offset,
    H,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    D: tl.constexpr,
    NG: tl.constexpr,
):
    # Same program layout / split-KV bookkeeping as _fwd_grouped_kernel_stage1
    # (Lk == Lv == D, BLOCK_DPE = 0). Channels are processed as four interleaved
    # sub-tiles (channel 4b + c) so the packed formats can be unpacked in registers.
    D4: tl.constexpr = D // 4
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_b = tl.arange(0, D4)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    # q sub-tiles: q_c[h, b] = q[h, 4b + c]
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + 4 * offs_b[None, :]
    mask_q = mask_h[:, None] & (offs_b[None, :] >= 0)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_H, D4], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, D4], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, D4], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, D4], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q0 = tl.load(Q + offs_q, mask=mask_q, other=0.0)
        q1 = tl.load(Q + offs_q + 1, mask=mask_q, other=0.0)
        q2 = tl.load(Q + offs_q + 2, mask=mask_q, other=0.0)
        q3 = tl.load(Q + offs_q + 3, mask=mask_q, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=mask_n,
                other=0,
            ).to(tl.int64)
            is2 = kv_loc < offset
            # Block classification on the valid lanes only.
            loc_hi = tl.max(tl.where(mask_n, kv_loc, 0))
            loc_lo = tl.min(tl.where(mask_n, kv_loc, offset))

            if loc_hi < offset:
                # every valid slot is INT2
                k0, k1, k2, k3 = _k2_tiles(K2D, K2M, kv_loc, cur_kv_head, H, mask_n, D, NG, BLOCK_N)
                v0, v1, v2, v3 = _v2_tiles(V2D, V2M, kv_loc, cur_kv_head, H, mask_n, D, NG, BLOCK_N)
            elif loc_lo >= offset:
                # every valid slot is INT4
                k0, k1, k2, k3 = _k4_tiles(K4D, K4M, kv_loc, offset, cur_kv_head, H, mask_n, D, NG, BLOCK_N)
                v0, v1, v2, v3 = _v4_tiles(V4D, V4M, kv_loc, offset, cur_kv_head, H, mask_n, D, NG, BLOCK_N)
            else:
                # mixed block: masked loads of both kinds, then select per slot
                m2 = mask_n & is2
                m4 = mask_n & (~is2)
                a0, a1, a2, a3 = _k2_tiles(K2D, K2M, kv_loc, cur_kv_head, H, m2, D, NG, BLOCK_N)
                c0, c1, c2, c3 = _k4_tiles(K4D, K4M, kv_loc, offset, cur_kv_head, H, m4, D, NG, BLOCK_N)
                s = is2[None, :]
                k0 = tl.where(s, a0, c0)
                k1 = tl.where(s, a1, c1)
                k2 = tl.where(s, a2, c2)
                k3 = tl.where(s, a3, c3)
                a0, a1, a2, a3 = _v2_tiles(V2D, V2M, kv_loc, cur_kv_head, H, m2, D, NG, BLOCK_N)
                c0, c1, c2, c3 = _v4_tiles(V4D, V4M, kv_loc, offset, cur_kv_head, H, m4, D, NG, BLOCK_N)
                s = is2[:, None]
                v0 = tl.where(s, a0, c0)
                v1 = tl.where(s, a1, c1)
                v2 = tl.where(s, a2, c2)
                v3 = tl.where(s, a3, c3)

            qk = tl.dot(q0, k0.to(q0.dtype))
            qk += tl.dot(q1, k1.to(q0.dtype))
            qk += tl.dot(q2, k2.to(q0.dtype))
            qk += tl.dot(q3, k3.to(q0.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc0 *= re_scale[:, None]
            acc1 *= re_scale[:, None]
            acc2 *= re_scale[:, None]
            acc3 *= re_scale[:, None]
            p16 = p.to(q0.dtype)
            acc0 += tl.dot(p16, v0.to(q0.dtype))
            acc1 += tl.dot(p16, v1.to(q0.dtype))
            acc2 += tl.dot(p16, v2.to(q0.dtype))
            acc3 += tl.dot(p16, v3.to(q0.dtype))

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + 4 * offs_b[None, :]
        )
        tl.store(Att_Out + offs_mid_o, acc0 / e_sum[:, None], mask=mask_q)
        tl.store(Att_Out + offs_mid_o + 1, acc1 / e_sum[:, None], mask=mask_q)
        tl.store(Att_Out + offs_mid_o + 2, acc2 / e_sum[:, None], mask=mask_q)
        tl.store(Att_Out + offs_mid_o + 3, acc3 / e_sum[:, None], mask=mask_q)

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // D

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd_mixed(
    q,
    kv: TriaxialLayerKV,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    H, D = kv.num_kv_heads, kv.head_dim

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num // H
    assert kv_group_num * H == head_num

    BLOCK_H = 16
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        max_kv_splits,
    )

    _fwd_grouped_kernel_stage1_mixed[grid](
        q,
        kv.k2_data, kv.k2_meta, kv.v2_data, kv.v2_meta,
        kv.k4_data, kv.k4_meta, kv.v4_data, kv.v4_meta,
        kv.offset,
        H,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        D=D,
        NG=D // _G,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    kv_indptr,
    num_kv_splits,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    # Fork of decode_attention._fwd_kernel_stage2 with v_scale = 1 and no sinks.
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    Lv,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        kv_indptr,
        num_kv_splits,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=max_kv_splits,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )


def decode_attention_fwd_mixed(
    q,
    kv: TriaxialLayerKV,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    """Drop-in replacement for decode_attention.decode_attention_fwd (GQA/MHA, Lk == Lv == D,
    no rope split, no sinks, v_scale = 1) reading a TriaxialLayerKV instead of dense buffers.

    q: [bs, num_q_heads, D] bf16/fp16; o: same shape. kv_indptr int32 [bs+1], kv_indices
    int64/int32 [sum seq_lens] of slot ids; attn_logits fp32 [bs, num_q_heads, max_kv_splits, D];
    attn_lse fp32 [bs, num_q_heads, max_kv_splits]; num_kv_splits int32 [bs].
    """
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]
    D = kv.head_dim
    assert q.shape[-1] == D and o.shape == q.shape
    assert attn_logits.shape[-1] == D
    assert q.stride(-1) == 1 and o.stride(-1) == 1

    _decode_grouped_att_m_fwd_mixed(
        q,
        kv,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        D,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
    )


# =============================================================================
# Pure-torch references (for tests). All return exactly what the kernels write.
# =============================================================================


def _ref_scale_min(x_fp32: torch.Tensor, qmax: int):
    """x_fp32 [..., n] -> (scale, min) fp16 tensors [...] over the last dim, kernel rules."""
    mx = x_fp32.amax(dim=-1)
    mn = x_fp32.amin(dim=-1)
    sc = ((mx - mn) / float(qmax)).to(torch.float16)
    sc = torch.where(sc == 0, torch.ones_like(sc), sc)
    return sc, mn.to(torch.float16)


def _ref_quantize(x_fp32: torch.Tensor, sc: torch.Tensor, mn: torch.Tensor, qmax: int):
    """Same rounding as the kernels (round half to even, IEEE division)."""
    q = ((x_fp32 - mn.float()) / sc.float()).round().clamp_(0, qmax)
    return q.to(torch.int32)


def ref_quant_int4_tokens(x: torch.Tensor):
    """x [n,H,D] -> (data uint8 [n,H,D//2], meta fp16 [n,H,D//32,2])."""
    n, H, D = x.shape
    xf = x.float().reshape(n, H, D // _G, _G)
    sc, mn = _ref_scale_min(xf, 15)  # [n,H,NG]
    q = _ref_quantize(xf, sc[..., None], mn[..., None], 15).reshape(n, H, D // 2, 2)
    data = (q[..., 0] | (q[..., 1] << 4)).to(torch.uint8)
    meta = torch.stack([sc, mn], dim=-1)
    return data, meta


def ref_quant_int2_v_tokens(x: torch.Tensor):
    """x [n,H,D] -> (data uint8 [n,H,D//4], meta fp16 [n,H,D//32,2])."""
    n, H, D = x.shape
    xf = x.float().reshape(n, H, D // _G, _G)
    sc, mn = _ref_scale_min(xf, 3)
    q = _ref_quantize(xf, sc[..., None], mn[..., None], 3).reshape(n, H, D // 4, 4)
    data = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).to(
        torch.uint8
    )
    meta = torch.stack([sc, mn], dim=-1)
    return data, meta


def ref_quant_int2_k_pages(x_pages: torch.Tensor):
    """x_pages [num_pages, 32, H, D] (tokens of each page in j order)
    -> (data uint8 [num_pages,H,D,8], meta fp16 [num_pages,H,D,2])."""
    P, G, H, D = x_pages.shape
    assert G == _G
    xf = x_pages.float().permute(0, 2, 3, 1)  # [P,H,D,32]
    sc, mn = _ref_scale_min(xf, 3)  # [P,H,D]
    q = _ref_quantize(xf, sc[..., None], mn[..., None], 3).reshape(P, H, D, 8, 4)
    data = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).to(
        torch.uint8
    )
    meta = torch.stack([sc, mn], dim=-1)
    return data, meta


def ref_unpack_int4(data: torch.Tensor):
    """uint8 [..., D//2] -> int32 levels [..., D]."""
    d = data.to(torch.int32)
    return torch.stack([d & 15, (d >> 4) & 15], dim=-1).flatten(-2)


def ref_unpack_int2(data: torch.Tensor):
    """uint8 [..., D//4] -> int32 levels [..., D] (LSB pair first)."""
    d = data.to(torch.int32)
    return torch.stack([d & 3, (d >> 2) & 3, (d >> 4) & 3, (d >> 6) & 3], dim=-1).flatten(
        -2
    )


def ref_dequant_tokens(data: torch.Tensor, meta: torch.Tensor, bits: int):
    """Per-token layout: data [n,H,D//(8/bits)], meta [n,H,D//32,2] -> fp32 [n,H,D]."""
    q = (ref_unpack_int4(data) if bits == 4 else ref_unpack_int2(data)).float()
    n, H, D = q.shape
    q = q.reshape(n, H, D // _G, _G)
    x = q * meta[..., 0:1].float() + meta[..., 1:2].float()
    return x.reshape(n, H, D)


def ref_dequant_k_pages(data: torch.Tensor, meta: torch.Tensor):
    """data [P,H,D,8], meta [P,H,D,2] -> fp32 [P,32,H,D]."""
    q = ref_unpack_int2(data).float()  # [P,H,D,32]
    x = q * meta[..., 0:1].float() + meta[..., 1:2].float()
    return x.permute(0, 3, 1, 2).contiguous()


def ref_dequant_gather(slots: torch.Tensor, kv: TriaxialLayerKV):
    """slots int64 [m] -> (k, v) fp32 [m,H,D] dequantized with the same formula as the kernels."""
    m = slots.shape[0]
    H, D = kv.num_kv_heads, kv.head_dim
    dev = slots.device
    k = torch.zeros(m, H, D, dtype=torch.float32, device=dev)
    v = torch.zeros(m, H, D, dtype=torch.float32, device=dev)
    is2 = slots < kv.offset
    s2 = slots[is2]
    if s2.numel():
        page, j = s2 // _G, s2 % _G
        byte = kv.k2_data[page][:, :, :, :]  # [m2,H,D,8]
        byte = byte.gather(3, (j // 4)[:, None, None, None].expand(-1, H, D, 1))[..., 0]
        q = (byte.to(torch.int32) >> (2 * (j % 4))[:, None, None].to(torch.int32)) & 3
        meta = kv.k2_meta[page].float()  # [m2,H,D,2]
        k[is2] = q.float() * meta[..., 0] + meta[..., 1]
        v[is2] = ref_dequant_tokens(kv.v2_data[s2], kv.v2_meta[s2], 2)
    s4 = slots[~is2] - kv.offset
    if s4.numel():
        k[~is2] = ref_dequant_tokens(kv.k4_data[s4], kv.k4_meta[s4], 4)
        v[~is2] = ref_dequant_tokens(kv.v4_data[s4], kv.v4_meta[s4], 4)
    return k, v


def ref_attention(q, k, v, kv_indptr, sm_scale, logit_cap=0.0):
    """q [bs,Hq,D] fp32; k, v [total,H,D] fp32 in kv_indptr order -> o [bs,Hq,D] fp32."""
    bs, Hq, D = q.shape
    H = k.shape[1]
    group = Hq // H
    indptr = kv_indptr.tolist()
    o = torch.empty_like(q)
    for b in range(bs):
        kb = k[indptr[b] : indptr[b + 1]]  # [n,H,D]
        vb = v[indptr[b] : indptr[b + 1]]
        kb = kb.repeat_interleave(group, dim=1)  # [n,Hq,D]
        vb = vb.repeat_interleave(group, dim=1)
        s = torch.einsum("hd,nhd->hn", q[b], kb) * sm_scale
        if logit_cap > 0:
            s = logit_cap * torch.tanh(s / logit_cap)
        p = torch.softmax(s, dim=-1)
        o[b] = torch.einsum("hn,nhd->hd", p, vb)
    return o


# Spec §3 names: quantize + dequantize round trips.
def ref_quant_dequant_int4(x):
    return ref_dequant_tokens(*ref_quant_int4_tokens(x), 4)


def ref_quant_dequant_int2_v(x):
    return ref_dequant_tokens(*ref_quant_int2_v_tokens(x), 2)


def ref_quant_dequant_int2_k_page(x_pages):
    return ref_dequant_k_pages(*ref_quant_int2_k_pages(x_pages))
