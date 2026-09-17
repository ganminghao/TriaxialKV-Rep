"""TriAxialKV: mixed INT2/INT4 paged KV pool and its allocator.

Layouts and conventions are defined in TRIAXIAL_DESIGN.md (repo root). In short:

* one slot index space; slots ``[0, offset)`` are INT2 (page = slot // 32, in-page
  index = slot % 32), slots ``[offset, offset + size4)`` are INT4 (row = slot - offset);
* INT2 keys are quantized per channel over the 32 tokens of a page, INT2 values and all
  INT4 entries per token in groups of 32 channels;
* page 0 and INT4 row 0 are reserved dummies (SGLang writes padded tokens to slot 0).

The allocator presents ``page_size == 1`` to the rest of SGLang (radix cache, scheduler)
and groups INT2 tokens into pages internally.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.attention.triton_ops.triaxial_kv import (
    TriaxialLayerKV,
    quant_int2_k_pages,
    quant_int2_v_tokens,
    quant_int4_tokens,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

logger = logging.getLogger(__name__)

PAGE = 32  # INT2 page size == quantization group size
GB = 1024**3


def triaxial_bytes_per_token_layer(head_num: int, head_dim: int, bits: int) -> int:
    """Bytes for K+V of one token in one layer (data + fp16 scale/min)."""
    assert bits in (2, 4)
    groups = head_dim // 32
    if bits == 4:
        per_head = head_dim // 2 + groups * 4
    else:
        # K: per-channel over 32 tokens -> (D*8 + D*4) bytes per page-head = 12*D/32 per token
        # V: per-token -> D/4 + groups*4
        per_head = (head_dim * 12) // 32 + head_dim // 4 + groups * 4
    return 2 * head_num * per_head if bits == 4 else head_num * per_head


def triaxial_cell_size(
    head_num: int, head_dim: int, num_layers: int, int2_fraction: float
) -> int:
    """Average bytes per token over all layers for the given INT2 slot fraction."""
    b2 = triaxial_bytes_per_token_layer(head_num, head_dim, 2)
    b4 = triaxial_bytes_per_token_layer(head_num, head_dim, 4)
    return int(round((int2_fraction * b2 + (1.0 - int2_fraction) * b4) * num_layers))


class TriaxialKVPool(KVCache):
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        int2_fraction: float,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        assert page_size == 1, "TriAxialKV requires --page-size 1 (INT2 paging is internal)"
        assert head_dim % 32 == 0, "head_dim must be a multiple of the group size 32"
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.head_num = head_num
        self.head_dim = head_dim
        self.int2_fraction = int2_fraction

        # Usable slots: `size`. Split by fraction, INT2 side rounded down to whole pages.
        usable_pages2 = int(size * int2_fraction) // PAGE
        self.usable_int2 = usable_pages2 * PAGE
        self.usable_int4 = size - self.usable_int2
        self.num_pages2 = usable_pages2 + 1  # + dummy page 0
        self.offset = self.num_pages2 * PAGE  # first INT4 slot
        self.size2 = self.offset
        self.size4 = self.usable_int4 + 1  # + dummy row 0
        self.num_slots = self.offset + self.size4

        self._create_buffers()

        # per-batch cached grouping of the last `loc` seen by set_kv_buffer
        self._grp_loc: Optional[torch.Tensor] = None
        self._grp: Optional[Tuple[torch.Tensor, ...]] = None

        self._finalize_allocation_log(size)
        logger.info(
            "TriAxialKV pool: %d INT2 slots (%d pages of %d) + %d INT4 slots, "
            "offset=%d, int2_fraction=%.3f",
            self.usable_int2,
            usable_pages2,
            PAGE,
            self.usable_int4,
            self.offset,
            int2_fraction,
        )

    # ------------------------------------------------------------------ buffers
    def _create_buffers(self):
        L, H, D = self.layer_num, self.head_num, self.head_dim
        G = D // 32
        dev = self.device
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k2_data = torch.zeros((L, self.num_pages2, H, D, 8), dtype=torch.uint8, device=dev)
            self.k2_meta = torch.zeros((L, self.num_pages2, H, D, 2), dtype=torch.float16, device=dev)
            self.v2_data = torch.zeros((L, self.size2, H, D // 4), dtype=torch.uint8, device=dev)
            self.v2_meta = torch.zeros((L, self.size2, H, G, 2), dtype=torch.float16, device=dev)
            self.k4_data = torch.zeros((L, self.size4, H, D // 2), dtype=torch.uint8, device=dev)
            self.k4_meta = torch.zeros((L, self.size4, H, G, 2), dtype=torch.float16, device=dev)
            self.v4_data = torch.zeros((L, self.size4, H, D // 2), dtype=torch.uint8, device=dev)
            self.v4_meta = torch.zeros((L, self.size4, H, G, 2), dtype=torch.float16, device=dev)
        self._layer_views = [
            TriaxialLayerKV(
                offset=self.offset,
                k2_data=self.k2_data[i],
                k2_meta=self.k2_meta[i],
                v2_data=self.v2_data[i],
                v2_meta=self.v2_meta[i],
                k4_data=self.k4_data[i],
                k4_meta=self.k4_meta[i],
                v4_data=self.v4_data[i],
                v4_meta=self.v4_meta[i],
            )
            for i in range(L)
        ]

    def _clear_buffers(self):
        for name in (
            "k2_data", "k2_meta", "v2_data", "v2_meta",
            "k4_data", "k4_meta", "v4_data", "v4_meta", "_layer_views",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def get_kv_size_bytes(self):
        total = 0
        for t in (
            self.k2_data, self.k2_meta, self.v2_data, self.v2_meta,
            self.k4_data, self.k4_meta, self.v4_data, self.v4_meta,
        ):
            total += t.numel() * t.element_size()
        return total

    def layer_kv(self, layer_id: int) -> TriaxialLayerKV:
        return self._layer_views[layer_id - self.start_layer]

    def get_v_head_dim(self) -> int:
        return self.head_dim

    # dense views do not exist for this pool
    def get_key_buffer(self, layer_id: int):
        raise RuntimeError("TriaxialKVPool has no dense key buffer; use layer_kv()")

    def get_value_buffer(self, layer_id: int):
        raise RuntimeError("TriaxialKVPool has no dense value buffer; use layer_kv()")

    def get_kv_buffer(self, layer_id: int):
        raise RuntimeError("TriaxialKVPool has no dense KV buffer; use layer_kv()")

    # ------------------------------------------------------------------ writes
    def _grouping(self, loc: torch.Tensor):
        """Split `loc` into INT4 rows, INT2 value rows and INT2 K pages.

        Cached per `loc` tensor object so the (synchronizing) mask work happens once per
        forward batch rather than once per layer.
        """
        if self._grp_loc is loc and self._grp is not None:
            return self._grp
        orig_loc = loc
        loc = loc.to(torch.int64)
        is4 = loc >= self.offset
        idx4 = torch.nonzero(is4, as_tuple=True)[0]
        rows4 = loc[idx4] - self.offset
        # INT2 tokens outside the dummy page 0
        is2 = (~is4) & (loc >= PAGE)
        idx2 = torch.nonzero(is2, as_tuple=True)[0]
        slots2 = loc[idx2]
        if slots2.numel() > 0:
            order = torch.argsort(slots2)
            sorted_slots = slots2[order]
            n_full = sorted_slots.numel() // PAGE
            # every INT2 page is written whole in one call (allocator invariant)
            assert sorted_slots.numel() % PAGE == 0, (
                f"INT2 slots in set_kv_buffer are not whole pages: {sorted_slots.numel()}"
            )
            tok_rows = idx2[order].view(n_full, PAGE).to(torch.int32)
            pages = sorted_slots.view(n_full, PAGE)[:, 0] // PAGE
        else:
            tok_rows = torch.empty((0, PAGE), dtype=torch.int32, device=loc.device)
            pages = torch.empty((0,), dtype=torch.int64, device=loc.device)
        self._grp_loc = orig_loc
        self._grp = (idx4, rows4, idx2, slots2, tok_rows, pages)
        return self._grp

    def set_kv_buffer(
        self,
        layer,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        layer_id = layer.layer_id if layer_id_override is None else layer_id_override
        kv = self.layer_kv(layer_id)
        cache_k = cache_k.view(-1, self.head_num, self.head_dim)
        cache_v = cache_v.view(-1, self.head_num, self.head_dim)
        if cache_k.dtype not in (torch.bfloat16, torch.float16):
            cache_k = cache_k.to(torch.bfloat16)
            cache_v = cache_v.to(torch.bfloat16)
        idx4, rows4, idx2, slots2, tok_rows, pages = self._grouping(loc)
        if rows4.numel() > 0:
            k4 = cache_k[idx4].contiguous()
            v4 = cache_v[idx4].contiguous()
            quant_int4_tokens(k4, rows4, kv.k4_data, kv.k4_meta)
            quant_int4_tokens(v4, rows4, kv.v4_data, kv.v4_meta)
        if slots2.numel() > 0:
            v2 = cache_v[idx2].contiguous()
            quant_int2_v_tokens(v2, slots2, kv.v2_data, kv.v2_meta)
            quant_int2_k_pages(cache_k.contiguous(), tok_rows, pages, kv.k2_data, kv.k2_meta)

    def set_kv_buffer_int4(self, layer_id: int, loc: torch.Tensor, cache_k, cache_v):
        """Fast path for decode batches: every slot in `loc` is INT4 (no host sync)."""
        kv = self.layer_kv(layer_id)
        # padded / CUDA-graph-capture tokens use slot 0 -> route them to the dummy INT4 row 0
        rows = torch.clamp_min(loc.to(torch.int64) - self.offset, 0)
        cache_k = cache_k.view(-1, self.head_num, self.head_dim)
        cache_v = cache_v.view(-1, self.head_num, self.head_dim)
        if cache_k.dtype not in (torch.bfloat16, torch.float16):
            cache_k = cache_k.to(torch.bfloat16)
            cache_v = cache_v.to(torch.bfloat16)
        quant_int4_tokens(cache_k.contiguous(), rows, kv.k4_data, kv.k4_meta)
        quant_int4_tokens(cache_v.contiguous(), rows, kv.v4_data, kv.v4_meta)


class TriaxialAllocator(BaseTokenToKVPoolAllocator):
    """Token allocator over the two regions of a TriaxialKVPool.

    Outside view: page_size 1, `alloc`/`free` with arbitrary slot tensors. Inside: INT2
    tokens are handed out as whole pages of 32 consecutive slots; a page returns to the
    free list only when all 32 of its slots have been freed.
    """

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: TriaxialKVPool,
        need_sort: bool = False,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.pool = kvcache
        self.offset = kvcache.offset
        self.num_pages2 = kvcache.num_pages2
        self.clear()

    def clear(self):
        dev = self.device
        # pages 1..num_pages2-1 ; INT4 rows 1..size4-1
        self.free_pages2 = torch.arange(1, self.num_pages2, dtype=torch.int64, device=dev)
        self.free_slots4 = torch.arange(
            self.offset + 1, self.offset + self.pool.size4, dtype=torch.int64, device=dev
        )
        self.page_free_cnt = torch.zeros((self.num_pages2,), dtype=torch.int32, device=dev)
        self.is_not_in_free_group = True
        self.free_group = []
        # kept for BaseTokenToKVPoolAllocator API compatibility
        self.free_pages = self.free_slots4
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=dev)
        self.fallback_int2_to_int4 = 0
        self.partial_slots = 0  # freed INT2 slots whose page is not yet fully released

    # ------------------------------------------------------------- accounting
    def available_size(self) -> int:
        # Slots freed inside partially-freed INT2 pages are counted so that SGLang's
        # idle-time leak check (available + evictable == max_total_num_tokens) balances;
        # they become allocatable once their page is fully released.
        return len(self.free_pages2) * PAGE + len(self.free_slots4) + self.partial_slots

    def admission_size(self) -> int:
        """Tokens the scheduler may still admit for prefill.

        INT2 overflow falls back to INT4 but not the other way round, so when INT4 is the
        binding region this is the number of *mixed* tokens it can still absorb (a request
        needs roughly ``1 - int2_fraction`` of its tokens plus its decode tokens in INT4).
        Kept separate from available_size(), which must stay exact for SGLang's leak check.
        """
        int4_share = max(1.0 - self.pool.int2_fraction, 0.05)
        return min(self.available_size(), int(len(self.free_slots4) / int4_share))

    def available_size_int4(self) -> int:
        return len(self.free_slots4)

    def available_size_int2(self) -> int:
        return len(self.free_pages2) * PAGE

    def debug_print(self) -> str:
        return (
            f"int2_free_pages={len(self.free_pages2)} int4_free={len(self.free_slots4)} "
            f"fallback2to4={self.fallback_int2_to_int4}"
        )

    def backup_state(self):
        return (
            self.free_pages2,
            self.free_slots4,
            self.page_free_cnt.clone(),
            self.partial_slots,
        )

    def restore_state(self, state):
        self.free_pages2, self.free_slots4, self.page_free_cnt, self.partial_slots = state
        self.free_pages = self.free_slots4

    # -------------------------------------------------------------- allocate
    def alloc(self, need_size: int):
        """Plain allocation (no bitwidth info): everything INT4."""
        if need_size > len(self.free_slots4):
            return None
        out = self.free_slots4[:need_size]
        self.free_slots4 = self.free_slots4[need_size:]
        self.free_pages = self.free_slots4
        return out

    def alloc_decode(self, bs: int):
        return self.alloc(bs)

    def alloc_mixed(self, bits: torch.Tensor, req_lens: List[int]) -> Optional[torch.Tensor]:
        """Allocate slots for prefill tokens in position order.

        bits: uint8 tensor [n] on device with values 2 or 4, concatenated per request in
        batch order; req_lens: number of tokens per request. INT2 tokens of a request are
        grouped 32 at a time (in position order) into whole pages; the remainder and all
        INT4 tokens receive INT4 slots. If INT2 pages run out, the excess groups fall back
        to INT4. Returns int64 [n] or None if INT4 slots are insufficient.
        """
        n = bits.numel()
        if n == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        dev = self.device
        lens = torch.tensor(req_lens, dtype=torch.int64, device=dev)
        seg = torch.repeat_interleave(torch.arange(len(req_lens), device=dev), lens)
        is2 = bits.to(torch.int64) == 2
        # rank of each INT2 token among the INT2 tokens of its request
        c = torch.cumsum(is2.to(torch.int64), dim=0)
        seg_start = torch.cumsum(lens, dim=0) - lens  # first index of each segment
        c_before_seg = torch.cat([c.new_zeros(1), c])[seg_start]  # inclusive cumsum before segment
        rank = c - c_before_seg[seg] - 1  # valid where is2
        n2_per_seg = torch.bincount(seg[is2], minlength=len(req_lens))
        full_per_seg = (n2_per_seg // PAGE) * PAGE
        keep2 = is2 & (rank < full_per_seg[seg])
        # global page index for kept INT2 tokens (they are contiguous runs of 32 per request)
        k = torch.cumsum(keep2.to(torch.int64), dim=0) - 1
        page_idx = k // PAGE
        j = k % PAGE
        num_pages = int(keep2.sum().item()) // PAGE
        avail_pages = len(self.free_pages2)
        if num_pages > avail_pages:
            demoted = keep2 & (page_idx >= avail_pages)
            self.fallback_int2_to_int4 += int(demoted.sum().item())
            keep2 = keep2 & (page_idx < avail_pages)
            num_pages = avail_pages
        n4 = n - num_pages * PAGE
        if n4 > len(self.free_slots4):
            return None
        out = torch.empty((n,), dtype=torch.int64, device=dev)
        if num_pages > 0:
            pages = self.free_pages2[:num_pages]
            self.free_pages2 = self.free_pages2[num_pages:]
            idx2 = torch.nonzero(keep2, as_tuple=True)[0]
            out[idx2] = pages[page_idx[idx2]] * PAGE + j[idx2]
        if n4 > 0:
            idx4 = torch.nonzero(~keep2, as_tuple=True)[0]
            out[idx4] = self.free_slots4[:n4]
            self.free_slots4 = self.free_slots4[n4:]
        self.free_pages = self.free_slots4
        return out

    # ------------------------------------------------------------------ free
    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        if not self.is_not_in_free_group:
            self.free_group.append(free_index)
            return
        free_index = free_index.to(torch.int64)
        is4 = free_index >= self.offset
        s4 = free_index[is4]
        if s4.numel() > 0:
            self.free_slots4 = torch.cat((self.free_slots4, s4))
            self.free_pages = self.free_slots4
        s2 = free_index[~is4]
        s2 = s2[s2 >= PAGE]  # never recycle the dummy page 0
        if s2.numel() > 0:
            pages = s2 // PAGE
            self.page_free_cnt.index_add_(
                0, pages, torch.ones_like(pages, dtype=torch.int32)
            )
            cand = torch.unique(pages)
            full = cand[self.page_free_cnt[cand] >= PAGE]
            n_full = int(full.numel())
            if n_full > 0:
                self.page_free_cnt[full] = 0
                self.free_pages2 = torch.cat((self.free_pages2, full))
            self.partial_slots += int(s2.numel()) - n_full * PAGE

    def get_cpu_copy(self, indices):
        raise NotImplementedError("TriAxialKV does not support host offload")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("TriAxialKV does not support host offload")
