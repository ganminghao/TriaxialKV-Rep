"""TriAxialKV attention backends.

* ``TriaxialFlashInferBackend`` — prefill (extend). The paged FlashInfer wrapper is planned
  over a per-batch bf16 scratch buffer laid out as
  ``[all prefix tokens of the batch (dequantized from the pool)] ++ [all new tokens (bf16 k/v)]``,
  so FlashInfer's FP16 prefill kernel runs unmodified on full-precision new tokens and
  on-the-fly dequantized cached prefixes (paper Sec. 3.2.3 "Prefill").
* ``TriaxialTritonBackend`` — decode. SGLang's Triton flash-decoding metadata is reused;
  the stage-1 kernel is replaced by ``decode_attention_fwd_mixed`` which unpacks and
  dequantizes INT2/INT4 KV inside the kernel (paper Sec. 3.2.3 "Decode").

They are combined by SGLang's ``HybridAttnBackend`` (prefill != decode backend).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferIndicesUpdaterPrefill,
    PrefillMetadata,
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.attention.triton_ops.triaxial_kv import (
    decode_attention_fwd_mixed,
    dequant_gather,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class TriaxialExtendMeta:
    prefix_slots: torch.Tensor  # int64 [P] pool slots of all cached prefix tokens (batch order)
    num_prefix: int  # P
    num_new: int  # N (= extend_num_tokens)


class _TriaxialIndicesUpdaterPrefill(FlashInferIndicesUpdaterPrefill):
    """Plans the paged wrapper with indices into the scratch buffer instead of the pool."""

    def __init__(self, model_runner, attn_backend):
        super().__init__(model_runner, attn_backend)
        self.extend_meta: Optional[TriaxialExtendMeta] = None

    def call_begin_forward(
        self,
        wrapper_ragged,
        wrapper_paged,
        req_pool_indices,
        paged_kernel_lens,
        paged_kernel_lens_sum,
        seq_lens,
        prefix_lens,
        kv_start_idx,
        kv_indptr,
        qo_indptr,
        use_ragged,
        spec_info,
        use_sliding_window_kv_pool=False,
        fixed_split_size=None,
        multi_item_params=None,
    ):
        assert spec_info is None, "TriAxialKV does not support speculative decoding"
        assert not use_ragged and not use_sliding_window_kv_pool
        assert multi_item_params is None or not multi_item_params.is_enabled()
        bs = len(seq_lens)
        fb = self.attn_backend.current_forward_batch
        seq_lens_np = fb.seq_lens_cpu.numpy().astype(np.int64)
        prefix_np = np.asarray(fb.extend_prefix_lens_cpu, dtype=np.int64)
        assert len(prefix_np) == bs
        total = int(seq_lens_np.sum())
        P = int(prefix_np.sum())
        N = total - P

        # position-order slot list of the whole sequences (prefix ++ new) from req_to_token
        kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: bs + 1]
        kv_slots = torch.empty(total + 256, dtype=torch.int32, device=req_pool_indices.device)
        create_flashinfer_kv_indices_triton[(bs,)](
            self.req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_slots,
            self.req_to_token.shape[1],
        )

        # scratch row for each (request, position): prefix rows first, then new rows
        pos_in_req = np.concatenate([np.arange(L) for L in seq_lens_np]) if total else np.zeros(0, np.int64)
        is_prefix = pos_in_req < np.repeat(prefix_np, seq_lens_np)
        pidx = np.cumsum(is_prefix) - 1
        nidx = np.cumsum(~is_prefix) - 1
        scratch_rows = np.where(is_prefix, pidx, P + nidx).astype(np.int32)
        prefix_pos = np.nonzero(is_prefix)[0]

        dev = req_pool_indices.device
        scratch_rows_t = torch.from_numpy(scratch_rows).to(dev, non_blocking=True)
        if P > 0:
            prefix_pos_t = torch.from_numpy(prefix_pos).to(dev, non_blocking=True)
            prefix_slots = kv_slots[prefix_pos_t].to(torch.int64)
        else:
            prefix_slots = torch.empty((0,), dtype=torch.int64, device=dev)
        self.extend_meta = TriaxialExtendMeta(prefix_slots, P, N)

        kv_indices = torch.empty(total + 256, dtype=torch.int32, device=dev)
        kv_indices[:total] = scratch_rows_t

        qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
        qo_indptr = qo_indptr[: bs + 1]

        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.q_data_type,  # scratch is bf16/fp16
            custom_mask=None,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
        )


class TriaxialFlashInferBackend(FlashInferAttnBackend):
    def __init__(self, model_runner: "ModelRunner", init_new_workspace: bool = False):
        self.current_forward_batch: Optional[ForwardBatch] = None
        super().__init__(model_runner, init_new_workspace=init_new_workspace)
        assert self.num_wrappers == 1, "TriAxialKV: sliding window / cross attention unsupported"
        self.indices_updater_prefill = _TriaxialIndicesUpdaterPrefill(model_runner, self)
        self.pool = model_runner.token_to_kv_pool
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.q_dtype = model_runner.dtype
        self._scratch_k: Optional[torch.Tensor] = None
        self._scratch_v: Optional[torch.Tensor] = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        assert forward_batch.forward_mode.is_extend(), (
            "TriaxialFlashInferBackend only serves extend batches"
        )
        self.current_forward_batch = forward_batch
        self.indices_updater_prefill.update(
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.seq_lens_cpu,
            forward_batch.seq_lens_sum,
            forward_batch.extend_prefix_lens,
            prefill_wrappers=self.prefill_wrappers_paged,
            use_ragged=False,
            encoder_lens=forward_batch.encoder_lens,
            spec_info=None,
            fixed_split_size=self.prefill_split_tile_size,
            multi_item_params=None,
        )
        self.forward_metadata = PrefillMetadata(self.prefill_wrappers_paged, False, False)
        meta = self.indices_updater_prefill.extend_meta
        total = meta.num_prefix + meta.num_new
        self._scratch_k = torch.empty(
            (total, self.num_kv_heads, self.head_dim), dtype=self.q_dtype, device=forward_batch.seq_lens.device
        )
        self._scratch_v = torch.empty_like(self._scratch_k)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        assert k is not None and v is not None
        meta = self.indices_updater_prefill.extend_meta
        cache_loc = forward_batch.out_cache_loc
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if save_kv_cache:
            self.pool.set_kv_buffer(layer, cache_loc, k, v)

        P, N = meta.num_prefix, meta.num_new
        sk, sv = self._scratch_k, self._scratch_v
        if P > 0:
            dequant_gather(meta.prefix_slots, self.pool.layer_kv(layer.layer_id), sk[:P], sv[:P])
        sk[P : P + N].copy_(k)
        sv[P : P + N].copy_(v)

        wrapper = self.forward_metadata.prefill_wrappers[0]
        o = wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            (sk, sv),
            causal=True,
            sm_scale=layer.scaling,
            window_left=-1,
            logits_soft_cap=layer.logit_cap,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(self, *args, **kwargs):
        raise RuntimeError("TriaxialFlashInferBackend does not serve decode")


class TriaxialTritonBackend(TritonAttnBackend):
    def __init__(self, model_runner: "ModelRunner", skip_prefill: bool = True, kv_indptr_buf=None):
        super().__init__(model_runner, skip_prefill=skip_prefill, kv_indptr_buf=kv_indptr_buf)
        self.pool = model_runner.token_to_kv_pool
        self.decode_attention_fwd_mixed = torch.compiler.disable(decode_attention_fwd_mixed)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        assert sinks is None and layer.sliding_window_size in (None, -1)
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        o = torch.empty_like(q)
        if save_kv_cache:
            # decode tokens always live in the INT4 region
            self.pool.set_kv_buffer_int4(layer.layer_id, forward_batch.out_cache_loc, k, v)
        self.decode_attention_fwd_mixed(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.pool.layer_kv(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            self.forward_metadata.attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            logit_cap=layer.logit_cap if layer.logit_cap else 0.0,
        )
        return o

    def forward_extend(self, *args, **kwargs):
        raise RuntimeError("TriaxialTritonBackend does not serve extend")
