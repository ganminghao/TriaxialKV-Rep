# TriAxialKV on SGLang v0.5.10 — implementation spec

Reproduction of TriAxialKV (arXiv 2605.17170): mixed INT2/INT4 paged KV cache with
per-token bitwidth from a chat-template tagger. Target model: Qwen3-VL-32B (64 layers,
H = 8 KV heads, D = 128) on one H100. Also runs on Qwen3-8B (36 layers, H=8, D=128) for tests.

Everything below is the single source of truth for layouts and kernel signatures.
`G = 32` is the quantization group size. `H` = num KV heads, `D` = head_dim (must be a
multiple of 32). All "meta" tensors are fp16 pairs `[scale, min]`; dequant is
`x = q * scale + min`, quant is `q = clamp(round((x - min) / scale), 0, 2^bits - 1)` with
`scale = (max - min) / (2^bits - 1)`, and `scale := 1.0` when `max == min` (q = 0).

## 1. Token address space

- One int slot index space shared by both pools. `offset = size2`.
  - INT2 slots: `s in [0, size2)`; page `p = s // 32`, in-page index `j = s % 32`.
    `size2` is a multiple of 32; `num_pages2 = size2 // 32`. Page 0 is the reserved
    dummy page (never allocated), matching SGLang's reserved slot 0.
  - INT4 slots: `s in [size2, size2 + size4)`; row `r = s - size2`. Row 0 is reserved.
- SGLang runs with `page_size = 1` (radix cache and scheduler think in tokens). INT2 page
  grouping is internal to the allocator (`TriaxialAllocator`).
- Decode-generated tokens always go to INT4. Prefill tokens are routed by the per-token
  bitwidth array; INT2 tokens of one request in one prefill chunk are grouped 32 at a
  time (position order) into whole pages; the remainder (< 32) is demoted to INT4.
  Therefore a page is always written in full by a single `set_kv_buffer` call.

## 2. Buffer layouts (per layer; one tensor per kind with a leading layer dim)

```
k2_data: uint8 [L, num_pages2, H, D, 8]     # per-channel INT2 keys. byte b holds tokens j=4b..4b+3,
                                             # token j sits at bits 2*(j%4) (LSB first)
k2_meta: fp16  [L, num_pages2, H, D, 2]     # [scale, min] per (page, head, channel), over the page's 32 tokens
v2_data: uint8 [L, size2, H, D//4]          # per-token INT2 values. byte b holds channels d=4b..4b+3,
                                             # channel d at bits 2*(d%4)
v2_meta: fp16  [L, size2, H, D//32, 2]      # [scale, min] per (slot, head, 32-channel group)
k4_data: uint8 [L, size4, H, D//2]          # per-token INT4 keys. byte b holds channels d=2b (low nibble), 2b+1 (high nibble)
k4_meta: fp16  [L, size4, H, D//32, 2]
v4_data, v4_meta: same as k4_*
```
Bytes per token-head: INT2 = 48 (K) + 48 (V); INT4 = 80 + 80. bf16 = 256 + 256.

A per-layer view is passed to kernels as a `TriaxialLayerKV` dataclass:
```python
@dataclass
class TriaxialLayerKV:
    offset: int            # = size2
    k2_data: torch.Tensor  # [num_pages2, H, D, 8] uint8
    k2_meta: torch.Tensor  # [num_pages2, H, D, 2] fp16
    v2_data: torch.Tensor  # [size2, H, D//4] uint8
    v2_meta: torch.Tensor  # [size2, H, D//32, 2] fp16
    k4_data: torch.Tensor  # [size4, H, D//2] uint8
    k4_meta: torch.Tensor  # [size4, H, D//32, 2] fp16
    v4_data: torch.Tensor
    v4_meta: torch.Tensor
```
Defined in `python/sglang/srt/layers/attention/triton_ops/triaxial_kv.py` (kernel module)
so both the pool and the backends import it from one place.

## 3. Kernel module: `python/sglang/srt/layers/attention/triton_ops/triaxial_kv.py`

All functions take contiguous CUDA tensors. `x` inputs are bf16 or fp16 `[n, H, D]`.

```python
def quant_int4_tokens(x, rows, data, meta):
    # x [n,H,D]; rows int64 [n] (row index into data/meta, i.e. slot - offset);
    # writes data[rows] (uint8 [.,H,D//2]) and meta[rows] (fp16 [.,H,D//32,2]).

def quant_int2_v_tokens(x, rows, data, meta):
    # same as above for INT2 per-token values: data [.,H,D//4], meta [.,H,D//32,2]

def quant_int2_k_pages(x, tok_rows, pages, data, meta):
    # x [n,H,D] (all new K of this call); tok_rows int32 [num_pages, 32] = row indices into x
    # for the 32 tokens of each page in j order; pages int64 [num_pages] = page ids.
    # Per (page, head, channel): min/max over the 32 tokens -> scale/min -> pack 8 bytes.
    # writes data[pages] (uint8 [.,H,D,8]) and meta[pages] (fp16 [.,H,D,2]).

def dequant_gather(slots, kv: TriaxialLayerKV, out_k, out_v):
    # slots int64 [m] (any mix of INT2/INT4 slots); out_k/out_v bf16 [m,H,D] (written in place).
    # INT2 K: page = s//32, j = s%32, byte = k2_data[page,h,d,j//4] >> (2*(j%4)) & 3.

def decode_attention_fwd_mixed(q, kv: TriaxialLayerKV, o, kv_indptr, kv_indices,
                               attn_logits, attn_lse, num_kv_splits, max_kv_splits,
                               sm_scale, logit_cap=0.0):
    # Drop-in replacement for sglang.srt.layers.attention.triton_ops.decode_attention.
    # decode_attention_fwd for the GQA case (kv_group_num >= 1, Lk == Lv == D, no rope
    # split, no sinks). Stage 1 is a fork of _fwd_grouped_kernel_stage1 whose K/V loads
    # dequantize on the fly: per BLOCK_N block of kv_indices decide INT2 / INT4 / mixed
    # by comparing slots with kv.offset (runtime scalar `if` on tl.min/tl.max, or masked
    # loads for the mixed case). Stage 2 reuses SGLang's _decode_softmax_reducev_fwd
    # unchanged (v_scale = 1.0). q: [bs, num_q_heads, D] bf16; o: same shape.
```

Torch reference implementations (for tests) live in the same module under
`ref_quant_dequant_int4`, `ref_quant_dequant_int2_v`, `ref_quant_dequant_int2_k_page`,
`ref_dequant_gather`, and tests in `python/sglang/test/test_triaxial_kv.py`
(run with `python -m pytest`). Acceptance for the decode kernel: output matches a
torch attention over the *dequantized* K/V within bf16 tolerance (atol 2e-2, rtol 2e-2)
for random pools with random INT2/INT4 slot mixes, bs 1..8, seq_len 1..4096,
kv_group_num in {1, 4, 8}, and slot lists that include page-unaligned INT2 runs.

## 4. Pool, allocator, tagger, backends (integration side)

- `python/sglang/srt/mem_cache/triaxial_pool.py`
  - `TriaxialKVPool(KVCache)`: allocates the eight tensors; `set_kv_buffer(layer, loc, k, v, ...)`
    splits `loc` by `offset`, INT4 rows via `quant_int4_tokens`, INT2 V via
    `quant_int2_v_tokens`, INT2 K via `quant_int2_k_pages` after grouping the INT2 slots of
    `loc` into pages (`argsort` of INT2 slots, view `[-1, 32]`; cached per `loc` tensor so
    the grouping is computed once per forward batch, not per layer).
    `layer_kv(layer_id) -> TriaxialLayerKV`. `get_key_buffer`/`get_value_buffer` raise
    (nothing may read the pool as dense tensors).
  - `TriaxialAllocator(BaseTokenToKVPoolAllocator)`: page_size = 1 to the outside.
    `alloc_mixed(bits_flat: uint8 tensor [n] on device, req_lens: list[int]) -> int64 [n]`
    returns slots in position order; `alloc_decode(bs)`; `alloc(n)` (all INT4, used by
    SGLang paths that do not carry bits); `free(slots)`: INT4 rows back to the free list,
    INT2 slots increment a per-page freed counter and a page is recycled when the
    counter reaches 32. `available_size()` = total free tokens (INT2 pages*32 + INT4);
    `available_size_int4()` for decode admission. INT2 → INT4 fallback when INT2 pages run out.
- Tagger: `python/sglang/srt/managers/triaxial_tagger.py`. Runs on the scheduler when a
  request is created (after multimodal padding). Input: `origin_input_ids` (list[int]),
  special-token ids resolved from the tokenizer. Produces `bits: np.ndarray uint8 [n]`
  with values 2 or 4. Tags: temporal {older, m2, m1, current} by `<|im_start|>user`
  boundaries; semantic {inst, user, assistant, reasoning, tool_call, obs, delim}; modality
  {text, image} (image = pad values >= MM_PAD_SHIFT_VALUE or `<|image_pad|>`).
  Policy = dict tag -> bits, loaded from `--triaxial-policy` (JSON path) or the default:
  INT4 for inst, delim, tool_call, everything in the current turn; INT2 otherwise.
- Server args: `--triaxial-kv` (bool), `--triaxial-int2-fraction` (float, default 0.75,
  share of *slots* given to INT2), `--triaxial-policy` (path or "default").
  When enabled: `page_size` must be 1, attention backend forced to
  prefill=flashinfer(triaxial) / decode=triton(triaxial), no speculative decoding,
  no CUDA-graph restrictions beyond SGLang's own (kernel is graph-safe).
- Backends: `python/sglang/srt/layers/attention/triaxial_backend.py`
  - `TriaxialFlashInferBackend(FlashInferAttnBackend)`: extend path only. Plans the paged
    wrapper with `kv_indices = arange(sum(seq_lens))` and, per layer, builds a bf16
    scratch `[sum(seq_lens), H, D]` = dequantized prefix rows (via `dequant_gather`) plus
    the new tokens' bf16 k/v copied in, then calls the paged wrapper on the scratch.
  - `TriaxialTritonBackend(TritonAttnBackend)`: decode path only; `forward_decode` calls
    `decode_attention_fwd_mixed` with `pool.layer_kv(layer_id)`.
  - Combined through the existing `HybridAttnBackend`.
