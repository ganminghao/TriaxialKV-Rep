"""Offline TriAxialKV tag statistics for a replay JSONL (OpenAI chat format).

Applies the model's chat template, expands every image placeholder to its real vision
token count (Qwen-VL: 32x32 px per token after resize), runs the tagger + policy, and
reports the tag distribution, the INT2 share of prompt tokens, and the recommended
``--triaxial-int2-fraction`` for a given decode length (decode tokens are always INT4).

Usage:
  python -m sglang.test.triaxial_tag_stats --model /tmp/models/Qwen3-VL-32B-Instruct \
      --replay /tmp/datasets/osworld_trajs/replay_15step_h4.jsonl --decode-len 300 [--limit 500]
"""

import argparse
import base64
import collections
import json
import struct

import numpy as np
from transformers import AutoTokenizer

from sglang.srt.managers.triaxial_tagger import (
    MM_PAD_SHIFT_VALUE,
    TriaxialPolicy,
    TriaxialSpecialTokens,
    TriaxialTagger,
    tag_name,
)


def _png_wh(url: str):
    b64 = url.split(",", 1)[1][:64]
    raw = base64.b64decode(b64 + "=" * (-len(b64) % 4))
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", raw[16:24])
    return None


def expand_ids(tok, sp, messages, patch_px=32):
    """Token ids with each <|image_pad|> expanded to the real count of pad values."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False)
    img_tokens = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    wh = _png_wh(part["image_url"]["url"])
                    w, h = wh if wh else (1920, 1088)
                    img_tokens.append(max(1, round(w / patch_px)) * max(1, round(h / patch_px)))
    out = []
    k = 0
    for i in ids:
        if i == sp.image_pad:
            n = img_tokens[k] if k < len(img_tokens) else 1
            k += 1
            out.extend([MM_PAD_SHIFT_VALUE + 1 + k] * n)
        else:
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--decode-len", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    sp = TriaxialSpecialTokens.from_tokenizer(tok)
    tagger = TriaxialTagger(sp, TriaxialPolicy.from_arg(args.policy))

    tag_counts = collections.Counter()
    n_req = 0
    tot_tokens = 0
    tot_int2 = 0
    lens = []
    with open(args.replay) as f:
        for line in f:
            if args.limit and n_req >= args.limit:
                break
            msgs = json.loads(line)["messages"]
            ids = expand_ids(tok, sp, msgs)
            codes = tagger.tag(ids)
            bits = tagger.policy.lut[codes]
            for c, cnt in zip(*np.unique(codes, return_counts=True)):
                tag_counts[tag_name(int(c))] += int(cnt)
            n_req += 1
            tot_tokens += len(ids)
            tot_int2 += int((bits == 2).sum())
            lens.append(len(ids))

    lens = np.array(lens)
    print(f"requests: {n_req}  prompt tokens: mean {lens.mean():.0f} median {np.median(lens):.0f} p90 {np.percentile(lens, 90):.0f} max {lens.max()}")
    print(f"INT2 share of prompt tokens: {tot_int2 / tot_tokens:.4f}")
    per_req_tokens = tot_tokens / n_req + args.decode_len
    per_req_int2 = tot_int2 / n_req
    print(f"recommended --triaxial-int2-fraction (decode {args.decode_len} tok/req at INT4): {per_req_int2 / per_req_tokens:.4f}")
    print("tag distribution (tokens, share, bits):")
    lut = tagger.policy.lut
    names = {tag_name(i): i for i in range(len(lut))}
    for name, cnt in tag_counts.most_common():
        print(f"  {name:28s} {cnt:12d} {cnt / tot_tokens:8.4f}  INT{lut[names[name]]}")


if __name__ == "__main__":
    main()
