"""TriAxialKV tagger: chat-template driven per-token (temporal, modal, semantic) tags
and the tag -> bitwidth policy.

Runs on the scheduler process on `origin_input_ids` after multimodal padding, so image
positions are the pad values (>= MM_PAD_SHIFT_VALUE) or the raw `<|image_pad|>` id.
No model inference; a single pass over the special-token positions.

Tag axes (TriAxialKV paper, Sec. 3.1):
  temporal: older, m2, m1, current   (distance in user turns from the last user turn)
  modal:    text, image
  semantic: inst, user, assistant, reasoning, tool_call, obs, delim
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

TEMPORAL = ["older", "m2", "m1", "current"]
MODAL = ["text", "image"]
SEMANTIC = ["inst", "user", "assistant", "reasoning", "tool_call", "obs", "delim"]
T_OLDER, T_M2, T_M1, T_CUR = range(4)
M_TEXT, M_IMAGE = range(2)
S_INST, S_USER, S_ASSISTANT, S_REASONING, S_TOOL_CALL, S_OBS, S_DELIM = range(7)

MM_PAD_SHIFT_VALUE = 1_000_000  # keep in sync with schedule_batch.MM_PAD_SHIFT_VALUE


def tag_id(t: int, m: int, s: int) -> int:
    return (t * 2 + m) * 7 + s


def tag_name(code: int) -> str:
    s = code % 7
    m = (code // 7) % 2
    t = code // 14
    return f"{TEMPORAL[t]}|{MODAL[m]}|{SEMANTIC[s]}"


@dataclass
class TriaxialSpecialTokens:
    im_start: int
    im_end: int
    vision_start: int
    vision_end: int
    image_pad: int
    video_pad: int
    think_start: int
    think_end: int
    tool_call_start: int
    tool_call_end: int
    tool_resp_start: int
    tool_resp_end: int
    newline: int
    role_system: int
    role_user: int
    role_assistant: int
    role_tool: int

    @classmethod
    def from_tokenizer(cls, tok) -> "TriaxialSpecialTokens":
        def tid(s):
            i = tok.convert_tokens_to_ids(s)
            return -1 if i is None else int(i)

        def word(s):
            ids = tok.encode(s, add_special_tokens=False)
            return int(ids[0]) if len(ids) == 1 else -1

        return cls(
            im_start=tid("<|im_start|>"),
            im_end=tid("<|im_end|>"),
            vision_start=tid("<|vision_start|>"),
            vision_end=tid("<|vision_end|>"),
            image_pad=tid("<|image_pad|>"),
            video_pad=tid("<|video_pad|>"),
            think_start=tid("<think>"),
            think_end=tid("</think>"),
            tool_call_start=tid("<tool_call>"),
            tool_call_end=tid("</tool_call>"),
            tool_resp_start=tid("<tool_response>"),
            tool_resp_end=tid("</tool_response>"),
            newline=word("\n"),
            role_system=word("system"),
            role_user=word("user"),
            role_assistant=word("assistant"),
            role_tool=word("tool"),
        )


class TriaxialPolicy:
    """tag -> bits. JSON format: {"default": 4, "rules": [{"temporal": [...], "modal": [...],
    "semantic": [...], "bits": 2}, ...]}; the first matching rule wins, missing axes match
    everything."""

    def __init__(self, default_bits: int, rules: List[dict]):
        lut = np.full((4 * 2 * 7,), default_bits, dtype=np.uint8)
        # apply rules in reverse so that the first rule has the final say
        for rule in reversed(rules):
            ts = [TEMPORAL.index(x) for x in rule.get("temporal", TEMPORAL)]
            ms = [MODAL.index(x) for x in rule.get("modal", MODAL)]
            ss = [SEMANTIC.index(x) for x in rule.get("semantic", SEMANTIC)]
            b = int(rule["bits"])
            assert b in (2, 4)
            for t in ts:
                for m in ms:
                    for s in ss:
                        lut[tag_id(t, m, s)] = b
        self.lut = lut

    @classmethod
    def default(cls) -> "TriaxialPolicy":
        # INT4: instructions/tool schemas, template scaffolding, tool calls, and the
        # current turn's text. INT2: every image (screenshot observations) and all older
        # text. Images are INT2 even in the current turn because with prefix caching a
        # token keeps the precision it was first stored with: a screenshot enters the
        # cache as "current" and is then reused by the following steps as history, so
        # tagging current images INT4 would keep the whole history at INT4.
        return cls(
            default_bits=2,
            rules=[
                {"semantic": ["inst", "delim", "tool_call"], "bits": 4},
                {"temporal": ["current"], "modal": ["text"], "bits": 4},
            ],
        )

    @classmethod
    def current_int4(cls) -> "TriaxialPolicy":
        """Variant with the whole current turn (images included) at INT4."""
        return cls(
            default_bits=2,
            rules=[
                {"semantic": ["inst", "delim", "tool_call"], "bits": 4},
                {"temporal": ["current"], "bits": 4},
            ],
        )

    @classmethod
    def from_arg(cls, arg: Optional[str]) -> "TriaxialPolicy":
        if arg is None or arg == "default":
            return cls.default()
        if arg == "current_int4":
            return cls.current_int4()
        if arg == "all4":
            return cls(4, [])
        if arg == "all2":
            return cls(2, [])
        with open(arg) as f:
            cfg = json.load(f)
        return cls(int(cfg.get("default", 4)), cfg.get("rules", []))

    def describe(self) -> str:
        n2 = int((self.lut == 2).sum())
        return f"TriaxialPolicy: {n2}/{len(self.lut)} tag combinations at INT2"


class TriaxialTagger:
    def __init__(self, special: TriaxialSpecialTokens, policy: TriaxialPolicy):
        self.sp = special
        self.policy = policy
        self.stats_tokens = np.zeros((4 * 2 * 7,), dtype=np.int64)

    # ------------------------------------------------------------------ core
    def tag(self, ids: Sequence[int]) -> np.ndarray:
        """Return per-token tag codes (uint8) for the prompt token ids."""
        ids = np.asarray(ids, dtype=np.int64)
        n = len(ids)
        sp = self.sp
        temporal = np.full(n, T_OLDER, dtype=np.int8)
        modal = np.full(n, M_TEXT, dtype=np.int8)
        semantic = np.full(n, S_USER, dtype=np.int8)

        # image tokens: pad values or raw image/video pad ids
        img = (ids >= MM_PAD_SHIFT_VALUE) | (ids == sp.image_pad) | (ids == sp.video_pad)
        modal[img] = M_IMAGE

        specials = np.isin(
            ids,
            [
                sp.im_start, sp.im_end, sp.vision_start, sp.vision_end,
                sp.think_start, sp.think_end, sp.tool_call_start, sp.tool_call_end,
                sp.tool_resp_start, sp.tool_resp_end,
            ],
        )
        pos = np.nonzero(specials)[0]

        # ---- pass 1: message segmentation by <|im_start|> ... <|im_end|>
        # message = (start, end_exclusive, role)
        msgs = []
        i = 0
        starts = pos[ids[pos] == sp.im_start]
        for st in starts:
            role = S_USER
            role_tok = ids[st + 1] if st + 1 < n else -1
            if role_tok == sp.role_system:
                role = S_INST
            elif role_tok == sp.role_assistant:
                role = S_ASSISTANT
            elif role_tok == sp.role_tool:
                role = S_OBS
            else:
                role = S_USER
            # end = next <|im_end|> after st (or n)
            ends = pos[(pos > st) & (ids[pos] == sp.im_end)]
            en = int(ends[0]) + 1 if len(ends) else n
            msgs.append((int(st), en, role))

        # ---- pass 2: semantic + delim per message
        for st, en, role in msgs:
            semantic[st:en] = role
            # scaffolding: <|im_start|>, role word, newline after it; <|im_end|> and the newline after it
            semantic[st] = S_DELIM
            if st + 1 < n:
                semantic[st + 1] = S_DELIM
            if st + 2 < n and ids[st + 2] == sp.newline:
                semantic[st + 2] = S_DELIM
            if en - 1 < n and ids[en - 1] == sp.im_end:
                semantic[en - 1] = S_DELIM
                if en < n and ids[en] == sp.newline:
                    semantic[en] = S_DELIM
            if role == S_USER:
                # screenshots / tool outputs are environment observations
                semantic[st:en][img[st:en]] = S_OBS
                self._mark_span(ids, semantic, st, en, sp.tool_resp_start, sp.tool_resp_end, S_OBS)
            elif role == S_ASSISTANT:
                self._mark_span(ids, semantic, st, en, sp.think_start, sp.think_end, S_REASONING)
                self._mark_span(ids, semantic, st, en, sp.tool_call_start, sp.tool_call_end, S_TOOL_CALL)
        # vision markers are scaffolding
        semantic[(ids == sp.vision_start) | (ids == sp.vision_end)] = S_DELIM

        # ---- pass 3: temporal by user turns (a turn = from a user message to the next user message)
        user_starts = [st for st, _, role in msgs if role == S_USER]
        n_turns = len(user_starts)
        if n_turns == 0:
            temporal[:] = T_CUR
        else:
            bounds = user_starts + [n]
            for k in range(n_turns):
                dist = n_turns - 1 - k
                t = T_CUR if dist == 0 else (T_M1 if dist == 1 else (T_M2 if dist == 2 else T_OLDER))
                temporal[bounds[k] : bounds[k + 1]] = t
            # everything before the first user turn (system prompt) is "older"
            temporal[: bounds[0]] = T_OLDER

        codes = ((temporal.astype(np.int16) * 2 + modal) * 7 + semantic).astype(np.uint8)
        return codes

    @staticmethod
    def _mark_span(ids, semantic, st, en, open_id, close_id, label):
        seg = ids[st:en]
        opens = np.nonzero(seg == open_id)[0]
        if len(opens) == 0:
            return
        closes = np.nonzero(seg == close_id)[0]
        for o in opens:
            c = closes[closes > o]
            c_end = int(c[0]) if len(c) else (en - st)
            semantic[st + o : st + c_end + 1] = label
            # the bracket tokens themselves are scaffolding
            semantic[st + o] = S_DELIM
            if len(c):
                semantic[st + c_end] = S_DELIM

    def bits(self, ids: Sequence[int], collect_stats: bool = False) -> np.ndarray:
        codes = self.tag(ids)
        if collect_stats:
            np.add.at(self.stats_tokens, codes, 1)
        return self.policy.lut[codes]

    def summary(self, ids: Sequence[int]) -> Dict[str, int]:
        codes = self.tag(ids)
        out: Dict[str, int] = {}
        for c, cnt in zip(*np.unique(codes, return_counts=True)):
            out[tag_name(int(c))] = int(cnt)
        return out
