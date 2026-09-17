import json
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import BaseDataset, DatasetRow


@dataclass
class OpenAIDataset(BaseDataset):
    dataset_path: str
    num_requests: int
    fixed_output_len: Optional[int]

    @classmethod
    def from_args(cls, args: Namespace) -> "OpenAIDataset":
        return cls(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            fixed_output_len=args.sharegpt_output_len,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_openai_requests(
            dataset_path=self.dataset_path,
            num_requests=self.num_requests,
            tokenizer=tokenizer,
            fixed_output_len=self.fixed_output_len,
        )


def sample_openai_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[DatasetRow]:
    """
    Load OpenAI-compatible chat completion requests from a JSONL file.

    Each line should be a JSON object with:
    - "messages": list of {"role": str, "content": str}
    - "max_tokens": int (used as output_len if fixed_output_len not set)
    - "tools": optional list of tool definitions
    - "temperature": optional temperature value
    - "top_p": optional top_p value
    - Other OpenAI API parameters are also extracted and passed through
    """
    dataset = []
    with open(dataset_path, "r") as f:
        for line in f:
            if num_requests > 0 and len(dataset) >= num_requests:
                break
            if line.strip():
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue

    # Fields that should NOT be passed through extra_request_body
    # These are either handled separately or are metadata
    # max_tokens is excluded because it's handled via output_len -> max_completion_tokens
    # max_completion_tokens is also excluded to avoid conflicts
    EXCLUDED_FIELDS = {"messages", "max_tokens", "max_completion_tokens", "model"}

    filtered_dataset: List[DatasetRow] = []
    for data in dataset:
        messages = data.get("messages", [])
        if not messages:
            continue

        # Use max_tokens from the request, or fall back to fixed_output_len
        output_len = fixed_output_len or data.get("max_tokens", 256)

        # Extract extra request body parameters (tools, temperature, top_p, etc.)
        extra_body = {k: v for k, v in data.items() if k not in EXCLUDED_FIELDS}

        # Calculate prompt length by applying chat template
        # This includes the messages but not the tools
        templated = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if not isinstance(templated, list):  # transformers >= 5 returns a BatchEncoding
            templated = templated["input_ids"]
        prompt_len = len(templated)
        # Multimodal content: the text chat template counts one placeholder per image;
        # add the real vision token count (Qwen-VL style: 32x32 px per token after the
        # processor's resize, images in the replay files are already resized).
        vision_tokens = _count_vision_tokens(messages)
        prompt_len += vision_tokens

        # If tools are present, we need to add their token count
        # Tools are sent as part of the request and count toward input tokens
        if "tools" in extra_body:
            # Encode tools as JSON string to estimate token count
            tools_str = json.dumps(extra_body["tools"])
            tools_tokens = len(tokenizer.encode(tools_str))
            prompt_len += tools_tokens

        # Pass messages list directly - bench_serving handles List[Dict] prompts
        filtered_dataset.append(
            DatasetRow(
                prompt=messages,
                prompt_len=prompt_len,
                output_len=output_len,
                text_prompt_len=prompt_len - vision_tokens,
                vision_prompt_len=vision_tokens,
                extra_request_body=extra_body,  # Store per-request parameters
            )
        )

    print(f"Loaded {len(filtered_dataset)} OpenAI-format requests")
    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset


def _png_size_from_data_url(url: str):
    """Return (width, height) of a base64 PNG/JPEG data URL without decoding the image."""
    import base64
    import struct

    try:
        head, b64 = url.split(",", 1)
    except ValueError:
        return None
    raw = base64.b64decode(b64[:4096] + "=" * (-len(b64[:4096]) % 4))
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        w, h = struct.unpack(">II", raw[16:24])
        return w, h
    # JPEG: scan for SOF marker
    if raw[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", raw[i + 5 : i + 9])
                return w, h
            seg_len = struct.unpack(">H", raw[i + 2 : i + 4])[0]
            i += 2 + seg_len
    return None


def _count_vision_tokens(messages, patch_px: int = 32) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            wh = _png_size_from_data_url(url) if url.startswith("data:") else None
            if wh is None:
                continue
            w, h = wh
            # the chat template already counted one <|image_pad|> placeholder
            total += (max(1, round(w / patch_px)) * max(1, round(h / patch_px))) - 1
    return total
