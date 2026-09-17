#!/usr/bin/env python3
"""Build a chat-request replay file (OpenAI JSONL) from an OSWorld-Verified
trajectory archive, reproducing the requests that OSWorld's Qwen3-VL agent
(``mm_agents/qwen3vl_agent.py``) sends at every recorded step.

Output: one JSON object per line, consumable by SGLang's
``bench_serving --dataset-name openai``::

    {"messages": [...], "max_tokens": N, "ignore_eos": true, "temperature": 0}

Images are OpenAI ``image_url`` parts with ``data:image/png;base64,...`` URLs,
resized exactly like the agent's ``process_image`` (smart_resize, factor 32).

See README.md next to this file for the alignment rule and the screenshot
index semantics.
"""

from __future__ import annotations

import argparse
import ast
import base64
import collections
import io
import json
import math
import multiprocessing as mp
import os
import random
import statistics
import sys
import time
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Constants copied from the OSWorld agent (mm_agents/qwen3vl_agent.py)
# ---------------------------------------------------------------------------

DEFAULT_TOKENIZER = "/tmp/models/Qwen3-VL-32B-Instruct"
IMAGE_FACTOR = 32  # Qwen3-VL: patch 16 x spatial merge 2 -> 32 px per token side
MAX_PIXELS = 16 * 16 * 4 * 12800  # process_image() cap in qwen3vl_agent.py

ACTION_DESCRIPTION_PROMPT = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
        """


def description_prompt(coordinate_type: str, processed_width: int, processed_height: int) -> str:
    lines = [
        "Use a mouse and keyboard to interact with a computer, and take screenshots.",
        "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
        "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
        (
            f"* The screen's resolution is {processed_width}x{processed_height}."
            if coordinate_type == "absolute"
            else "* The screen's resolution is 1000x1000."
        ),
        "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
        "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
    ]
    return "\n".join(lines)


def build_system_prompt(coordinate_type: str = "relative", processed_width: int = 0, processed_height: int = 0) -> str:
    tools_def = {
        "type": "function",
        "function": {
            "name_for_human": "computer_use",
            "name": "computer_use",
            "description": description_prompt(coordinate_type, processed_width, processed_height),
            "parameters": {
                "properties": {
                    "action": {
                        "description": ACTION_DESCRIPTION_PROMPT,
                        "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag",
                                 "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"],
                        "type": "string",
                    },
                    "keys": {"description": "Required only by `action=key`.", "type": "array"},
                    "text": {"description": "Required only by `action=type`.", "type": "string"},
                    "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                    "pixels": {"description": "The amount of scrolling.", "type": "number"},
                    "time": {"description": "The seconds to wait.", "type": "number"},
                    "status": {
                        "description": "The status of the task.",
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
                "required": ["action"],
                "type": "object",
            },
            "args_format": "Format the arguments as a JSON object.",
        },
    }
    return """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""


def build_instruction_prompt(instruction: str, previous_actions_str: str) -> str:
    return f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
{previous_actions_str}"""


# ---------------------------------------------------------------------------
# smart_resize copied from OSWorld mm_agents/utils/qwen_vl_utils.py
# ---------------------------------------------------------------------------

def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280, max_long_side=8192):
    if height < 2 or width < 2:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def process_image(image_bytes: bytes) -> Tuple[str, int, int]:
    """Agent's process_image(): smart_resize(factor=32, max_pixels=MAX_PIXELS),
    PIL resize, PNG encode, base64. Returns (b64, width, height)."""
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    resized_height, resized_width = smart_resize(
        height=height, width=width, factor=IMAGE_FACTOR, max_pixels=MAX_PIXELS)
    image = image.resize((resized_width, resized_height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), resized_width, resized_height


def image_tokens(width: int, height: int) -> int:
    """Qwen3-VL token count of an already-resized image: one token per 32x32
    block plus <|vision_start|> and <|vision_end|>."""
    return (round_by_factor(width, IMAGE_FACTOR) // IMAGE_FACTOR) * \
           (round_by_factor(height, IMAGE_FACTOR) // IMAGE_FACTOR) + 2


# ---------------------------------------------------------------------------
# Archive parsing
# ---------------------------------------------------------------------------

@dataclass
class Step:
    step_num: int                 # predict-call index in the OSWorld runner (1-based)
    actions: List[str]            # pyautogui code strings executed for this call
    screenshot_files: List[str]   # one png per executed action, traj order
    response: str                 # model output text (Qwen25VL Output block)
    low_level: str                # low-level instruction (see low_level_instruction())


@dataclass
class Task:
    domain: str
    task_id: str
    prefix: str                   # zip member prefix ".../<domain>/<task_id>"
    instruction: str = ""
    steps: List[Step] = field(default_factory=list)
    n_blocks: int = 0
    n_step_groups: int = 0        # recorded predict calls in traj.jsonl
    n_traj_lines: int = 0
    error: Optional[str] = None   # {"Error": ...} line in traj.jsonl
    align_note: Optional[str] = None


def list_tasks(zf: zipfile.ZipFile) -> List[Task]:
    tasks = []
    for name in zf.namelist():
        if not name.endswith("/traj.jsonl"):
            continue
        prefix = name[: -len("/traj.jsonl")]
        parts = prefix.split("/")
        tasks.append(Task(domain=parts[-2], task_id=parts[-1], prefix=prefix))
    tasks.sort(key=lambda t: (t.domain, t.task_id))
    return tasks


def parse_runtime_log(text: str) -> List[dict]:
    """Split runtime.log into one block per model call.

    A block starts at a line 'Qwen25VL Output: ...' and its response text runs
    until the 'Low level instruction: ...' line that is directly followed by a
    'Pyautogui code: [...]' line. Other lines ('Generating content ...',
    'Error calling Qwen model ...') are ignored.
    """
    lines = text.split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        if not ln.startswith("Qwen25VL Output: "):
            i += 1
            continue
        resp = [ln[len("Qwen25VL Output: "):]]
        i += 1
        low, code = None, None
        while i < n:
            if lines[i].startswith("Low level instruction: ") and i + 1 < n \
                    and lines[i + 1].startswith("Pyautogui code: "):
                low = lines[i][len("Low level instruction: "):]
                raw = lines[i + 1][len("Pyautogui code: "):]
                try:
                    code = ast.literal_eval(raw)
                except Exception:
                    code = None
                i += 2
                break
            resp.append(lines[i])
            i += 1
        blocks.append({"response": "\n".join(resp), "low": low, "code": code})
    return blocks


def read_task(zf: zipfile.ZipFile, task: Task) -> None:
    """Fill task.steps from traj.jsonl + runtime.log using the index rule
    (block[step_num-1] is the model call that produced step step_num) and
    verify each step's executed actions equal that block's parsed code list.
    On the first verification failure the task is truncated to the aligned
    prefix and task.align_note records why."""
    groups: "collections.OrderedDict[int, List[dict]]" = collections.OrderedDict()
    for line in zf.read(task.prefix + "/traj.jsonl").decode("utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        task.n_traj_lines += 1
        if "Error" in obj:
            task.error = str(obj["Error"])
            continue
        groups.setdefault(obj["step_num"], []).append(obj)
    task.n_step_groups = len(groups)
    if not groups:
        return
    try:
        log = zf.read(task.prefix + "/runtime.log").decode("utf-8", errors="replace")
    except KeyError:
        task.align_note = "runtime.log missing"
        return
    blocks = parse_runtime_log(log)
    task.n_blocks = len(blocks)
    for step_num, recs in groups.items():
        actions = [r["action"] for r in recs]
        if step_num - 1 >= len(blocks):
            task.align_note = f"step {step_num}: no log block (only {len(blocks)} blocks)"
            break
        b = blocks[step_num - 1]
        if b["code"] != actions:
            task.align_note = f"step {step_num}: traj actions {actions!r} != log code {b['code']!r}"
            break
        task.steps.append(Step(
            step_num=step_num,
            actions=actions,
            screenshot_files=[r["screenshot_file"] for r in recs],
            response=b["response"],
            low_level=low_level_instruction(b["response"], b["low"] or ""),
        ))


# ---------------------------------------------------------------------------
# Request construction (mirrors Qwen3VLAgent.predict)
# ---------------------------------------------------------------------------

def low_level_instruction(response: str, log_low: str) -> str:
    """The low-level instruction as Qwen3VLAgent.parse_response derives it:
    the first line starting with 'Action:' (case-insensitive), prefix removed.
    The archive's 'Low level instruction' log line was written by the
    Qwen2.5-VL agent, which keeps the 'Action: ' prefix; its fallback text
    ('Performing <x> action') is identical in both agents, so the log line is
    used when no 'Action:' line exists."""
    for line in response.split("\n"):
        line = line.strip()
        if line.lower().startswith("action:"):
            low = line.split("Action:")[-1].strip()
            if low:
                return low
    return log_low


def image_part(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def build_messages(system_prompt: str, instruction: str, history_n: int,
                   prev_low_levels: List[str], prev_responses: List[str],
                   prev_screenshots: List[str], current_screenshot: str) -> List[dict]:
    """Exactly the message list Qwen3VLAgent.predict() builds at a step with
    len(prev_*) earlier steps. Screenshots are base64 strings."""
    current_step = len(prev_low_levels)
    history_start_idx = max(0, current_step - history_n)
    previous_actions = [f"Step {i + 1}: {prev_low_levels[i]}" for i in range(history_start_idx)]
    previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"
    instruction_prompt = build_instruction_prompt(instruction, previous_actions_str)

    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    history_len = min(history_n, len(prev_responses))
    if history_len > 0:
        history_responses = prev_responses[-history_len:]
        history_screenshots = prev_screenshots[-history_len:]
        for idx in range(history_len):
            content = [image_part(history_screenshots[idx])]
            if idx == 0:
                content.append({"type": "text", "text": instruction_prompt})
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant",
                             "content": [{"type": "text", "text": f"{history_responses[idx]}"}]})
        messages.append({"role": "user", "content": [image_part(current_screenshot)]})
    else:
        messages.append({"role": "user", "content": [image_part(current_screenshot),
                                                     {"type": "text", "text": instruction_prompt}]})
    return messages


def observation_file(task: Task, j: int, mapping: str) -> str:
    """Screenshot the agent saw at recorded step index j (0-based).

    previous-step: the runner saves the screenshot taken *after* each executed
      action as step_<k>.png, so the observation before step j is the last png
      of step j-1. The initial observation (before the first action) is not in
      the archive; step_<first>.png stands in for it.
    same-step: the png of step j itself (one step later than what the agent saw).
    """
    if mapping == "previous-step":
        src = task.steps[max(0, j - 1)]
    elif mapping == "same-step":
        src = task.steps[j]
    else:
        raise ValueError(mapping)
    return src.screenshot_files[-1]


_WORKER_ZIP: Optional[zipfile.ZipFile] = None
_WORKER_CFG: dict = {}


def _worker_init(zip_path: str, cfg: dict) -> None:
    global _WORKER_ZIP, _WORKER_CFG
    _WORKER_ZIP = zipfile.ZipFile(zip_path)
    _WORKER_CFG = cfg


def build_task_requests(task: Task) -> dict:
    """Build every request of one task. Returns serialized JSON lines plus
    light-weight per-request metadata (no image data) for the statistics."""
    zf = _WORKER_ZIP
    cfg = _WORKER_CFG
    history_n = cfg["history_n"]
    mapping = cfg["mapping"]
    system_prompt = cfg["system_prompt"]

    files = [observation_file(task, j, mapping) for j in range(len(task.steps))]
    cache: Dict[str, Tuple[str, int, int]] = {}
    for f in files:
        if f not in cache:
            cache[f] = process_image(zf.read(f"{task.prefix}/{f}"))

    lines, meta = [], []
    for j, step in enumerate(task.steps):
        prev_steps = task.steps[:j]
        cur_b64, w, h = cache[files[j]]
        messages = build_messages(
            system_prompt, task.instruction, history_n,
            [s.low_level for s in prev_steps],
            [s.response for s in prev_steps],
            [cache[files[i]][0] for i in range(j)],
            cur_b64,
        )
        req = {
            "messages": messages,
            "max_tokens": cfg["max_tokens"],
            "ignore_eos": cfg["ignore_eos"],
            "temperature": 0,
        }
        lines.append(req)
        # skeleton: same messages with image payloads blanked, for token counting
        skeleton = []
        dims = []
        for m in messages:
            parts = []
            for p in m["content"]:
                if p["type"] == "image_url":
                    parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64,"}})
                    dims.append((w, h))
                else:
                    parts.append(p)
            skeleton.append({"role": m["role"], "content": parts})
        meta.append({
            "task_id": task.task_id, "domain": task.domain, "step_num": step.step_num,
            "step_index": j + 1, "n_images": len(dims), "image_dims": dims,
            "screenshot": files[j], "response": step.response, "skeleton": skeleton,
        })
    return {"task_id": task.task_id, "requests": lines, "meta": meta}


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TokenCounter:
    def __init__(self, path: str):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)

    def prompt_tokens(self, skeleton: List[dict], dims: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Returns (text_tokens, image_tokens). The chat template renders each
        image as <|vision_start|><|image_pad|><|vision_end|> (3 tokens); those
        are removed from the text count and replaced by image_tokens()."""
        ids = self.tok.apply_chat_template(skeleton, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        text = len(ids) - 3 * len(dims)
        img = sum(image_tokens(w, h) for w, h in dims)
        return text, img

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    k = min(len(xs) - 1, int(math.ceil(p / 100.0 * len(xs))) - 1)
    return xs[max(0, k)]


def summarize(xs):
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(statistics.mean(xs), 1), "median": statistics.median(xs),
            "p90": pct(xs, 90), "min": min(xs), "max": max(xs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="OSWorld trajectory archive (.zip)")
    ap.add_argument("--instructions", required=True, help="tasks_instructions.json (task_id -> instruction)")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--history-n", type=int, default=4, help="agent history_n (default 4)")
    ap.add_argument("--max-tokens", type=int, default=300, help="max_tokens per request (default 300)")
    ap.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True,
                    help="ignore_eos flag written into every request (default on)")
    ap.add_argument("--max-tokens-from-trace", action="store_true",
                    help="set max_tokens to the tokenized length of the step's real response")
    ap.add_argument("--screenshot-mapping", choices=["previous-step", "same-step"], default="previous-step",
                    help="which archive png is the current observation of a step (see README)")
    ap.add_argument("--limit-tasks", type=int, default=0, help="keep only the first N tasks (after ordering)")
    ap.add_argument("--domains", nargs="*", default=None, help="keep only these domains")
    ap.add_argument("--seed", type=int, default=None, help="shuffle task order with this seed (default: sorted)")
    ap.add_argument("--stats", default=None, help="write statistics JSON here")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="HF tokenizer dir for token counts")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip token statistics")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="image-processing worker processes (default min(8, ncpu))")
    args = ap.parse_args()

    if args.max_tokens_from_trace and args.no_tokenizer:
        ap.error("--max-tokens-from-trace needs the tokenizer")

    t0 = time.time()
    with open(args.instructions) as f:
        instructions = json.load(f)

    zf = zipfile.ZipFile(args.zip)
    tasks = list_tasks(zf)
    n_archive = len(tasks)
    if args.domains:
        keep = set(args.domains)
        tasks = [t for t in tasks if t.domain in keep]
    if args.seed is not None:
        random.Random(args.seed).shuffle(tasks)
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]

    skipped: List[dict] = []
    unaligned: List[dict] = []
    ready: List[Task] = []
    for t in tasks:
        read_task(zf, t)
        if t.task_id not in instructions:
            skipped.append({"task": f"{t.domain}/{t.task_id}", "reason": "no instruction"})
            continue
        t.instruction = instructions[t.task_id]
        if not t.steps:
            skipped.append({"task": f"{t.domain}/{t.task_id}",
                            "reason": t.align_note or t.error or "no recorded steps"})
            continue
        if t.align_note:
            unaligned.append({"task": f"{t.domain}/{t.task_id}", "aligned_steps": len(t.steps),
                              "recorded_steps": t.n_step_groups, "note": t.align_note})
        ready.append(t)
    zf.close()

    counter = None if args.no_tokenizer else TokenCounter(args.tokenizer)
    cfg = {"history_n": args.history_n, "mapping": args.screenshot_mapping,
           "system_prompt": build_system_prompt("relative"),
           "max_tokens": args.max_tokens, "ignore_eos": bool(args.ignore_eos)}

    per_request: List[dict] = []
    images_hist: collections.Counter = collections.Counter()
    n_requests = 0
    out_bytes = 0
    n_tasks_done = [0]
    print(f"[build_replay] {len(ready)} tasks with steps ({n_archive} in archive, "
          f"{len(skipped)} skipped, {len(unaligned)} partially aligned); workers={args.workers}",
          file=sys.stderr)

    def consume(result: dict, fout) -> None:
        nonlocal n_requests, out_bytes
        for req, m in zip(result["requests"], result["meta"]):
            if args.max_tokens_from_trace:
                req["max_tokens"] = max(1, counter.count(m["response"]))
            line = json.dumps(req, ensure_ascii=False) + "\n"
            fout.write(line)
            out_bytes += len(line.encode("utf-8"))
            n_requests += 1
            images_hist[m["n_images"]] += 1
            rec = {"task": f"{m['domain']}/{m['task_id']}", "step_num": m["step_num"],
                   "step_index": m["step_index"], "n_images": m["n_images"],
                   "screenshot": m["screenshot"], "max_tokens": req["max_tokens"]}
            if counter is not None:
                text_tok, img_tok = counter.prompt_tokens(m["skeleton"], m["image_dims"])
                rec.update({"text_tokens": text_tok, "image_tokens": img_tok,
                            "prompt_tokens": text_tok + img_tok,
                            "response_tokens": counter.count(m["response"])})
            per_request.append(rec)
        n_tasks_done[0] += 1
        if n_tasks_done[0] % 25 == 0:
            print(f"[build_replay] {n_tasks_done[0]}/{len(ready)} tasks, {n_requests} requests, "
                  f"{out_bytes / 1e9:.2f} GB, {time.time() - t0:.0f}s", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as fout:
        if args.workers > 1:
            with mp.Pool(args.workers, initializer=_worker_init, initargs=(args.zip, cfg)) as pool:
                for result in pool.imap(build_task_requests, ready):
                    consume(result, fout)
        else:
            _worker_init(args.zip, cfg)
            for t in ready:
                consume(build_task_requests(t), fout)

    # ---- statistics ----
    full = args.history_n + 1
    stats = {
        "zip": os.path.abspath(args.zip),
        "out": os.path.abspath(args.out),
        "history_n": args.history_n,
        "screenshot_mapping": args.screenshot_mapping,
        "max_tokens": None if args.max_tokens_from_trace else args.max_tokens,
        "max_tokens_from_trace": args.max_tokens_from_trace,
        "ignore_eos": bool(args.ignore_eos),
        "tasks_in_archive": n_archive,
        "tasks_selected": len(tasks),
        "tasks_with_requests": len(ready),
        "tasks_skipped": skipped,
        "tasks_partially_aligned": unaligned,
        "requests": n_requests,
        "recorded_steps_total": sum(t.n_step_groups for t in tasks),
        "model_calls_total": sum(t.n_blocks for t in tasks),
        "output_bytes": out_bytes,
        "images_per_request": {str(k): v for k, v in sorted(images_hist.items())},
        "elapsed_s": round(time.time() - t0, 1),
    }
    if counter is not None:
        stats["prompt_tokens"] = summarize([r["prompt_tokens"] for r in per_request])
        stats["text_tokens"] = summarize([r["text_tokens"] for r in per_request])
        stats["image_tokens"] = summarize([r["image_tokens"] for r in per_request])
        stats["prompt_tokens_full_history"] = summarize(
            [r["prompt_tokens"] for r in per_request if r["n_images"] == full])
        stats["prompt_tokens_by_n_images"] = {
            str(k): summarize([r["prompt_tokens"] for r in per_request if r["n_images"] == k])
            for k in sorted(images_hist)}
        stats["response_tokens"] = summarize([r["response_tokens"] for r in per_request])
        stats["max_tokens_stats"] = summarize([r["max_tokens"] for r in per_request])
    stats["per_request"] = per_request
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=1)

    # ---- summary ----
    print("=== build_replay summary ===")
    print(f"tasks in archive: {n_archive}; selected: {len(tasks)}; with requests: {len(ready)}; "
          f"skipped: {len(skipped)}")
    print(f"recorded steps (model calls with an executed action): {stats['recorded_steps_total']}; "
          f"model calls in logs: {stats['model_calls_total']}; requests written: {n_requests}")
    print(f"output: {args.out} ({out_bytes / 1e9:.2f} GB)")
    print("images per request: " + ", ".join(f"{k}:{v}" for k, v in sorted(images_hist.items())))
    if counter is not None:
        pt = stats["prompt_tokens"]
        print(f"prompt tokens (all): mean {pt['mean']}, median {pt['median']}, p90 {pt['p90']}, max {pt['max']}")
        pf = stats["prompt_tokens_full_history"]
        print(f"prompt tokens ({full} images, full history): n {pf.get('n', 0)}, mean {pf.get('mean')}, "
              f"median {pf.get('median')}, p90 {pf.get('p90')}")
        print(f"text tokens: mean {stats['text_tokens']['mean']}; image tokens: mean {stats['image_tokens']['mean']}")
        rt = stats["response_tokens"]
        print(f"real response tokens: mean {rt['mean']}, median {rt['median']}, p90 {rt['p90']}, max {rt['max']}")
        mt = stats["max_tokens_stats"]
        print(f"max_tokens written: mean {mt['mean']}, min {mt['min']}, max {mt['max']}; ignore_eos={bool(args.ignore_eos)}")
    if unaligned:
        print("partially aligned tasks (truncated to aligned prefix):")
        for u in unaligned:
            print(f"  {u['task']}: {u['aligned_steps']}/{u['recorded_steps']} steps -- {u['note']}")
    else:
        print("partially aligned tasks: none")
    if skipped:
        print(f"skipped tasks ({len(skipped)}):")
        for s in skipped:
            print(f"  {s['task']}: {s['reason'][:100]}")
    print(f"elapsed: {stats['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
