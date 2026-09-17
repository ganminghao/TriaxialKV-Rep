# OSWorld replay requests for serving benchmarks

`build_replay.py` turns a public OSWorld-Verified trajectory archive into the
chat requests that OSWorld's Qwen3-VL agent (`mm_agents/qwen3vl_agent.py`,
`history_n=4`, `coordinate_type="relative"`) would send to the model at every
recorded step, written as OpenAI-format JSONL for
`python -m sglang.bench_serving --dataset-name openai`:

```json
{"messages": [...], "max_tokens": 300, "ignore_eos": true, "temperature": 0}
```

The system prompt, the `instruction_prompt`, the message order and the
content-part order (image first, then text) are copied from `predict()`.
Screenshots go through the agent's `process_image` (`smart_resize`, factor 32,
`max_pixels = 16*16*4*12800`, PIL resize, PNG, base64): every 1920x1080
archive screenshot becomes a 1920x1088 PNG = 60x34 = 2040 image tokens + 2
(`<|vision_start|>`, `<|vision_end|>`) for Qwen3-VL.

Files:

- `build_replay.py` - the builder (needs `pillow`, `transformers` for token stats).
- `tasks_instructions.json` - `task_id -> instruction` for the 369 tasks, copied
  from `evaluation_examples/examples/<domain>/<task_id>.json` of the OSWorld
  repo, so the builder does not need the repo.

## Inputs

- Archive `qwen2.5-vl-32b-instruct_15step.zip` from the
  `xlangai/ubuntu_osworld_verified_trajs` dataset, layout
  `pyautogui/screenshot/qwen2.5-vl-32b-instruct_15step/<domain>/<task_id>/`
  with `traj.jsonl`, `runtime.log`, `result.txt`, `step_<k>_<ts>.png`.
  369 tasks, 2494 `traj.jsonl` lines = 2486 executed actions + 8 `{"Error": ...}`
  lines (8 `multi_apps` tasks whose Google-Drive setup failed; they have no steps).
- The archive was produced by `mm_agents/qwen25vl_agent.py` (the log lines are
  `Qwen25VL Output:`); the replay re-formats those responses the way the
  Qwen3-VL agent would present them as history.

## Alignment of `runtime.log` blocks with `traj.jsonl` steps

`traj.jsonl` is written by `lib_run_single.run_single_example`: `step_num` is
the index of the `agent.predict()` call, incremented on every call, and one
line is written per *executed* action. So `step_num` has gaps (calls whose
parsed code list was empty execute nothing) and repeats (a call that returned
two actions, e.g. click + type, gives two lines with the same `step_num`).
`runtime.log` has exactly one `Qwen25VL Output:` block per `predict()` call
(`Generating content ...` lines can repeat on API retries and are ignored).

Rule used: **block `step_num - 1` is the model call that produced the lines
with that `step_num`**, and the block's parsed `Pyautogui code` list must equal
the list of executed actions of that `step_num`. This holds for all 369 tasks
(5000 blocks, 2473 distinct `(task, step_num)` groups, 0 mismatches). A task
failing the check would be truncated to its aligned prefix and listed under
"partially aligned" in the summary and stats; none does. (The sequential
"scan to the next block whose code equals the traj action" rule aligns only
358/369: the 11 others are exactly the tasks with a multi-action step.)

One request is written per `(task, step_num)` group, i.e. per model call that
executed at least one action: 2473 requests. The 2527 model calls that executed
nothing are not in the replay, neither as requests nor in the history of later
steps (the real agent did keep them in its history; the archive has no
screenshot for them). `Step i:` numbering in `Previous actions` counts recorded
steps.

## Which screenshot is the "current" observation of a step

Verified in `lib_run_single.py` of the OSWorld repo, both at `main` (commit
`fc31a90`, 2026-08-30) and at commit `91bc6bb` (2025-07-20, the version current
when this archive was recorded on 2025-07-22; its traj format - no `response`
key, `%Y%m%d@%H%M%S` timestamps - is exactly the archive's):

```python
obs = env._get_obs()                      # initial observation: never saved
while not done and step_idx < max_steps:
    response, actions = agent.predict(instruction, obs)
    for action in actions:
        obs, reward, done, info = env.step(action, ...)
        save obs['screenshot'] as f"step_{step_idx + 1}_{ts}.png"   # AFTER the action
        append traj line {"step_num": step_idx + 1, "action": action, ...}
    step_idx += 1
```

So `step_k.png` is the screen **after** step k's action, not what the model saw
when producing it (checked visually too: `step_2.png` of
`libreoffice_writer/0810415c` shows the Format menu that step 2 clicked open).
The observation the agent saw at recorded step k is the last png of the
previous recorded step; calls that executed nothing do not change `obs`, so the
gaps do not matter. The initial observation (before the first action) is not in
the archive.

`--screenshot-mapping` (default `previous-step`):

- `previous-step`: current screenshot of recorded step j = last png of recorded
  step j-1 (exactly what the agent saw). For the first recorded step the
  missing initial observation is replaced by that step's own png (`step_1.png`),
  so the first two requests of a task carry the same image twice.
- `same-step`: current screenshot of step j = its own png (one action later than
  what the agent saw). Same request structure and token counts.

## Other reproduction details

- Low-level instruction for `Previous actions`: derived from the response as
  `Qwen3VLAgent.parse_response` does (first `Action:` line, prefix stripped).
  The archive's `Low level instruction:` line keeps the `Action: ` prefix
  (Qwen2.5-VL agent) and is only used as the fallback (`Performing <x> action`).
- History responses are the `Qwen25VL Output` text verbatim.
- `temperature` is written as 0 (the agent's default); `top_p` is not written.
- Checked against the real agent: `qwen3vl_agent.py` imported with its API
  dependencies stubbed, `call_llm` replaced by the recorded response, fed the
  same screenshots: 198/198 requests over 31 tasks are byte-identical to the
  builder's `messages`.
- Prompt-token estimate = chat-template text tokens + 2042 per image. It equals
  the `input_ids` length produced by the HF `Qwen3VLProcessor` on the same
  request (checked on 1-, 4- and 5-image requests).

## Commands

```bash
PY=/tmp/envs/quancache/bin/python
DIR=benchmarks/triaxialkv_osworld

# full replay: 2473 requests, 6.0 GB, ~2 min with 16 workers
$PY $DIR/build_replay.py \
  --zip /tmp/datasets/osworld_trajs/qwen2.5-vl-32b-instruct_15step.zip \
  --instructions $DIR/tasks_instructions.json \
  --out /tmp/datasets/osworld_trajs/replay_15step_h4.jsonl \
  --stats /tmp/datasets/osworld_trajs/replay_15step_h4.stats.json \
  --history-n 4 --max-tokens 300 --ignore-eos --workers 16

# smoke file: first 20 requests
head -n 20 /tmp/datasets/osworld_trajs/replay_15step_h4.jsonl \
  > /tmp/datasets/osworld_trajs/replay_smoke_20.jsonl

# real decode lengths instead of a fixed 300
$PY $DIR/build_replay.py ... --max-tokens-from-trace --no-ignore-eos

# subset / shuffled task order (steps of a task always stay in order)
$PY $DIR/build_replay.py ... --domains chrome os --limit-tasks 20 --seed 0

# replay with SGLang (flags vary by version)
python -m sglang.bench_serving --backend sglang-oai-chat --host 127.0.0.1 --port 30000 \
  --model /tmp/models/Qwen3-VL-32B-Instruct --dataset-name openai \
  --dataset-path /tmp/datasets/osworld_trajs/replay_15step_h4.jsonl \
  --num-prompts 2473 --request-rate inf --max-concurrency 16
```

Other options: `--screenshot-mapping`, `--tokenizer DIR` (default
`/tmp/models/Qwen3-VL-32B-Instruct`, loaded with `HF_HUB_OFFLINE=1`),
`--no-tokenizer` (skip token stats), `--workers N` (image processing; each
worker holds one task's images, the writer streams line by line).

## Result of the full build (`previous-step`, `history_n=4`)

- 369 tasks in archive, 361 with requests, 8 skipped (setup errors), 0 partially aligned.
- 2473 requests; images per request 1:361, 2:354, 3:333, 4:299, 5:1126.
- Prompt tokens: mean 8829, median 9528, p90 12318, max 14449.
  Full-history requests (5 images): n=1126, mean 12056, median 11846, p90 12782.
  Text tokens mean 1485 (system prompt alone is ~950); image tokens mean 7344.
- Real response length (Qwen2.5-VL-32B): mean 156 tokens, median 107, p90 303.
- The paper's "average prefill 11,000 tokens" is 5 screenshots (10,210 image
  tokens) plus short text; here the text part is larger because the
  Qwen2.5-VL history responses carry a `Thought:` paragraph.

Notes for `bench_serving`: its `openai` loader reads the whole JSONL into
memory (6 GB) and counts each image as 3 tokens in its own `prompt_len`
(`apply_chat_template` without image expansion), so its reported input-token
totals undercount by 2039 per image; use `replay_15step_h4.stats.json`
(`per_request[*].prompt_tokens`) for the real prefill lengths. Per-line
`ignore_eos`/`temperature` override the loader's defaults;
`--sharegpt-output-len` overrides `max_tokens`.
