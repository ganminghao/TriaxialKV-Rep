"""End-to-end sanity check for TriAxialKV: send fixed greedy chat prompts to a running
SGLang server and save the outputs; compare two saved runs (e.g. bf16 vs triaxial).

Usage:
  python -m sglang.test.triaxial_e2e_check run  --url http://127.0.0.1:30000 --out bf16.json
  python -m sglang.test.triaxial_e2e_check run  --url http://127.0.0.1:30001 --out tri.json
  python -m sglang.test.triaxial_e2e_check diff bf16.json tri.json
  # multimodal: add --replay /tmp/datasets/osworld_trajs/replay_smoke_20.jsonl --num 4
"""

import argparse
import json
import sys
import time

import requests


def text_prompts():
    long_ctx = (
        "The following is a log of a data-center incident. "
        + " ".join(
            f"At {h:02d}:{m:02d} node-{(h*7+m)%23:02d} reported temperature {40 + (h*m)%17} C and fan speed {1200 + (h*13+m*7)%900} rpm."
            for h in range(0, 24)
            for m in range(0, 60, 15)
        )
    )
    return [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain in three sentences why the sky is blue."},
        ],
        [
            {"role": "system", "content": "You are a careful assistant that answers precisely."},
            {"role": "user", "content": long_ctx + "\n\nWhich node reported the highest temperature and when?"},
            {"role": "assistant", "content": "Let me scan the log for the maximum temperature value."},
            {"role": "user", "content": "Go ahead, and also report the fan speed at that time."},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively."},
            {"role": "assistant", "content": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
            {"role": "user", "content": "Now add memoization and explain the complexity in two sentences."},
        ],
        [
            {"role": "user", "content": "List the capitals of France, Japan, Brazil, Kenya and Canada as a JSON object."},
        ],
    ]


def run(args):
    if args.replay:
        prompts = []
        with open(args.replay) as f:
            for line in f:
                if len(prompts) >= args.num:
                    break
                prompts.append(json.loads(line)["messages"])
    else:
        prompts = text_prompts()
    results = []
    for i, messages in enumerate(prompts):
        body = {
            "model": "default",
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "top_p": 1,
        }
        t0 = time.time()
        r = requests.post(f"{args.url}/v1/chat/completions", json=body, timeout=600)
        r.raise_for_status()
        js = r.json()
        text = js["choices"][0]["message"]["content"]
        usage = js.get("usage", {})
        dt = time.time() - t0
        print(f"--- prompt {i}: prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')} {dt:.1f}s")
        print(text[:600])
        results.append({"idx": i, "text": text, "usage": usage})
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"saved {args.out}")


def diff(args):
    a = json.load(open(args.a))
    b = json.load(open(args.b))
    for ra, rb in zip(a, b):
        ta, tb = ra["text"], rb["text"]
        n = 0
        for x, y in zip(ta.split(), tb.split()):
            if x != y:
                break
            n += 1
        print(
            f"prompt {ra['idx']}: common word prefix {n} / {len(ta.split())} vs {len(tb.split())} words; "
            f"identical={ta == tb}"
        )


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:30000")
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=200)
    r.add_argument("--replay", default=None)
    r.add_argument("--num", type=int, default=4)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = p.parse_args()
    if args.cmd == "run":
        run(args)
    elif args.cmd == "diff":
        diff(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
