#!/usr/bin/env python3
"""Resolve wheel URLs for the heavy pinned dependencies and write an aria2c input file.

Why this exists: behind some corporate proxies pip downloads big wheels at ~0.2 MB/s
(its HTTP cache plus one long-lived connection), while aria2c with 8 connections gets
4-8 MB/s on the same link. This script only resolves URLs; the caller runs aria2c and
then `pip install --no-index --no-deps <dir>/*.whl`.

    python scripts/triaxial_prefetch_wheels.py --out-dir /tmp/cache/wheels
    cd /tmp/cache/wheels && aria2c -x 8 -s 8 -j 4 -k 8M -c -i urls.txt

Wheels already present with the right size are skipped. Roughly 4.5 GB in total.
"""

import argparse
import json
import os
import sys
import urllib.request

# torch 2.9.1's own pins (from its wheel metadata) plus SGLang v0.5.10's pins.
PACKAGES = [
    "torch==2.9.1",
    "torchvision==0.24.1",
    "torchaudio==2.9.1",
    "triton==3.5.1",
    "nvidia-cuda-nvrtc-cu12==12.8.93",
    "nvidia-cuda-runtime-cu12==12.8.90",
    "nvidia-cuda-cupti-cu12==12.8.90",
    "nvidia-cudnn-cu12==9.10.2.21",
    "nvidia-cublas-cu12==12.8.4.1",
    "nvidia-cufft-cu12==11.3.3.83",
    "nvidia-curand-cu12==10.3.9.90",
    "nvidia-cusolver-cu12==11.7.3.90",
    "nvidia-cusparse-cu12==12.5.8.93",
    "nvidia-cusparselt-cu12==0.7.1",
    "nvidia-nccl-cu12==2.27.5",
    "nvidia-nvshmem-cu12==3.3.20",
    "nvidia-nvtx-cu12==12.8.90",
    "nvidia-nvjitlink-cu12==12.8.93",
    "nvidia-cufile-cu12==1.13.1.3",
    "flashinfer-python==0.6.7.post2",
    "flashinfer-cubin==0.6.7.post2",
    "sglang-kernel==0.4.1",
    "xgrammar==0.1.32",
    "torchao==0.9.0",
    "transformers==5.3.0",
    "torchcodec==0.9.1",
    "timm==1.0.16",
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "quack-kernels",
    "cuda-python==12.9",
]


def score(filename: str, py_tag: str) -> int:
    """Rank a wheel for this interpreter/platform; negative means unusable."""
    if not filename.endswith(".whl"):
        return -1
    s = 0
    if py_tag in filename:
        s += 4
    elif "abi3" in filename or "py3-none" in filename or "py2.py3" in filename:
        s += 3
    elif "cp3" in filename:
        return -1
    if "aarch64" in filename or "arm64" in filename or "win" in filename:
        return -1
    if "manylinux" in filename and "x86_64" in filename:
        s += 2
    elif "any" in filename:
        s += 1
    elif "linux" in filename or "macosx" in filename:
        return -1
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--py-tag", default=f"cp{sys.version_info.major}{sys.version_info.minor}")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    entries, total = [], 0
    for req in PACKAGES:
        name, _, version = req.partition("==")
        url = f"https://pypi.org/pypi/{name}/{version}/json" if version else f"https://pypi.org/pypi/{name}/json"
        with urllib.request.urlopen(url, timeout=60) as fh:
            meta = json.load(fh)
        files = meta["urls"]
        best = max(files, key=lambda f: score(f["filename"], args.py_tag))
        if score(best["filename"], args.py_tag) < 0:
            print(f"!! no usable wheel for {req}", file=sys.stderr)
            continue
        entries.append(best)
        total += best["size"]
        print(f"{name}=={version or meta['info']['version']}: {best['filename']} "
              f"{best['size'] / 2**20:.0f} MB")

    todo = []
    for e in entries:
        path = os.path.join(args.out_dir, e["filename"])
        if os.path.exists(path) and os.path.getsize(path) == e["size"]:
            continue
        todo.append(f"{e['url']}\n  out={e['filename']}\n")
    with open(os.path.join(args.out_dir, "urls.txt"), "w") as fh:
        fh.writelines(todo)
    print(f"total {total / 2**20:.0f} MB, {len(todo)} wheel(s) still to download "
          f"-> {os.path.join(args.out_dir, 'urls.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
