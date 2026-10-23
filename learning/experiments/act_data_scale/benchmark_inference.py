#!/usr/bin/env python3
"""Measure batch-1 ACT chunk prediction latency with CUDA events.

This times the neural-network forward pass that predicts one 100-action chunk.
It excludes environment simulation, image preprocessing, video writing and the
19 cached-action retrievals between replans when n_action_steps=20.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lerobot.policies.act.modeling_act import ACTPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    policy = ACTPolicy.from_pretrained(args.model).to("cuda").eval()
    batch = {
        "observation.image": torch.rand(1, 3, 96, 96, device="cuda"),
        "observation.state": torch.rand(1, 2, device="cuda"),
    }
    with torch.inference_mode():
        for _ in range(args.warmup):
            policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        times = []
        for _ in range(args.repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            policy.predict_action_chunk(batch)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))

    values = torch.tensor(times)
    result = {
        "model": str(args.model),
        "device": torch.cuda.get_device_name(0),
        "batch_size": 1,
        "predicted_chunk_size": policy.config.chunk_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "mean_ms": round(values.mean().item(), 3),
        "median_ms": round(values.median().item(), 3),
        "p95_ms": round(torch.quantile(values, 0.95).item(), 3),
        "scope": "ACT predict_action_chunk forward only; excludes preprocessing, simulation and video",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
