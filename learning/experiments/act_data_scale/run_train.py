#!/usr/bin/env python3
"""Launch one reproducible ACT data-scale training run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["10pct", "25pct", "50pct", "100pct"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--trainer", default="lerobot-train")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, help="optional combined stdout/stderr log")
    args = parser.parse_args()

    split = json.loads((HERE / "episode_split.json").read_text())
    episode_ids = split["groups"][args.group]["episode_ids"]
    output = args.output_root / f"{args.group}_seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"输出目录已存在，为避免覆盖而停止：{output}")

    command = [
        args.trainer,
        "--policy.type=act",
        "--policy.n_decoder_layers=7",
        "--policy.n_action_steps=1",
        "--policy.chunk_size=100",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--dataset.repo_id=lerobot/pusht",
        "--dataset.video_backend=torchcodec",
        f"--dataset.episodes={episode_ids}",
        f"--output_dir={output}",
        f"--job_name=act_data_scale_{args.group}_seed{args.seed}",
        "--wandb.enable=false",
        "--batch_size=8",
        f"--steps={args.steps}",
        "--save_freq=50000",
        f"--seed={args.seed}",
        "--num_workers=4",
    ]
    print(f"group={args.group}, episodes={len(episode_ids)}, seed={args.seed}, output={output}", flush=True)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        print(f"log={args.log}", flush=True)
        with args.log.open("w") as log_file:
            result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(command)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
