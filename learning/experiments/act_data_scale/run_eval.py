#!/usr/bin/env python3
"""Evaluate one ACT checkpoint with the shared PushT protocol."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1000, help="first of 50 fixed environment seeds")
    parser.add_argument("--n-action-steps", type=int, default=20)
    parser.add_argument("--evaluator", default="lerobot-eval")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，为避免覆盖而停止：{args.output}")
    command = [
        args.evaluator,
        f"--policy.path={args.model}",
        "--env.type=pusht",
        "--policy.device=cuda",
        f"--policy.n_action_steps={args.n_action_steps}",
        "--eval.n_episodes=50",
        "--eval.batch_size=50",
        f"--seed={args.seed}",
        f"--output_dir={args.output}",
        f"--job_name={args.output.name}",
    ]
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
