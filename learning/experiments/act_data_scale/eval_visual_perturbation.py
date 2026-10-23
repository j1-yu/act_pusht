#!/usr/bin/env python3
"""Evaluate ACT under deterministic image corruptions on PushT."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device


class ImageCorruption:
    """Apply input-only corruption before the saved ACT normalizer."""

    def __init__(self, pipeline, kind: str, severity: float, seed: int, device: torch.device):
        self.pipeline = pipeline
        self.kind = kind
        self.severity = severity
        # The environment image is still on CPU here; the saved policy
        # preprocessor moves it to CUDA after this corruption step.
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = self.pipeline(observation)
        image = observation.get("observation.image")
        if image is None:
            raise KeyError("Expected LeRobot key 'observation.image' after environment preprocessing")
        if self.kind == "clean":
            return observation

        image = image.clone()
        if self.kind == "brightness":
            image.mul_(self.severity).clamp_(0, 1)
        elif self.kind == "gaussian_noise":
            noise = torch.randn(
                image.shape,
                dtype=image.dtype,
                device="cpu",
                generator=self.generator,
            )
            image.add_(noise * self.severity).clamp_(0, 1)
        elif self.kind == "occlusion":
            height, width = image.shape[-2:]
            side_fraction = math.sqrt(self.severity)
            patch_h = max(1, round(height * side_fraction))
            patch_w = max(1, round(width * side_fraction))
            tops = torch.randint(
                max(1, height - patch_h + 1),
                (image.shape[0],),
                generator=self.generator,
                device="cpu",
            )
            lefts = torch.randint(
                max(1, width - patch_w + 1),
                (image.shape[0],),
                generator=self.generator,
                device="cpu",
            )
            for batch_ix, (top, left) in enumerate(zip(tops.tolist(), lefts.tolist(), strict=True)):
                image[batch_ix, :, top : top + patch_h, left : left + patch_w] = 0
        else:
            raise ValueError(f"Unknown corruption: {self.kind}")

        observation["observation.image"] = image
        return observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corruption", choices=["clean", "brightness", "gaussian_noise", "occlusion"], required=True)
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--corruption-seed", type=int, default=22001)
    parser.add_argument("--n-action-steps", type=int, default=20)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--render-episodes", type=int, default=0)
    args = parser.parse_args()

    if args.corruption == "occlusion" and not 0 < args.severity <= 1:
        parser.error("For occlusion, --severity must be the fraction of image area in (0, 1].")
    if args.render_episodes < 0 or args.render_episodes > args.n_episodes:
        parser.error("--render-episodes must be between 0 and --n-episodes.")

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.mkdir(parents=True)
    set_seed(args.seed)
    device = get_safe_torch_device("cuda", log=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    env_cfg = PushtEnv()
    policy_cfg = PreTrainedConfig.from_pretrained(args.model)
    policy_cfg.n_action_steps = args.n_action_steps
    policy_cfg.device = str(device)
    policy_cfg.pretrained_path = args.model
    envs = make_env(env_cfg, n_envs=args.n_episodes)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg).eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.model,
        preprocessor_overrides=preprocessor_overrides,
    )
    base_env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )
    env_preprocessor = ImageCorruption(
        base_env_preprocessor,
        args.corruption,
        args.severity,
        args.corruption_seed,
        device,
    )

    try:
        with torch.inference_mode():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.n_episodes,
                max_episodes_rendered=args.render_episodes,
                videos_dir=args.output / "videos",
                start_seed=args.seed,
                max_parallel_tasks=1,
            )
    finally:
        close_envs(envs)

    result = {
        **info,
        "protocol": {
            "model": str(args.model),
            "corruption": args.corruption,
            "severity": args.severity,
            "environment_seed_start": args.seed,
            "corruption_seed": args.corruption_seed,
            "n_action_steps": args.n_action_steps,
            "n_episodes": args.n_episodes,
            "corruption_applied_to": "policy RGB input in [0,1], before saved ACT normalizer",
            "physics_affected": False,
            "deterministic_torch_algorithms": True,
            "cudnn_benchmark": False,
            "tf32": False,
        },
    }
    (args.output / "eval_info.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"overall": info["overall"], "protocol": result["protocol"]}, indent=2))


if __name__ == "__main__":
    main()
