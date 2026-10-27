"""Evaluate the minimal state-only BC policy in the Gymnasium PushT task."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  Registers gym_pusht/PushT-v0.
import numpy as np
import pyarrow.parquet as pq
import torch

from bc_pusht import BehaviorCloningMLP


def normalize_observation(values: np.ndarray, obs_mode: str) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    values[..., :4] = values[..., :4] / 256.0 - 1.0
    if obs_mode == "full_state":
        values[..., 4] = values[..., 4] / np.pi - 1.0
    return values


def denormalize_pixels(values: np.ndarray) -> np.ndarray:
    return (values + 1.0) * 256.0


def load_policy(path: Path) -> tuple[BehaviorCloningMLP, np.ndarray, str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    obs_mode = checkpoint.get("obs_mode", "agent_pos")
    model = BehaviorCloningMLP(input_dim=checkpoint.get("input_dim", 2))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, np.asarray(checkpoint["train_action_mean"], dtype=np.float32), obs_mode


def compute_train_action_mean(data_path: Path, train_episode_ids: list[int]) -> np.ndarray:
    table = pq.read_table(data_path, columns=["action", "episode_index"])
    episode = table["episode_index"].to_numpy()
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    mask = np.isin(episode, np.asarray(train_episode_ids, dtype=episode.dtype))
    return action[mask].mean(axis=0).astype(np.float32)


def evaluate_episode(
    env: gym.Env,
    model: BehaviorCloningMLP | None,
    mean_action: np.ndarray,
    obs_mode: str,
    seed: int,
    max_steps: int,
) -> dict:
    observation, _ = env.reset(seed=seed)
    max_reward = 0.0
    max_coverage = 0.0
    terminated = False
    truncated = False
    for step in range(max_steps):
        if model is None:
            action = mean_action
        else:
            policy_observation = observation[:2] if obs_mode == "agent_pos" else observation[:5]
            input_tensor = torch.from_numpy(
                normalize_observation(policy_observation, obs_mode)
            ).unsqueeze(0)
            with torch.no_grad():
                normalized_action = model(input_tensor).squeeze(0).numpy()
            action = denormalize_pixels(normalized_action)
        action = np.clip(action, 0.0, 512.0).astype(np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        max_reward = max(max_reward, float(reward))
        max_coverage = max(max_coverage, float(info.get("coverage", reward * 0.95)))
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "steps": step + 1,
        "max_reward": max_reward,
        "max_coverage": max_coverage,
        "success": bool(info.get("is_success", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def summarize(rows: list[dict]) -> dict:
    successes = sum(row["success"] for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "mean_max_reward": float(np.mean([row["max_reward"] for row in rows])),
        "mean_max_coverage": float(np.mean([row["max_coverage"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "episodes_raw": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("learning/runs/bc_pusht_seed42.pt"),
    )
    parser.add_argument("--obs-mode", choices=["auto", "agent_pos", "full_state"], default="auto")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/pusht/lerobot/file-000.parquet"),
        help="用于计算训练集平均动作的 LeRobot PushT parquet 文件",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("learning/runs/bc_rollout_results.json"),
    )
    parser.add_argument("--mode", choices=["bc", "mean"], default="bc")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    if args.mode == "bc":
        model, mean_action, checkpoint_obs_mode = load_policy(args.checkpoint)
        obs_mode = checkpoint_obs_mode if args.obs_mode == "auto" else args.obs_mode
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = None
        checkpoint_obs_mode = checkpoint.get("obs_mode", "agent_pos")
        obs_mode = checkpoint_obs_mode if args.obs_mode == "auto" else args.obs_mode
        mean_action = compute_train_action_mean(
            args.data.expanduser(), checkpoint["episode_split"]["train"]
        )
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    try:
        rows = [
            evaluate_episode(
                env,
                model,
                mean_action,
                obs_mode,
                seed=args.start_seed + index,
                max_steps=args.max_steps,
            )
            for index in range(args.episodes)
        ]
    finally:
        env.close()

    result = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint) if args.mode == "bc" else None,
        "constant_action": mean_action.tolist(),
        "environment": "gym_pusht/PushT-v0",
        "obs_type": f"state; policy consumes {obs_mode}",
        "success_definition": "coverage > 0.95",
        "start_seed": args.start_seed,
        "max_steps": args.max_steps,
        **summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"mode={args.mode}  success={result['successes']}/{result['episodes']} "
        f"({result['success_rate']:.1%})  "
        f"mean_max_coverage={result['mean_max_coverage']:.4f}  "
        f"mean_steps={result['mean_steps']:.1f}"
    )
    print(f"saved results: {args.output}")


if __name__ == "__main__":
    main()
