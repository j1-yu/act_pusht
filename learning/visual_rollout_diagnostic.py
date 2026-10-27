"""Record step-by-step failure traces for a visual PushT policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
from PIL import Image

from visual_proprio_action_chunking_transformer import VisualProprioTransformer


def load_policy(checkpoint: Path, device: torch.device) -> tuple[VisualProprioTransformer, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = VisualProprioTransformer(
        int(payload["chunk_size"]), int(payload["d_model"]), int(payload["layers"])
    )
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), int(payload["chunk_size"])


def first_reach(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values):
        if value >= threshold:
            return index + 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("learning/runs/visual_proprio_transformer_short_full_data_seed42.pt"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-delta", type=float, default=None,
                        help="限制单步目标相对当前末端的最大位移（像素）")
    parser.add_argument("--save-seeds", type=int, nargs="*", default=[],
                        help="保存这些 seed 的关键 rollout 画面")
    parser.add_argument("--frame-dir", type=Path,
                        default=Path("learning/runs/visual_rollout_frames_seed42"))
    parser.add_argument("--output", type=Path,
                        default=Path("learning/runs/visual_proprio_transformer_rollout_diagnostic_seed42.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, chunk_size = load_policy(args.checkpoint, device)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    episodes = []
    try:
        for episode_index in range(args.episodes):
            seed = args.start_seed + episode_index
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            rows = []
            save_frames = seed in args.save_seeds
            if save_frames:
                (args.frame_dir / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
            while len(rows) < args.max_steps and not (terminated or truncated):
                step_number = len(rows) + 1
                if save_frames and step_number in {1, 25, 50, 100, 200, 300}:
                    Image.fromarray(np.asarray(observation["pixels"], dtype=np.uint8)).save(
                        args.frame_dir / f"seed_{seed}" / f"step_{step_number:03d}.png"
                    )
                current_position = np.asarray(observation["agent_pos"], dtype=np.float32)
                image = torch.from_numpy(observation["pixels"].astype(np.float32) / 255.0)
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                position = torch.from_numpy(current_position / 256.0 - 1.0)
                position = position.unsqueeze(0).to(device)
                with torch.no_grad():
                    normalized_chunk = model(image, position)[0].cpu().numpy()
                target = (normalized_chunk[0] + 1.0) * 256.0
                delta = target - current_position
                if args.max_delta is not None:
                    delta_norm = float(np.linalg.norm(delta))
                    if delta_norm > args.max_delta:
                        delta = delta * (args.max_delta / max(delta_norm, 1e-8))
                        target = current_position + delta
                observation, reward, terminated, truncated, info = env.step(
                    np.clip(target, 0.0, 512.0).astype(np.float32)
                )
                rows.append({
                    "step": step_number,
                    "agent_position_before": current_position.tolist(),
                    "predicted_target": target.tolist(),
                    "predicted_delta": delta.tolist(),
                    "predicted_delta_norm": float(np.linalg.norm(delta)),
                    "agent_position_after": np.asarray(observation["agent_pos"], dtype=np.float32).tolist(),
                    "reward": float(reward),
                    "coverage": float(info.get("coverage", 0.0)),
                    "is_success": bool(info.get("is_success", False)),
                })
            coverages = [row["coverage"] for row in rows]
            rewards = [row["reward"] for row in rows]
            deltas = [row["predicted_delta_norm"] for row in rows]
            episodes.append({
                "seed": seed,
                "steps": len(rows),
                "max_coverage": max(coverages, default=0.0),
                "max_reward": max(rewards, default=0.0),
                "mean_predicted_delta_norm": float(np.mean(deltas)) if deltas else 0.0,
                "max_predicted_delta_norm": max(deltas, default=0.0),
                "first_step_coverage": coverages[0] if coverages else 0.0,
                "coverage_at_steps": {
                    str(step): coverages[step - 1] if len(coverages) >= step else None
                    for step in (1, 10, 25, 50, 100, 200, 300)
                },
                "first_reach_step": {
                    str(threshold): first_reach(coverages, threshold)
                    for threshold in (0.2, 0.4, 0.6, 0.8, 0.95)
                },
                "trace": rows,
            })
    finally:
        env.close()

    all_coverages = [episode["max_coverage"] for episode in episodes]
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "chunk_size": chunk_size,
        "episodes": args.episodes,
        "start_seed": args.start_seed,
        "max_steps": args.max_steps,
        "max_delta": args.max_delta,
        "save_seeds": args.save_seeds,
        "frame_dir": str(args.frame_dir),
        "mean_max_coverage": float(np.mean(all_coverages)) if all_coverages else 0.0,
        "episodes_raw": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"episodes={args.episodes} mean_max_coverage={report['mean_max_coverage']:.4f}")
    for episode in episodes:
        print(f"seed={episode['seed']} max_coverage={episode['max_coverage']:.4f} "
              f"first_coverage={episode['first_step_coverage']:.4f} "
              f"mean_delta={episode['mean_predicted_delta_norm']:.2f}")
    print(f"saved diagnostic: {args.output}")


if __name__ == "__main__":
    main()
