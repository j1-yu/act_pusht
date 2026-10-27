"""Compare visual-only and image-plus-agent-position PushT action chunkers."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
import zarr
from torch import nn
from torch.utils.data import DataLoader

from visual_action_chunking import (
    PushTImageChunks,
    VisualActionChunkCNN,
    denormalize_action,
    normalize_action,
    preload_visual_arrays,
    split_episodes,
)


class PushTImageAgentPosChunks(PushTImageChunks):
    def __init__(
        self, *args, state_cache: np.ndarray, target_mode: str = "absolute",
        use_previous_image: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        if target_mode not in {"absolute", "relative"}:
            raise ValueError(f"未知 target_mode: {target_mode}")
        self.states = state_cache
        self.target_mode = target_mode
        self.use_previous_image = use_previous_image
        root = zarr.open(str(args[0]), mode="r")
        self.episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
        self.episode_starts = np.r_[0, self.episode_ends[:-1]]

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        index = int(self.indices[item])
        image = np.asarray(self.images[index], dtype=np.float32) / 255.0
        agent_pos = np.asarray(self.states[index, :2], dtype=np.float32) / 256.0 - 1.0
        action = np.asarray(self.actions[index : index + self.chunk_size], dtype=np.float32)
        if self.target_mode == "relative":
            current_agent_pos = np.asarray(self.states[index, :2], dtype=np.float32)
            action = (action - current_agent_pos[None, :]) / 256.0
        else:
            action = normalize_action(action)
        result = [torch.from_numpy(image).permute(2, 0, 1)]
        if self.use_previous_image:
            episode_id = int(np.searchsorted(self.episode_ends, index, side="right"))
            previous_index = max(int(self.episode_starts[episode_id]), index - 1)
            previous = np.asarray(self.images[previous_index], dtype=np.float32) / 255.0
            result.append(torch.from_numpy(previous).permute(2, 0, 1))
        result.extend((torch.from_numpy(agent_pos), torch.from_numpy(action)))
        return tuple(result)


class VisualAgentPosActionChunkCNN(nn.Module):
    def __init__(
        self, chunk_size: int, target_mode: str = "absolute", use_previous_image: bool = False
    ) -> None:
        super().__init__()
        if target_mode not in {"absolute", "relative"}:
            raise ValueError(f"未知 target_mode: {target_mode}")
        self.chunk_size = chunk_size
        self.target_mode = target_mode
        self.use_previous_image = use_previous_image
        self.action_scale = 2.0 if target_mode == "relative" else 1.0
        self.encoder = nn.Sequential(
            nn.Conv2d(6 if use_previous_image else 3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4 + 2, 256), nn.ReLU(),
            nn.Linear(256, 2 * chunk_size), nn.Tanh(),
        )

    def forward(
        self, image: torch.Tensor, agent_pos: torch.Tensor,
        previous_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_previous_image:
            if previous_image is None:
                raise ValueError("previous_image is required when temporal input is enabled")
            image = torch.cat((previous_image, image), dim=1)
        features = self.encoder(image).flatten(1)
        combined = torch.cat((features, agent_pos), dim=1)
        return self.head(combined).reshape(-1, self.chunk_size, 2) * self.action_scale


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, use_previous_image: bool = False
) -> float:
    model.eval()
    total = 0.0
    elements = 0
    with torch.no_grad():
        for batch in loader:
            if use_previous_image:
                image, previous_image, agent_pos, target = batch
                prediction = model(image.to(device), agent_pos.to(device), previous_image.to(device))
            else:
                image, agent_pos, target = batch
                prediction = model(image.to(device), agent_pos.to(device))
            total += torch.square(prediction.cpu() - target).sum().item()
            elements += target.numel()
    return total / elements


def rollout(
    model: nn.Module,
    use_agent_pos: bool,
    target_mode: str,
    use_previous_image: bool,
    chunk_size: int,
    n_action_steps: int,
    episodes: int,
    start_seed: int,
    max_steps: int,
    device: torch.device,
) -> dict:
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
    )
    rows = []
    try:
        model.eval()
        for index in range(episodes):
            observation, _ = env.reset(seed=start_seed + index)
            previous_image = observation["pixels"].copy()
            max_coverage = 0.0
            max_reward = 0.0
            terminated = truncated = False
            steps = 0
            info = {}
            while steps < max_steps and not (terminated or truncated):
                image = torch.from_numpy(
                    observation["pixels"].astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)
                previous_image_tensor = torch.from_numpy(
                    previous_image.astype(np.float32) / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    if use_agent_pos:
                        agent_pos = torch.from_numpy(
                            np.asarray(observation["agent_pos"], dtype=np.float32) / 256.0 - 1.0
                        ).unsqueeze(0).to(device)
                        if use_previous_image:
                            normalized_actions = model(
                                image, agent_pos, previous_image_tensor
                            )[0].cpu().numpy()
                        else:
                            normalized_actions = model(image, agent_pos)[0].cpu().numpy()
                    else:
                        normalized_actions = model(image)[0].cpu().numpy()
                if use_agent_pos and target_mode == "relative":
                    current_agent_pos = np.asarray(observation["agent_pos"], dtype=np.float32)
                    predicted = normalized_actions * 256.0 + current_agent_pos[None, :]
                else:
                    predicted = denormalize_action(normalized_actions)
                for action in predicted[:n_action_steps]:
                    prior_observation_image = observation["pixels"].copy()
                    observation, reward, terminated, truncated, info = env.step(
                        np.clip(action, 0.0, 512.0).astype(np.float32)
                    )
                    previous_image = prior_observation_image
                    steps += 1
                    max_reward = max(max_reward, float(reward))
                    max_coverage = max(max_coverage, float(info.get("coverage", 0.0)))
                    if steps >= max_steps or terminated or truncated:
                        break
            rows.append({
                "seed": start_seed + index,
                "steps": steps,
                "max_reward": max_reward,
                "max_coverage": max_coverage,
                "success": bool(info.get("is_success", False)),
            })
    finally:
        env.close()
    successes = sum(row["success"] for row in rows)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_max_reward": float(np.mean([row["max_reward"] for row in rows])),
        "mean_max_coverage": float(np.mean([row["max_coverage"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "episodes_raw": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/pusht/pusht_cchi_v7_replay.zarr"),
    )
    parser.add_argument("--baseline-checkpoint", type=Path,
                        default=Path("learning/runs/visual_cnn_full50_seed42.pt"))
    parser.add_argument("--agent-pos-checkpoint", type=Path,
                        default=Path("learning/runs/visual_cnn_agent_pos_full50_seed42.pt"))
    parser.add_argument("--target-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--use-previous-image", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="限制每个 split 的样本数，仅用于快速检查代码")
    parser.add_argument("--execute-steps", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--rollout-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("learning/runs/visual_cnn_agent_pos_full50_seed42.pt"))
    parser.add_argument("--metrics", type=Path,
                        default=Path("learning/runs/visual_cnn_agent_pos_full50_seed42.json"))
    args = parser.parse_args()
    if args.epochs < 1 or args.chunk_size < 1 or any(
        not 1 <= steps <= args.chunk_size for steps in args.execute_steps
    ):
        raise ValueError("chunk_size 必须为正数，且 execute_steps 必须在 [1, chunk_size] 范围内")
    if args.target_mode == "relative" and args.use_previous_image:
        raise ValueError("请分开测试相对位移和前一帧输入，每次只改变一个因素")
    if args.rollout_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA rollout was requested but is not available")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout_device = torch.device(args.rollout_device)

    train_ids, val_ids = split_episodes(args.data, args.seed, args.validation_ratio)
    image_cache, action_cache = preload_visual_arrays(args.data)
    root = zarr.open(str(args.data), mode="r")
    state_cache = np.asarray(root["data"]["state"], dtype=np.float32)
    train_set = PushTImageAgentPosChunks(
        args.data, args.chunk_size, train_ids, args.max_samples, "spread",
        image_cache, action_cache, state_cache=state_cache, target_mode=args.target_mode,
        use_previous_image=args.use_previous_image,
    )
    val_set = PushTImageAgentPosChunks(
        args.data, args.chunk_size, val_ids, args.max_samples, "spread",
        image_cache, action_cache, state_cache=state_cache, target_mode=args.target_mode,
        use_previous_image=args.use_previous_image,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VisualAgentPosActionChunkCNN(
        args.chunk_size, args.target_mode, args.use_previous_image
    ).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_mse = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        samples = 0
        for batch in train_loader:
            if args.use_previous_image:
                image, previous_image, agent_pos, target = batch
                prediction = model(
                    image.to(train_device), agent_pos.to(train_device), previous_image.to(train_device)
                )
            else:
                image, agent_pos, target = batch
                prediction = model(image.to(train_device), agent_pos.to(train_device))
            loss = torch.square(prediction - target.to(train_device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(image)
            samples += len(image)
        train_mse = total / samples
        val_mse = evaluate(model, val_loader, train_device, args.use_previous_image)
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse})
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"epoch {epoch:03d}/{args.epochs} train_mse={train_mse:.5f} "
            f"val_mse={val_mse:.5f} device={train_device}",
            flush=True,
        )

    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "chunk_size": args.chunk_size,
        "target_mode": args.target_mode,
        "use_previous_image": args.use_previous_image,
        "seed": args.seed,
        "episode_split": {"train": train_ids, "validation": val_ids},
    }, args.checkpoint)

    baseline_payload = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)
    baseline = VisualActionChunkCNN(args.chunk_size)
    baseline.load_state_dict(baseline_payload["model_state_dict"])
    agent_pos_payload = torch.load(args.agent_pos_checkpoint, map_location="cpu", weights_only=False)
    agent_pos_baseline = VisualAgentPosActionChunkCNN(args.chunk_size, "absolute")
    agent_pos_baseline.load_state_dict(agent_pos_payload["model_state_dict"])
    relative_payload = torch.load(
        Path("learning/runs/visual_cnn_agent_pos_relative_seed42.pt"),
        map_location="cpu", weights_only=False,
    )
    relative_baseline = VisualAgentPosActionChunkCNN(args.chunk_size, "relative")
    relative_baseline.load_state_dict(relative_payload["model_state_dict"])
    model.to(rollout_device).eval()
    baseline.to(rollout_device).eval()
    agent_pos_baseline.to(rollout_device).eval()
    relative_baseline.to(rollout_device).eval()
    comparisons = []
    for name, policy, use_agent_pos, policy_target_mode, policy_uses_previous_image, checkpoint in (
        ("visual_only", baseline, False, "absolute", False, str(args.baseline_checkpoint)),
        ("visual_plus_agent_pos_absolute", agent_pos_baseline, True, "absolute", False,
         str(args.agent_pos_checkpoint)),
        ("visual_plus_agent_pos_relative", relative_baseline, True, "relative", False,
         "learning/runs/visual_cnn_agent_pos_relative_seed42.pt"),
        (f"visual_plus_agent_pos_{args.target_mode}{'_two_frame' if args.use_previous_image else ''}",
         model, True, args.target_mode, args.use_previous_image,
         str(args.checkpoint)),
    ):
        for execute_steps in args.execute_steps:
            result = rollout(
                policy, use_agent_pos, policy_target_mode, policy_uses_previous_image,
                args.chunk_size, execute_steps,
                args.episodes, args.start_seed, args.max_steps, rollout_device,
            )
            comparisons.append({
                "model": name,
                "checkpoint": checkpoint,
                "n_action_steps": execute_steps,
                "rollout": result,
            })
            print(
                f"{name} execute={execute_steps} success={result['successes']}/{args.episodes} "
                f"coverage={result['mean_max_coverage']:.4f}",
                flush=True,
            )

    report = {
        "data": str(args.data),
        "seed": args.seed,
        "device": str(train_device),
        "rollout_device": str(rollout_device),
        "chunk_size": args.chunk_size,
        "target_mode": args.target_mode,
        "use_previous_image": args.use_previous_image,
        "epochs": args.epochs,
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "max_samples": args.max_samples,
        "best_validation_mse": best_mse,
        "best_epoch": min(history, key=lambda row: row["validation_mse"])["epoch"],
        "history": history,
        "episode_split": {"train": train_ids, "validation": val_ids},
        "comparisons": comparisons,
        "checkpoint": str(args.checkpoint),
    }
    args.metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved checkpoint: {args.checkpoint}", flush=True)
    print(f"saved metrics: {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
