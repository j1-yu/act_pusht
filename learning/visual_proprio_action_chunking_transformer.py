"""Deterministic visual Transformer with an agent-position token for PushT."""

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

from visual_proprio_action_chunking import PushTImageAgentPosChunks
from visual_action_chunking import preload_visual_arrays, split_episodes


class VisualProprioTransformer(nn.Module):
    """CNN image tokens + agent-position token + Transformer action queries."""

    def __init__(self, chunk_size: int, d_model: int = 128, layers: int = 2,
                 target_mode: str = "absolute") -> None:
        super().__init__()
        if target_mode not in {"absolute", "relative"}:
            raise ValueError(f"未知 target_mode: {target_mode}")
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.layers = layers
        self.target_mode = target_mode
        self.action_scale = 2.0 if target_mode == "relative" else 1.0
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, d_model, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.image_position = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        self.agent_position = nn.Linear(2, d_model)
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.action_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2), nn.Tanh())

    def forward(self, image: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image)
        image_tokens = features.flatten(2).transpose(1, 2) + self.image_position
        position_token = self.agent_position(agent_pos).unsqueeze(1)
        memory = torch.cat((image_tokens, position_token), dim=1)
        queries = self.action_queries.expand(image.shape[0], -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded) * self.action_scale


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    elements = 0
    with torch.no_grad():
        for image, agent_pos, target in loader:
            prediction = model(image.to(device), agent_pos.to(device))
            total += torch.square(prediction.cpu() - target).sum().item()
            elements += target.numel()
    return total / elements


def rollout(model: nn.Module, chunk_size: int, n_action_steps: int,
            episodes: int, start_seed: int, max_steps: int,
            device: torch.device, target_mode: str) -> dict:
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    rows = []
    try:
        model.eval()
        for index in range(episodes):
            observation, _ = env.reset(seed=start_seed + index)
            max_coverage = 0.0
            max_reward = 0.0
            terminated = truncated = False
            steps = 0
            info = {}
            while steps < max_steps and not (terminated or truncated):
                image = torch.from_numpy(observation["pixels"].astype(np.float32) / 255.0)
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                position = torch.from_numpy(
                    np.asarray(observation["agent_pos"], dtype=np.float32) / 256.0 - 1.0
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    normalized = model(image, position)[0].cpu().numpy()
                if target_mode == "relative":
                    current_position = np.asarray(observation["agent_pos"], dtype=np.float32)
                    predicted = normalized * 256.0 + current_position[None, :]
                else:
                    predicted = (normalized + 1.0) * 256.0
                for action in predicted[:n_action_steps]:
                    observation, reward, terminated, truncated, info = env.step(
                        np.clip(action, 0.0, 512.0).astype(np.float32)
                    )
                    steps += 1
                    max_reward = max(max_reward, float(reward))
                    max_coverage = max(max_coverage, float(info.get("coverage", 0.0)))
                    if steps >= max_steps or terminated or truncated:
                        break
            rows.append({"seed": start_seed + index, "steps": steps,
                         "max_reward": max_reward, "max_coverage": max_coverage,
                         "success": bool(info.get("is_success", False))})
    finally:
        env.close()
    successes = sum(row["success"] for row in rows)
    return {
        "episodes": episodes, "successes": successes,
        "success_rate": successes / episodes,
        "mean_max_reward": float(np.mean([row["max_reward"] for row in rows])),
        "mean_max_coverage": float(np.mean([row["max_coverage"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "episodes_raw": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("data/pusht/pusht_cchi_v7_replay.zarr"))
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--n-action-steps", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--target-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("learning/runs/visual_proprio_transformer_short_seed42.pt"))
    parser.add_argument("--metrics", type=Path,
                        default=Path("learning/runs/visual_proprio_transformer_short_seed42.json"))
    args = parser.parse_args()
    if args.chunk_size < 1 or not 1 <= args.n_action_steps <= args.chunk_size:
        raise ValueError("需要满足 chunk_size >= 1 且 1 <= n_action_steps <= chunk_size")
    if args.d_model % 4 or args.layers < 1 or args.max_samples < 1:
        raise ValueError("d_model 必须能被 4 整除，layers/max_samples >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, val_ids = split_episodes(args.data, args.seed, args.validation_ratio)
    image_cache, action_cache = preload_visual_arrays(args.data)
    root = zarr.open(str(args.data), mode="r")
    state_cache = np.asarray(root["data"]["state"], dtype=np.float32)
    train_set = PushTImageAgentPosChunks(
        args.data, args.chunk_size, train_ids, args.max_samples, "spread",
        image_cache, action_cache, state_cache=state_cache, target_mode=args.target_mode,
    )
    val_set = PushTImageAgentPosChunks(
        args.data, args.chunk_size, val_ids, args.max_samples, "spread",
        image_cache, action_cache, state_cache=state_cache, target_mode=args.target_mode,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = VisualProprioTransformer(
        args.chunk_size, args.d_model, args.layers, args.target_mode
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_mse = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        samples = 0
        for image, agent_pos, target in train_loader:
            prediction = model(image.to(device), agent_pos.to(device))
            loss = torch.square(prediction - target.to(device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(image)
            samples += len(image)
        train_mse = total / samples
        val_mse = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse})
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch:03d}/{args.epochs} train_mse={train_mse:.5f} "
              f"val_mse={val_mse:.5f} device={device}", flush=True)

    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "chunk_size": args.chunk_size,
                "target_mode": args.target_mode,
                "d_model": args.d_model, "layers": args.layers, "seed": args.seed,
                "episode_split": {"train": train_ids, "validation": val_ids}}, args.checkpoint)
    rollout_result = rollout(model, args.chunk_size, args.n_action_steps, args.episodes,
                             100000, args.max_steps, device, args.target_mode)
    result = {"data": str(args.data), "model": "CNN image tokens + agent-position token + Transformer",
              "target_mode": args.target_mode,
              "chunk_size": args.chunk_size, "n_action_steps": args.n_action_steps,
              "d_model": args.d_model, "layers": args.layers, "epochs": args.epochs,
              "seed": args.seed, "device": str(device), "max_samples": args.max_samples,
              "train_samples": len(train_set), "validation_samples": len(val_set),
              "best_validation_mse": best_mse,
              "best_epoch": min(history, key=lambda row: row["validation_mse"])["epoch"],
              "history": history, "episode_split": {"train": train_ids, "validation": val_ids},
              "rollout": rollout_result, "checkpoint": str(args.checkpoint)}
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"rollout success={rollout_result['successes']}/{args.episodes} "
          f"mean_max_coverage={rollout_result['mean_max_coverage']:.4f}", flush=True)
    print(f"saved checkpoint: {args.checkpoint}", flush=True)
    print(f"saved metrics: {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
