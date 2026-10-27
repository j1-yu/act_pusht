"""A minimal state-only behavior cloning baseline for LeRobot PushT data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from torch import nn


class BehaviorCloningMLP(nn.Module):
    def __init__(self, input_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


def load_data(path: Path, obs_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix == ".zarr" or path.is_dir():
        root = zarr.open(str(path), mode="r")
        state = np.asarray(root["data"]["state"], dtype=np.float32)
        action = np.asarray(root["data"]["action"], dtype=np.float32)
        episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
        episode = np.repeat(np.arange(len(episode_ends)), np.diff(np.r_[0, episode_ends]))
        if obs_mode == "agent_pos":
            observation = state[:, :2]
        else:
            observation = state[:, :5]
    else:
        table = pq.read_table(path, columns=["observation.state", "action", "episode_index"])
        agent_pos = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if obs_mode != "agent_pos":
            raise ValueError("完整状态 BC 需要把 --data 指向原始 pusht_cchi_v7_replay.zarr")
        observation = agent_pos
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        episode = table["episode_index"].to_numpy().astype(np.int64, copy=False)
    return observation, action, episode


def split_episodes(
    episode: np.ndarray, seed: int, validation_ratio: float
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    episode_ids = np.unique(episode)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episode_ids)
    n_validation = max(1, int(round(len(episode_ids) * validation_ratio)))
    validation_ids = np.sort(shuffled[:n_validation])
    train_ids = np.sort(shuffled[n_validation:])
    train_mask = np.isin(episode, train_ids)
    validation_mask = np.isin(episode, validation_ids)
    return train_mask, validation_mask, train_ids.tolist(), validation_ids.tolist()


def normalize_observations(values: np.ndarray, obs_mode: str) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    values[..., :4] = values[..., :4] / 256.0 - 1.0
    if obs_mode == "full_state":
        values[..., 4] = values[..., 4] / np.pi - 1.0
    return values


def normalize_actions(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32) / 256.0 - 1.0


def evaluate(
    model: nn.Module,
    observation: torch.Tensor,
    action: torch.Tensor,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    squared_error = 0.0
    absolute_error_pixels = 0.0
    with torch.no_grad():
        for start in range(0, len(observation), batch_size):
            obs = observation[start : start + batch_size]
            target = action[start : start + batch_size]
            prediction = model(obs)
            squared_error += torch.square(prediction - target).sum().item()
            absolute_error_pixels += (prediction - target).abs().sum().item() * 256.0
    elements = len(observation) * action.shape[-1]
    return squared_error / elements, absolute_error_pixels / elements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/pusht/lerobot/file-000.parquet"),
    )
    parser.add_argument("--obs-mode", choices=["agent_pos", "full_state"], default="agent_pos")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("learning/runs/bc_pusht_seed42.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("learning/runs/bc_pusht_seed42.json"))
    args = parser.parse_args()

    data_path = args.data.expanduser()
    if not data_path.exists():
        raise FileNotFoundError(
            f"PushT parquet not found: {data_path}. Pass its path with --data."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observation, action, episode = load_data(data_path, args.obs_mode)
    train_mask, validation_mask, train_ids, validation_ids = split_episodes(
        episode, args.seed, args.validation_ratio
    )
    train_obs = torch.from_numpy(normalize_observations(observation[train_mask], args.obs_mode)).to(device)
    train_action = torch.from_numpy(normalize_actions(action[train_mask])).to(device)
    validation_obs = torch.from_numpy(normalize_observations(observation[validation_mask], args.obs_mode)).to(device)
    validation_action = torch.from_numpy(normalize_actions(action[validation_mask])).to(device)

    mean_action = train_action.mean(dim=0, keepdim=True)
    baseline_mse = torch.square(validation_action - mean_action).mean().item()
    baseline_mae_pixels = (validation_action - mean_action).abs().mean().item() * 256.0

    model = BehaviorCloningMLP(input_dim=train_obs.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_validation_mse = float("inf")
    best_model_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_obs), device=device)
        total_squared_error = 0.0
        for indices in order.split(args.batch_size):
            prediction = model(train_obs[indices])
            loss = torch.square(prediction - train_action[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_squared_error += loss.item() * len(indices) * train_action.shape[-1]

        train_mse = total_squared_error / train_action.numel()
        validation_mse, validation_mae_pixels = evaluate(
            model, validation_obs, validation_action, args.batch_size
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        row = {
            "epoch": epoch,
            "train_mse_normalized": train_mse,
            "validation_mse_normalized": validation_mse,
            "validation_mae_pixels": validation_mae_pixels,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"train_mse={train_mse:.5f}  "
            f"val_mse={validation_mse:.5f}  "
            f"val_mae={validation_mae_pixels:.2f}px"
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "best_validation_mse_normalized": best_validation_mse,
            "train_action_mean": train_action.mean(dim=0).cpu().tolist(),
            "obs_mode": args.obs_mode,
            "input_dim": int(train_obs.shape[-1]),
            "seed": args.seed,
            "episode_split": {"train": train_ids, "validation": validation_ids},
        },
        args.checkpoint,
    )
    metrics = {
        "data": str(data_path),
        "frames": len(observation),
        "episodes": int(len(np.unique(episode))),
        "train_frames": int(train_mask.sum()),
        "validation_frames": int(validation_mask.sum()),
        "seed": args.seed,
        "episode_split": {"train": train_ids, "validation": validation_ids},
        "device": str(device),
        "model": f"{train_obs.shape[-1]} -> 128 -> 128 -> 2 MLP; ReLU; tanh output",
        "obs_mode": args.obs_mode,
        "input": "agent_pos or full_state; positions scaled from [0, 512] to [-1, 1]; angle scaled from [0, 2pi] to [-1, 1]",
        "target": "demonstrated absolute cursor target, scaled from [0, 512] to [-1, 1]",
        "best_epoch": min(history, key=lambda row: row["validation_mse_normalized"])["epoch"],
        "best_validation_mse_normalized": best_validation_mse,
        "mean_action_baseline": {
            "validation_mse_normalized": baseline_mse,
            "validation_mae_pixels": baseline_mae_pixels,
        },
        "history": history,
        "checkpoint": str(args.checkpoint),
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"mean-action baseline: val_mse={baseline_mse:.5f}, val_mae={baseline_mae_pixels:.2f}px")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved metrics: {args.metrics}")


if __name__ == "__main__":
    main()
