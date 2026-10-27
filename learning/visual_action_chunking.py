"""Small visual CNN action chunker for the PushT learning sequence."""

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
from torch.utils.data import DataLoader, Dataset


def normalize_action(action: np.ndarray) -> np.ndarray:
    return action.astype(np.float32) / 256.0 - 1.0


def denormalize_action(action: np.ndarray) -> np.ndarray:
    return (action + 1.0) * 256.0


class PushTImageChunks(Dataset):
    def __init__(
        self, data_path: Path, chunk_size: int, episode_ids: list[int],
        max_samples: int | None = None, sample_strategy: str = "first",
        image_cache: np.ndarray | None = None, action_cache: np.ndarray | None = None,
    ) -> None:
        root = zarr.open(str(data_path), mode="r")
        self.images = image_cache if image_cache is not None else root["data"]["img"]
        self.actions = action_cache if action_cache is not None else root["data"]["action"]
        ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
        starts = np.r_[0, ends[:-1]]
        self.indices = np.concatenate(
            [np.arange(starts[e], ends[e] - chunk_size + 1) for e in episode_ids]
        ).astype(np.int64)
        if max_samples is not None and max_samples < len(self.indices):
            if sample_strategy == "spread":
                positions = np.linspace(0, len(self.indices) - 1, max_samples, dtype=np.int64)
                self.indices = self.indices[positions]
            elif sample_strategy == "first":
                self.indices = self.indices[:max_samples]
            else:
                raise ValueError(f"未知 sample_strategy: {sample_strategy}")
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        image = np.asarray(self.images[index], dtype=np.float32) / 255.0
        action = np.asarray(self.actions[index : index + self.chunk_size], dtype=np.float32)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        action_tensor = torch.from_numpy(normalize_action(action))
        return image_tensor, action_tensor


def preload_visual_arrays(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open(str(data_path), mode="r")
    image_store = root["data"]["img"]
    image_cache = np.empty(image_store.shape, dtype=np.uint8)
    chunk_size = image_store.chunks[0]
    print(f"Caching {len(image_store)} images as uint8 (~{image_cache.nbytes / 1e9:.2f} GB)", flush=True)
    for start in range(0, len(image_store), chunk_size):
        end = min(start + chunk_size, len(image_store))
        block = np.asarray(image_store[start:end])
        rounded = np.rint(block)
        if not np.array_equal(block, rounded):
            raise ValueError(f"Image pixels are not integer-valued in frames {start}:{end}")
        image_cache[start:end] = np.clip(rounded, 0, 255).astype(np.uint8)
        chunk_index = start // chunk_size + 1
        total_chunks = (len(image_store) + chunk_size - 1) // chunk_size
        if chunk_index % max(1, total_chunks // 10) == 0 or end == len(image_store):
            print(f"  cached frames {end}/{len(image_store)}", flush=True)
    action_cache = np.asarray(root["data"]["action"], dtype=np.float32)
    return image_cache, action_cache


class VisualActionChunkCNN(nn.Module):
    def __init__(self, chunk_size: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 2 * chunk_size), nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(image)
        return self.head(encoded).reshape(-1, self.chunk_size, 2)


def split_episodes(data_path: Path, seed: int, validation_ratio: float) -> tuple[list[int], list[int]]:
    root = zarr.open(str(data_path), mode="r")
    episode_ids = np.arange(len(root["meta"]["episode_ends"]))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episode_ids)
    n_val = max(1, int(round(len(episode_ids) * validation_ratio)))
    return np.sort(shuffled[n_val:]).tolist(), np.sort(shuffled[:n_val]).tolist()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    elements = 0
    with torch.no_grad():
        for image, target in loader:
            prediction = model(image.to(device))
            total += torch.square(prediction.cpu() - target).sum().item()
            elements += target.numel()
    return total / elements


def rollout(
    model: nn.Module,
    chunk_size: int,
    n_action_steps: int,
    episodes: int,
    start_seed: int,
    max_steps: int,
    device: torch.device,
) -> dict:
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")
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
                image = torch.from_numpy(observation.astype(np.float32) / 255.0).permute(2, 0, 1)
                with torch.no_grad():
                    predicted = denormalize_action(model(image.unsqueeze(0).to(device))[0].cpu().numpy())
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
    return {"episodes": episodes, "successes": successes, "success_rate": successes / episodes,
            "mean_max_reward": float(np.mean([row["max_reward"] for row in rows])),
            "mean_max_coverage": float(np.mean([row["max_coverage"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])), "episodes_raw": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/pusht/pusht_cchi_v7_replay.zarr"))
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="限制每个 split 的样本数，便于 CPU 快速验证")
    parser.add_argument("--sample-strategy", choices=["first", "spread"], default="first")
    parser.add_argument("--checkpoint", type=Path, default=Path("learning/runs/visual_action_chunk_seed42.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("learning/runs/visual_action_chunk_seed42.json"))
    args = parser.parse_args()
    if not 1 <= args.n_action_steps <= args.chunk_size:
        raise ValueError("需要满足 1 <= n_action_steps <= chunk_size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, val_ids = split_episodes(args.data, args.seed, args.validation_ratio)
    image_cache, action_cache = preload_visual_arrays(args.data)
    train_set = PushTImageChunks(args.data, args.chunk_size, train_ids, args.max_samples,
                                 args.sample_strategy, image_cache, action_cache)
    val_set = PushTImageChunks(args.data, args.chunk_size, val_ids, args.max_samples,
                               args.sample_strategy, image_cache, action_cache)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = VisualActionChunkCNN(args.chunk_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_mse = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        samples = 0
        for image, target in train_loader:
            prediction = model(image.to(device))
            loss = torch.square(prediction - target.to(device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(image)
            samples += len(image)
        train_mse = total / samples
        val_mse = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse})
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:03d}/{args.epochs} train_mse={train_mse:.5f} val_mse={val_mse:.5f} device={device}")
    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "chunk_size": args.chunk_size, "seed": args.seed,
                "episode_split": {"train": train_ids, "validation": val_ids}}, args.checkpoint)
    result = {"data": str(args.data), "model": "CNN visual action chunker", "chunk_size": args.chunk_size,
              "n_action_steps": args.n_action_steps, "epochs": args.epochs, "seed": args.seed,
              "device": str(device), "train_samples": len(train_set), "validation_samples": len(val_set),
              "max_samples": args.max_samples, "sample_strategy": args.sample_strategy,
              "best_validation_mse": best_mse, "history": history,
              "episode_split": {"train": train_ids, "validation": val_ids},
              "rollout": rollout(model, args.chunk_size, args.n_action_steps, args.episodes,
                                  100000, args.max_steps, device), "checkpoint": str(args.checkpoint)}
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"rollout success={result['rollout']['successes']}/{args.episodes} "
          f"mean_max_coverage={result['rollout']['mean_max_coverage']:.4f}")
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
