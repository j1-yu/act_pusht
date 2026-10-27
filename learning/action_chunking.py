"""Low-dimensional action chunking baseline for PushT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch
from torch import nn
import zarr


def normalize_state(state: np.ndarray) -> np.ndarray:
    value = state.astype(np.float32, copy=True)
    value[..., :4] = value[..., :4] / 256.0 - 1.0
    value[..., 4] = value[..., 4] / np.pi - 1.0
    return value


def normalize_action(action: np.ndarray) -> np.ndarray:
    return action.astype(np.float32) / 256.0 - 1.0


def denormalize_action(action: np.ndarray) -> np.ndarray:
    return (action + 1.0) * 256.0


class ActionChunkMLP(nn.Module):
    def __init__(self, chunk_size: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * chunk_size),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).reshape(-1, self.chunk_size, 2)


def load_chunks(
    data_path: Path, chunk_size: int, seed: int, validation_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    root = zarr.open(str(data_path), mode="r")
    states = np.asarray(root["data"]["state"], dtype=np.float32)
    actions = np.asarray(root["data"]["action"], dtype=np.float32)
    ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
    episode_ids = np.arange(len(ends))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episode_ids)
    n_val = max(1, int(round(len(episode_ids) * validation_ratio)))
    val_ids = np.sort(shuffled[:n_val])
    train_ids = np.sort(shuffled[n_val:])

    starts = np.r_[0, ends[:-1]]

    def make_samples(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.concatenate(
            [np.arange(starts[e], ends[e] - chunk_size + 1) for e in ids]
        )
        state = normalize_state(states[indices])
        chunks = np.stack(
            [normalize_action(actions[i : i + chunk_size]) for i in indices]
        )
        return state, chunks

    train_state, train_chunks = make_samples(train_ids)
    val_state, val_chunks = make_samples(val_ids)
    return train_state, train_chunks, val_state, val_chunks, train_ids.tolist(), val_ids.tolist()


def evaluate(
    model: ActionChunkMLP, state: torch.Tensor, chunks: torch.Tensor, batch_size: int
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(state), batch_size):
            prediction = model(state[start : start + batch_size])
            target = chunks[start : start + batch_size]
            total += torch.square(prediction - target).sum().item()
    return total / chunks.numel()


def rollout(
    model: ActionChunkMLP,
    chunk_size: int,
    n_action_steps: int,
    episodes: int,
    start_seed: int,
    max_steps: int,
) -> dict:
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    rows = []
    try:
        for index in range(episodes):
            observation, _ = env.reset(seed=start_seed + index)
            max_coverage = 0.0
            max_reward = 0.0
            terminated = False
            truncated = False
            steps = 0
            while steps < max_steps and not (terminated or truncated):
                state = torch.from_numpy(normalize_state(observation)).unsqueeze(0)
                with torch.no_grad():
                    model_output = model(state)
                    if isinstance(model_output, tuple):
                        model_output = model_output[0]
                    predicted = denormalize_action(model_output.squeeze(0).numpy())
                for action in predicted[:n_action_steps]:
                    observation, reward, terminated, truncated, info = env.step(
                        np.clip(action, 0.0, 512.0).astype(np.float32)
                    )
                    steps += 1
                    max_reward = max(max_reward, float(reward))
                    max_coverage = max(max_coverage, float(info.get("coverage", 0.0)))
                    if steps >= max_steps or terminated or truncated:
                        break
            rows.append(
                {
                    "seed": start_seed + index,
                    "steps": steps,
                    "max_reward": max_reward,
                    "max_coverage": max_coverage,
                    "success": bool(info.get("is_success", False)),
                }
            )
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
        "--data", type=Path, default=Path("data/pusht/pusht_cchi_v7_replay.zarr")
    )
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path, default=Path("learning/runs/action_chunk10_seed42.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("learning/runs/action_chunk10_seed42.json"))
    args = parser.parse_args()
    if args.chunk_size < 1 or args.n_action_steps < 1 or args.n_action_steps > args.chunk_size:
        raise ValueError("需要满足 1 <= n_action_steps <= chunk_size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    train_state, train_chunks, val_state, val_chunks, train_ids, val_ids = load_chunks(
        args.data, args.chunk_size, args.seed, args.validation_ratio
    )
    train_chunks_t = torch.from_numpy(train_chunks)
    train_state_t = torch.from_numpy(train_state)
    val_state_t = torch.from_numpy(val_state)
    val_chunks_t = torch.from_numpy(val_chunks)
    model = ActionChunkMLP(args.chunk_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_mse = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_state_t))
        total = 0.0
        for indices in order.split(args.batch_size):
            prediction = model(train_state_t[indices])
            loss = torch.square(prediction - train_chunks_t[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        train_mse = total / len(train_state_t)
        val_mse = evaluate(model, val_state_t, val_chunks_t, args.batch_size)
        history.append({"epoch": epoch, "train_mse": train_mse, "validation_mse": val_mse})
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs} train_mse={train_mse:.5f} val_mse={val_mse:.5f}")

    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": best_state, "chunk_size": args.chunk_size, "seed": args.seed,
         "episode_split": {"train": train_ids, "validation": val_ids}}, args.checkpoint
    )
    result = {
        "data": str(args.data), "chunk_size": args.chunk_size, "n_action_steps": args.n_action_steps,
        "epochs": args.epochs, "seed": args.seed, "train_samples": len(train_state),
        "validation_samples": len(val_state), "best_validation_mse": best_mse,
        "best_epoch": min(history, key=lambda row: row["validation_mse"])["epoch"],
        "episode_split": {"train": train_ids, "validation": val_ids}, "history": history,
        "checkpoint": str(args.checkpoint),
        "rollout": rollout(model, args.chunk_size, args.n_action_steps, args.episodes, 100000, args.max_steps),
    }
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"rollout success={result['rollout']['successes']}/{args.episodes} "
        f"({result['rollout']['success_rate']:.1%}) "
        f"mean_max_coverage={result['rollout']['mean_max_coverage']:.4f}"
    )
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved metrics: {args.metrics}")


if __name__ == "__main__":
    main()
