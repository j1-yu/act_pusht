"""Fine-tune a one-step visual policy with demonstrations and DAgger labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from visual_action_chunking import preload_visual_arrays, split_episodes
from visual_proprio_action_chunking import PushTImageAgentPosChunks
from visual_proprio_action_chunking_transformer import VisualProprioTransformer, rollout


class DaggerImageActions(Dataset):
    def __init__(self, path: Path) -> None:
        self.data = np.load(path)
        required = {"images", "agent_pos", "expert_action"}
        missing = required - set(self.data.files)
        if missing:
            raise ValueError(f"DAgger 数据缺少字段: {sorted(missing)}")
        self.images = self.data["images"]
        self.positions = self.data["agent_pos"]
        self.actions = self.data["expert_action"]
        if len(self.images) == 0:
            raise ValueError("DAgger 数据没有标签；先运行 collect_dagger.py 采集")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        image = torch.from_numpy(self.images[index].astype(np.float32) / 255.0).permute(2, 0, 1)
        position = torch.from_numpy(self.positions[index].astype(np.float32) / 256.0 - 1.0)
        action = torch.from_numpy(self.actions[index].astype(np.float32) / 256.0 - 1.0)
        return image, position, action.unsqueeze(0)


class DemonstrationOneStep(Dataset):
    def __init__(self, base: PushTImageAgentPosChunks) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        image, position, action = self.base[index]
        return image, position, action[:1]


def batch_one_step(batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    images, positions, actions = zip(*batch)
    return torch.stack(images), torch.stack(positions), torch.stack(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/pusht/pusht_cchi_v7_replay.zarr"))
    parser.add_argument("--dagger-data", type=Path, default=Path("learning/runs/dagger_interactive_seed42.npz"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("learning/runs/visual_proprio_transformer_short_full_data_seed42.pt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("learning/runs/visual_proprio_transformer_dagger_seed42.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("learning/runs/visual_proprio_transformer_dagger_seed42.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dagger-ratio", type=float, default=0.25)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.dagger_ratio < 1.0:
        raise ValueError("dagger-ratio 必须在 0 和 1 之间")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    model = VisualProprioTransformer(
        int(payload["chunk_size"]), int(payload["d_model"]), int(payload["layers"]), "absolute"
    )
    model.load_state_dict(payload["model_state_dict"])
    model.chunk_size = 1
    model.action_queries = torch.nn.Parameter(model.action_queries.detach()[:, :1].clone())
    model.to(device)

    train_ids, val_ids = split_episodes(args.data, args.seed, 0.2)
    image_cache, action_cache = preload_visual_arrays(args.data)
    root = zarr.open(str(args.data), mode="r")
    state_cache = np.asarray(root["data"]["state"], dtype=np.float32)
    base_train = PushTImageAgentPosChunks(
        args.data, 1, train_ids, None, "spread", image_cache, action_cache,
        state_cache=state_cache, target_mode="absolute",
    )
    base_val = PushTImageAgentPosChunks(
        args.data, 1, val_ids, None, "spread", image_cache, action_cache,
        state_cache=state_cache, target_mode="absolute",
    )
    base_train = DemonstrationOneStep(base_train)
    dagger = DaggerImageActions(args.dagger_data)
    combined = ConcatDataset([base_train, dagger])
    weights = torch.cat((
        torch.full((len(base_train),), (1.0 - args.dagger_ratio) / len(base_train)),
        torch.full((len(dagger),), args.dagger_ratio / len(dagger)),
    ))
    sampler = WeightedRandomSampler(weights, num_samples=len(base_train), replacement=True)
    loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=batch_one_step, num_workers=0)
    val_loader = DataLoader(base_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history = []
    best_mse = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        samples = 0
        for image, position, action in loader:
            prediction = model(image.to(device), position.to(device))
            loss = torch.square(prediction - action.to(device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(image)
            samples += len(image)
        model.eval()
        val_total = 0.0
        elements = 0
        with torch.no_grad():
            for image, position, action in val_loader:
                prediction = model(image.to(device), position.to(device))
                val_total += torch.square(prediction.cpu() - action).sum().item()
                elements += action.numel()
        val_mse = val_total / elements
        history.append({"epoch": epoch, "train_mse": total / samples, "validation_mse": val_mse})
        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"epoch {epoch:03d}/{args.epochs} train_mse={total / samples:.5f} "
              f"val_mse={val_mse:.5f} device={device}", flush=True)

    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "chunk_size": 1, "d_model": payload["d_model"],
                "layers": payload["layers"], "target_mode": "absolute", "seed": args.seed,
                "base_checkpoint": str(args.base_checkpoint), "dagger_data": str(args.dagger_data)}, args.checkpoint)
    rollout_result = rollout(model, 1, 1, args.episodes, 100000, 300, device, "absolute")
    result = {"base_checkpoint": str(args.base_checkpoint), "dagger_data": str(args.dagger_data),
              "dagger_samples": len(dagger), "dagger_ratio": args.dagger_ratio,
              "train_samples_per_epoch": len(base_train), "validation_samples": len(base_val),
              "epochs": args.epochs, "seed": args.seed, "device": str(device),
              "best_validation_mse": best_mse,
              "best_epoch": min(history, key=lambda row: row["validation_mse"])["epoch"],
              "history": history, "rollout": rollout_result, "checkpoint": str(args.checkpoint)}
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"rollout success={rollout_result['successes']}/{args.episodes} "
          f"coverage={rollout_result['mean_max_coverage']:.4f}", flush=True)
    print(f"saved checkpoint: {args.checkpoint}", flush=True)
    print(f"saved metrics: {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
