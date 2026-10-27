"""A small Transformer action-chunking baseline for PushT full state."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from action_chunking import load_chunks, rollout


class TransformerActionChunk(nn.Module):
    def __init__(self, chunk_size: int, d_model: int = 128, layers: int = 2) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.state_projection = nn.Linear(5, d_model)
        self.query_tokens = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.action_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2), nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        memory = self.state_projection(state).unsqueeze(1)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)


def evaluate(model: nn.Module, state: torch.Tensor, chunks: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(state), batch_size):
            prediction = model(state[start : start + batch_size])
            total += torch.square(prediction - chunks[start : start + batch_size]).sum().item()
    return total / chunks.numel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/pusht/pusht_cchi_v7_replay.zarr"))
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path, default=Path("learning/runs/action_chunk_transformer_seed42.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("learning/runs/action_chunk_transformer_seed42.json"))
    args = parser.parse_args()
    if args.chunk_size < 1 or args.layers < 1 or not 1 <= args.n_action_steps <= args.chunk_size:
        raise ValueError("需要满足 chunk_size >= n_action_steps >= 1 且 layers >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train_state, train_chunks, val_state, val_chunks, train_ids, val_ids = load_chunks(
        args.data, args.chunk_size, args.seed, args.validation_ratio
    )
    train_state_t = torch.from_numpy(train_state)
    train_chunks_t = torch.from_numpy(train_chunks)
    val_state_t = torch.from_numpy(val_state)
    val_chunks_t = torch.from_numpy(val_chunks)

    model = TransformerActionChunk(args.chunk_size, layers=args.layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    torch.save({"model_state_dict": best_state, "chunk_size": args.chunk_size, "layers": args.layers,
                "seed": args.seed, "episode_split": {"train": train_ids, "validation": val_ids}}, args.checkpoint)
    rollout_result = rollout(model, args.chunk_size, args.n_action_steps, args.episodes, 100000, args.max_steps)
    result = {"data": str(args.data), "model": "Transformer decoder with learned action query tokens",
              "chunk_size": args.chunk_size, "layers": args.layers, "n_action_steps": args.n_action_steps,
              "seed": args.seed, "train_samples": len(train_state), "validation_samples": len(val_state),
              "best_validation_mse": best_mse,
              "best_epoch": min(history, key=lambda row: row["validation_mse"])["epoch"],
              "episode_split": {"train": train_ids, "validation": val_ids}, "history": history,
              "rollout": rollout_result, "checkpoint": str(args.checkpoint)}
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"rollout success={rollout_result['successes']}/{args.episodes} ({rollout_result['success_rate']:.1%}) mean_max_coverage={rollout_result['mean_max_coverage']:.4f}")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved metrics: {args.metrics}")


if __name__ == "__main__":
    main()
