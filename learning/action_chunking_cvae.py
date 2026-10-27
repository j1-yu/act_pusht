"""Transformer action chunking with a small conditional VAE for PushT."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from action_chunking import load_chunks, rollout


class CVAETransformerActionChunk(nn.Module):
    def __init__(self, chunk_size: int, latent_dim: int = 16, layers: int = 2) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.state_projection = nn.Linear(5, 128)
        self.condition_projection = nn.Linear(5 + latent_dim, 128)
        self.query_tokens = nn.Parameter(torch.randn(1, chunk_size, 128) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.action_head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 2), nn.Tanh())

        # Posterior q(z | state, demonstrated action chunk), used only during training.
        self.posterior = nn.Sequential(
            nn.Linear(5 + chunk_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.posterior_mu = nn.Linear(128, latent_dim)
        self.posterior_logvar = nn.Linear(128, latent_dim)

    def encode_posterior(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.posterior(torch.cat([state, action_chunk.flatten(start_dim=1)], dim=-1))
        return self.posterior_mu(hidden), self.posterior_logvar(hidden).clamp(-10.0, 10.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        condition = self.condition_projection(torch.cat([state, latent], dim=-1)).unsqueeze(1)
        queries = self.query_tokens.expand(state.shape[0], -1, -1)
        decoded = self.decoder(queries, condition)
        return self.action_head(decoded)

    def forward(
        self, state: torch.Tensor, action_chunk: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if action_chunk is None:
            latent = torch.zeros(
                state.shape[0], self.latent_dim, dtype=state.dtype, device=state.device
            )
            return self.decode(state, latent), None, None
        mu, logvar = self.encode_posterior(state, action_chunk)
        latent = self.reparameterize(mu, logvar)
        return self.decode(state, latent), mu, logvar


def evaluate(
    model: CVAETransformerActionChunk,
    state: torch.Tensor,
    chunks: torch.Tensor,
    batch_size: int,
) -> tuple[float, float, float]:
    model.eval()
    reconstruction = 0.0
    posterior_kl = 0.0
    with torch.no_grad():
        for start in range(0, len(state), batch_size):
            prediction, mu, logvar = model(state[start : start + batch_size], chunks[start : start + batch_size])
            target = chunks[start : start + batch_size]
            reconstruction += torch.square(prediction - target).sum().item()
            posterior_kl += (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum().item()
    elements = chunks.numel()
    samples = len(state)
    return reconstruction / elements, posterior_kl / samples, reconstruction / elements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/pusht/pusht_cchi_v7_replay.zarr"))
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path, default=Path("learning/runs/action_chunk_cvae_seed42.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("learning/runs/action_chunk_cvae_seed42.json"))
    args = parser.parse_args()
    if args.chunk_size < 1 or args.latent_dim < 1 or args.layers < 1 or not 1 <= args.n_action_steps <= args.chunk_size:
        raise ValueError("需要满足 chunk_size >= n_action_steps >= 1，且 latent_dim/layers >= 1")

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
    model = CVAETransformerActionChunk(args.chunk_size, args.latent_dim, args.layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_objective = float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_state_t))
        total_reconstruction = 0.0
        total_kl = 0.0
        for indices in order.split(args.batch_size):
            prediction, mu, logvar = model(train_state_t[indices], train_chunks_t[indices])
            reconstruction = torch.square(prediction - train_chunks_t[indices]).mean()
            kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).mean()
            loss = reconstruction + args.kl_weight * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_reconstruction += reconstruction.item() * len(indices)
            total_kl += kl.item() * len(indices)
        train_reconstruction = total_reconstruction / len(train_state_t)
        train_kl = total_kl / len(train_state_t)
        val_reconstruction, val_kl, _ = evaluate(model, val_state_t, val_chunks_t, args.batch_size)
        objective = val_reconstruction + args.kl_weight * val_kl
        history.append({"epoch": epoch, "train_reconstruction": train_reconstruction,
                        "train_kl": train_kl, "validation_reconstruction": val_reconstruction,
                        "validation_kl": val_kl, "validation_objective": objective})
        if objective < best_objective:
            best_objective = objective
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs} train_recon={train_reconstruction:.5f} "
                  f"val_recon={val_reconstruction:.5f} val_kl={val_kl:.4f}")

    model.load_state_dict(best_state)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "chunk_size": args.chunk_size,
                "latent_dim": args.latent_dim, "layers": args.layers,
                "kl_weight": args.kl_weight, "seed": args.seed,
                "episode_split": {"train": train_ids, "validation": val_ids}}, args.checkpoint)
    rollout_result = rollout(model, args.chunk_size, args.n_action_steps, args.episodes, 100000, args.max_steps)
    result = {"data": str(args.data), "model": "CVAE posterior + Transformer action decoder",
              "chunk_size": args.chunk_size, "latent_dim": args.latent_dim, "layers": args.layers,
              "kl_weight": args.kl_weight, "n_action_steps": args.n_action_steps, "seed": args.seed,
              "train_samples": len(train_state), "validation_samples": len(val_state),
              "best_validation_objective": best_objective,
              "best_epoch": min(history, key=lambda row: row["validation_objective"])["epoch"],
              "episode_split": {"train": train_ids, "validation": val_ids}, "history": history,
              "rollout": rollout_result, "checkpoint": str(args.checkpoint)}
    args.metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"rollout success={rollout_result['successes']}/{args.episodes} ({rollout_result['success_rate']:.1%}) "
          f"mean_max_coverage={rollout_result['mean_max_coverage']:.4f}")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved metrics: {args.metrics}")


if __name__ == "__main__":
    main()
