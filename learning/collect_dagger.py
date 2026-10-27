"""Interactive DAgger collection for the visual PushT policy.

Move the mouse to the expert target, then press Space to label it while the
policy acts, or E to label it and execute the mouse target directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import pygame
import torch

from visual_proprio_action_chunking_transformer import VisualProprioTransformer


def load_policy(checkpoint: Path, device: torch.device) -> tuple[VisualProprioTransformer, str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    target_mode = payload.get("target_mode", "absolute")
    model = VisualProprioTransformer(
        int(payload["chunk_size"]), int(payload["d_model"]), int(payload["layers"]), target_mode
    )
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), target_mode


def policy_action(model: VisualProprioTransformer, observation: dict,
                  target_mode: str, device: torch.device) -> np.ndarray:
    image = torch.from_numpy(observation["pixels"].astype(np.float32) / 255.0)
    image = image.permute(2, 0, 1).unsqueeze(0).to(device)
    position_np = np.asarray(observation["agent_pos"], dtype=np.float32)
    position = torch.from_numpy(position_np / 256.0 - 1.0).unsqueeze(0).to(device)
    with torch.no_grad():
        normalized = model(image, position)[0, 0].cpu().numpy()
    if target_mode == "relative":
        return np.clip(normalized * 256.0 + position_np, 0.0, 512.0).astype(np.float32)
    return np.clip((normalized + 1.0) * 256.0, 0.0, 512.0).astype(np.float32)


def observation_image(env) -> np.ndarray:
    """Render the current state at the 96x96 resolution used during training."""
    screen = env.unwrapped._draw()
    return env.unwrapped._get_img(
        screen, width=96, height=96, render_action=False
    ).copy()


def draw_overlay(env, policy_target: np.ndarray, expert_target: np.ndarray | None,
                 step: int, episode: int) -> None:
    env.render()
    window = env.unwrapped.window
    if window is None:
        return
    pygame.draw.line(window, (255, 60, 60), (int(policy_target[0]), int(policy_target[1])),
                     (int(policy_target[0]), int(policy_target[1])), width=1)
    pygame.draw.line(window, (255, 60, 60),
                     (int(policy_target[0] - 8), int(policy_target[1])),
                     (int(policy_target[0] + 8), int(policy_target[1])), width=3)
    pygame.draw.line(window, (255, 60, 60),
                     (int(policy_target[0]), int(policy_target[1] - 8)),
                     (int(policy_target[0]), int(policy_target[1] + 8)), width=3)
    if expert_target is not None:
        pygame.draw.circle(window, (255, 180, 0),
                           (int(expert_target[0]), int(expert_target[1])), 8, width=3)
    pygame.display.set_caption(
        f"DAgger episode={episode} step={step} | mouse=expert target | "
        "SPACE label+policy | E label+expert | N next | R restart | Q save+quit"
    )
    pygame.display.flip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("learning/runs/visual_proprio_transformer_short_full_data_seed42.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("learning/runs/dagger_interactive_seed42.npz"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--start-seed", type=int, default=200000)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("episodes 和 max_steps 必须为正数")
    if args.output.exists():
        raise FileExistsError(f"输出文件已存在，请换一个名称以保留已有数据: {args.output}")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target_mode = load_policy(args.checkpoint, device)
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="human")
    clock = pygame.time.Clock()
    images, positions, velocities = [], [], []
    expert_actions, policy_actions = [], []
    episode_ids, step_ids = [], []
    executed_actions, execution_modes = [], []
    quit_requested = False
    episode = 0
    trajectory_id = 0
    try:
        while episode < args.episodes:
            observation, _ = env.reset(seed=args.start_seed + episode)
            observation["pixels"] = observation_image(env)
            previous_position = np.asarray(observation["agent_pos"], dtype=np.float32).copy()
            restart = False
            next_episode = False
            print(f"episode {episode + 1}/{args.episodes}, seed={args.start_seed + episode}; "
                  "SPACE=label+policy, E=label+mouse, N=next, R=restart, Q=save+quit",
                  flush=True)
            for step in range(args.max_steps):
                policy_target = policy_action(model, observation, target_mode, device)
                draw_overlay(env, policy_target, None, step, episode)
                expert_target = None
                action_to_execute = None
                execution_mode = None
                label = False
                while not label:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                            quit_requested = True
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                            restart = True
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                            next_episode = True
                        elif (event.type == pygame.KEYDOWN and
                              event.key in (pygame.K_SPACE, pygame.K_e)):
                            mouse_pos = pygame.mouse.get_pos()
                            if not (0 <= mouse_pos[0] < 512 and 0 <= mouse_pos[1] < 512):
                                continue
                            expert_target = np.asarray(mouse_pos, dtype=np.float32)
                            execution_mode = int(event.key == pygame.K_e)
                            action_to_execute = expert_target.copy() if execution_mode else policy_target
                            label = True
                            break
                    if quit_requested or restart or next_episode:
                        break
                    mouse = np.asarray(pygame.mouse.get_pos(), dtype=np.float32)
                    draw_overlay(env, policy_target, mouse, step, episode)
                    clock.tick(30)
                if quit_requested or restart or next_episode:
                    break

                current_position = np.asarray(observation["agent_pos"], dtype=np.float32)
                images.append(observation["pixels"].copy())
                positions.append(current_position.copy())
                velocities.append(current_position - previous_position)
                expert_actions.append(expert_target.copy())
                policy_actions.append(policy_target.copy())
                executed_actions.append(action_to_execute.copy())
                execution_modes.append(execution_mode)
                episode_ids.append(trajectory_id)
                step_ids.append(step)
                observation, _, terminated, truncated, _ = env.step(action_to_execute)
                observation["pixels"] = observation_image(env)
                previous_position = current_position.copy()
                if terminated or truncated:
                    break
            if quit_requested:
                break
            trajectory_id += 1
            if not restart:
                episode += 1
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, images=np.asarray(images, dtype=np.uint8),
        agent_pos=np.asarray(positions, dtype=np.float32),
        velocity=np.asarray(velocities, dtype=np.float32),
        expert_action=np.asarray(expert_actions, dtype=np.float32),
        policy_action=np.asarray(policy_actions, dtype=np.float32),
        executed_action=np.asarray(executed_actions, dtype=np.float32),
        execution_mode=np.asarray(execution_modes, dtype=np.int8),
        episode_id=np.asarray(episode_ids, dtype=np.int64),
        step_id=np.asarray(step_ids, dtype=np.int64),
        checkpoint=np.asarray(str(args.checkpoint)),
        start_seed=np.asarray(args.start_seed, dtype=np.int64),
    )
    print(f"saved {len(images)} DAgger labels to {args.output}")


if __name__ == "__main__":
    main()
