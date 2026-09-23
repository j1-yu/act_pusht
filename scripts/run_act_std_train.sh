#!/usr/bin/env bash
# ============================================================
# 标准 ACT 在 PushT 上重训（严格对齐论文 Table III）
# ------------------------------------------------------------
# 相对上一轮（act_pusht_v5）只改两处「不对等」：
#   ① n_decoder_layers  1 → 7   （论文 Table III = 7；这是唯一需要改的架构项）
#   ② batch_size       32 → 8   （论文 Table III = 8；batch 和 lr 要配对使用）
# 另外 n_action_steps 100 → 1，是为了让「时间集成」能在评估时打开
#   （LeRobot 要求：开时间集成时 n_action_steps 必须 = 1）
#   注意：n_action_steps 不参与训练损失，只影响推理
#
# 其余项本来就与论文一致，无需显式指定：
#   n_encoder_layers 4 / dim_model 512 / n_heads 8 / dim_feedforward 3200
#   chunk_size 100 / kl_weight(β) 10 / dropout 0.1 / vision_backbone resnet18
#   optimizer_lr 1e-5 / use_vae True
#
# 步数选择：200000 步 × batch 8 = 1.6M 样本
#   = 上一轮 50000 步 × batch 32 = 1.6M 样本
#   → 让「见过的数据量」相同，把差异尽量归因到架构上
# ============================================================
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH="$HOME/miniforge3/envs/ffmpeg5/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/lerobot_act"

exec systemd-inhibit --what=idle:sleep --mode=block --why="ACT 训练中，禁止休眠" \
  .venv/bin/lerobot-train \
    --policy.type=act \
    --policy.n_decoder_layers=7 \
    --policy.n_action_steps=1 \
    --policy.chunk_size=100 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/pusht \
    --dataset.video_backend=torchcodec \
    --output_dir=outputs/train/act_pusht_std \
    --job_name=act_pusht_std \
    --wandb.enable=false \
    --batch_size=8 \
    --steps=200000 \
    --save_freq=50000 \
    --seed=1000 \
    --num_workers=4
