#!/usr/bin/env bash
# ============================================================
# 标准 ACT 的三次评估（同一份权重，只改推理设置）
# 每次 50 个 episode，单次约 41 秒 ~ 1 分钟
# ------------------------------------------------------------
# 目的：把「时间集成」这一项单独拎出来看
#   A. 不开时间集成，n_action_steps=1   → 每步重新预测（单步重规划）
#   B. 开时间集成，  n_action_steps=1   → 论文的方法（每步查询 + 重叠加权平均）
#   C. 不开时间集成，n_action_steps=20  → 上一轮实验表现最好的设置
# ============================================================
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH="$HOME/miniforge3/envs/ffmpeg5/lib:${LD_LIBRARY_PATH:-}"

cd "$HOME/lerobot_act"

MODEL=outputs/train/act_pusht_std/checkpoints/last/pretrained_model

run () {
  local tag="$1"; shift
  echo "════════════════════════════════════════════"
  echo "  $tag"
  echo "════════════════════════════════════════════"
  .venv/bin/lerobot-eval \
    --policy.path="$MODEL" \
    --env.type=pusht \
    --policy.device=cuda \
    --eval.n_episodes=50 \
    --eval.batch_size=50 \
    "$@"
  echo
}

run "A. 不开时间集成 · n_action_steps=1"  --policy.n_action_steps=1
run "B. 开时间集成 coeff=0.01 · n_action_steps=1" --policy.n_action_steps=1 --policy.temporal_ensemble_coeff=0.01
run "C. 不开时间集成 · n_action_steps=20" --policy.n_action_steps=20

echo "════ 三次评估全部完成 ════"
