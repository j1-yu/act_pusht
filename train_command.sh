# ACT 训练命令（PushT 任务，最终版）
# 环境：Ubuntu 22.04 + Python 3.10 + PyTorch 2.8.0 + CUDA 12.8

export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=~/miniforge3/envs/ffmpeg5/lib:$LD_LIBRARY_PATH

lerobot-train \
  --policy.type=act \
  --policy.chunk_size=100 \
  --policy.n_action_steps=10 \
  --dataset.repo_id=lerobot/pusht \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.video_backend=torchcodec \
  --output_dir=outputs/train/act_pusht_v5 \
  --job_name=act_pusht_v5 \
  --wandb.enable=false \
  --batch_size=32 \
  --steps=50000 \
  --num_workers=1

# 关键依赖版本：
# torchcodec==0.7.0 (匹配 torch 2.8.0)
# FFmpeg 5 (conda: conda create -n ffmpeg5 -c conda-forge ffmpeg=5.1.2)
# 视频需重编码为 H.264 + 关键帧间隔10: ffmpeg -i in.mp4 -c:v libx264 -g 10 out.mp4
