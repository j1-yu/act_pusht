# PushT 学习过程

这里记录从最小行为克隆基线到视觉 Action Chunking 和 DAgger 的学习过程。代码是教学和诊断用途，ACT 主实验仍使用 LeRobot 官方实现，不能把这些小模型的结果当作完整 ACT 的结果。

## 运行前准备

需要 Python 3.10、PyTorch、NumPy、zarr、PyArrow、Gymnasium、`gym-pusht`、pygame 和 Pillow，以及 PushT 数据集。视觉及后续 DAgger 实验还会用到 torchvision。BC 数据加载可以使用 LeRobot PushT Parquet；低维与视觉序列实验使用官方 PushT Zarr 数据。默认数据路径是：

```text
data/pusht/pusht_cchi_v7_replay.zarr
```

也可以通过每个脚本的 `--data` 参数指定其他路径。BC 示例需要将 `--data` 指向 LeRobot PushT Parquet 文件；Zarr 示例使用上面的数据集路径。所有 checkpoint 和指标默认写入 `learning/runs/`，该目录被 `.gitignore` 排除；仓库只发布源代码和实验说明，不发布本地模型权重或数据集。

从仓库根目录运行示例：

```bash
python learning/bc_pusht.py --data data/pusht/lerobot/file-000.parquet --obs-mode agent_pos --epochs 25
python learning/bc_rollout.py --checkpoint learning/runs/bc_pusht_seed42.pt
```

## 学习路线

### 1. 行为克隆基线

`bc_pusht.py` 用当前末端位置或完整 5 维状态预测下一步绝对目标位置，按 episode 划分训练集和验证集，避免同一条轨迹泄漏到两侧。`bc_rollout.py` 在固定测试 seed 上做闭环评估，并与训练集平均动作的常数策略比较。

当前记录的结果：

| 输入 | 闭环成功率 | 平均最高 coverage |
|---|---:|---:|
| 末端位置（2 维） | 0/50 | 0.1356 |
| 完整状态（5 维） | 0/50 | 0.2237 |

离线动作误差下降并没有带来任务成功，说明缺少目标信息和闭环分布漂移是主要问题。

### 2. 低维 Action Chunking

`action_chunking.py` 一次预测连续 10 步动作；`action_chunking_transformer.py` 把 MLP 换成小型 Transformer decoder；`action_chunking_cvae.py` 进一步加入 posterior/prior latent。它们用于观察动作分块、网络结构和潜变量对闭环误差的影响。

低维实验中，Action Chunking 的平均最高 coverage 高于单步完整状态 BC，但当前配置仍没有稳定成功回合。CVAE 实验还显示，posterior 在训练时能看到真实动作块，而推理时只能使用 prior；如果两者没有对齐，验证重建误差不能代表闭环表现。

### 3. 视觉条件策略

`visual_proprio_action_chunking_transformer.py` 使用图像、末端位置和 Transformer action queries；`visual_rollout_diagnostic.py` 保存逐步 coverage、动作和关键帧，用于分析接触、动作限幅和失败原因。脚本保留了相对动作、两帧输入和初始状态审计等实验所需的命令行参数，默认路径均为仓库相对路径。

### 4. DAgger 入口

`collect_dagger.py` 提供鼠标专家与策略动作的交互式标注入口，`train_dagger_one_step.py` 用采集到的标签做单步微调。该阶段的结果仍是单种子探索，不能与 ACT 主实验直接比较；若继续扩展，应保留相同 horizon 的无新增数据对照。

## 复现建议

1. 先运行行为克隆基线，再运行 `bc_rollout.py`，确认离线误差和闭环表现的差异。
2. 保持 episode 划分、测试 seed 和成功定义不变，只改变一个模型变量。
3. 报告成功率时同时报告平均最高 coverage、测试 episode 数和 seed；单个 checkpoint 的 50 个 episode 只适合阶段性比较。
4. 训练日志和 checkpoint 请保存在本地 `learning/runs/`，不要提交到仓库。

## 与主实验的关系

主实验位于 [`learning/experiments/act_data_scale/`](experiments/act_data_scale/)，使用 LeRobot 官方 ACT，在固定场景上研究演示数据规模、动作重规划频率和视觉扰动。这里的学习脚本用于解释这些实验之前的建模问题和失败模式，不是主实验的实现替代品。
