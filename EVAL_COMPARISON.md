# ACT vs Diffusion Policy：同一管道下的对比评估

**任务**：PushT（把 T 形块推进目标区域），图像输入
**日期**：2026-09-21
**目的**：在**同一个框架、同一套评估协议**下对比 ACT 与 Diffusion Policy，
并**先校准评估管道本身**，确保两个数字可比。

---

## 为什么需要这次重做

此前的对比存在一个根本问题：**两个数字来自不同的仿真器实现**。

| | 此前的 DP 结果 | 此前的 ACT 结果 |
|---|---|---|
| 代码 | 官方 `real-stanford/diffusion_policy` | LeRobot 0.4.4 |
| 仿真环境 | DP 自带的 `diffusion_policy/env/pusht`（`legacy_test=true`） | `gym-pusht` |
| 测试种子 | 100000 ~ 100049 | seed 1000 |
| 指标定义 | `clip(coverage/0.95, 0, 1)` 的均值 | 原始重叠率 |

不同的物理引擎实现、不同的初始条件分布、不同的指标定义 —— **这两个数字放在一张表里
比较是不成立的**。

本次改为：**两个方法都在 LeRobot 里、用同一份数据、同一个评估脚本、同样的
`n_episodes=50` 评估。**

---

## 环境与协议

完整版本快照见 [`evidence/environment_versions.json`](evidence/environment_versions.json)。

```
Python 3.10.12
lerobot    0.4.4        （与训练 ACT 时一致，保证模型可加载）
torch      2.10.0+cu128
gym-pusht  0.1.6
pymunk     6.11.1       ← 必须 < 7（7.0 移除了 add_collision_handler）
gymnasium  1.3.0
GPU        RTX 5060 Laptop (8 GB)
```

评估命令（两个模型完全一致，只换 `--policy.path`）：

```bash
lerobot-eval \
  --policy.path=<模型目录> \
  --env.type=pusht \
  --eval.n_episodes=50 \
  --eval.batch_size=50 \
  --policy.device=cuda
```

---

## 第一步：校准评估管道

**做法**：用官方发布的、在 LeRobot 里训练好的 Diffusion Policy 权重
（`lerobot/diffusion_pusht`）跑评估。如果结果能对上官方公布的数字，
说明评估管道可信。

> 该权重是旧格式，需先迁移：
> `python -m lerobot.processor.migrate_policy_normalization --pretrained-path <src> --output-dir <dst>`

| 指标 | 官方公布（500 episode） | **本次复现（50 episode）** |
|---|---|---|
| 平均最大重叠率 | 0.955 | **0.9762** |
| 成功率 | 65.4% | **72.0%**（36/50） |
| 95% 置信区间 | — | [59.6%, 84.4%] |

**结论：管道可信。** 50 个 episode 的标准误约 6.4%，72.0% 的 95% 置信区间
覆盖官方公布的 65.4%。

原始数据：[`evidence/eval_info_diffusion_pusht.json`](evidence/eval_info_diffusion_pusht.json)

---

## 第二步：对比

| 模型 | 成功率 | 95% 置信区间 | 平均最大重叠率 |
|---|---|---|---|
| **官方 Diffusion Policy** | **36/50 = 72.0%** | [59.6%, 84.4%] | 0.9762 |
| **本仓库 ACT**（`act_pusht_v5`，50,000 步训练） | **5/50 = 10.0%** | [1.7%, 18.3%] | 0.5645 |

**统计检验**：Fisher 精确检验，双尾 **p = 2.1 × 10⁻¹⁰**

- 绝对差距 **62.0 个百分点**
- 倍数 **7.2 倍**
- 两者的 95% 置信区间**完全不重叠**

**结论：在同一评估协议下，ACT 的表现显著差于 Diffusion Policy，
差异远超统计噪声。**

原始数据：
[`eval_info_diffusion_pusht.json`](evidence/eval_info_diffusion_pusht.json) ·
[`eval_info_act_v5_nas10.json`](evidence/eval_info_act_v5_nas10.json)

---

## 第三步：`n_action_steps` 消融

这是本次最有价值的发现。ACT 的 `chunk_size=100`（一次预测 100 步动作），
但 `n_action_steps`（实际执行多少步后重新预测，即**重规划间隔**）
决定了闭环程度：

- `n_action_steps = chunk_size` → 执行完整段才重规划 = **完全开环**
- `n_action_steps` 小 → 频繁重规划 = 更闭环，但动作可能抖动

扫描结果（每次 50 个 episode，`chunk_size` 固定 100）：

| `n_action_steps` | 成功率 | 平均最大重叠率 | 说明 |
|---:|---:|---:|---|
| 1 | 0.0% | 0.063 | 每步重规划，动作抖动到无法完成任务 |
| 2 | 0.0% | 0.083 | 同上 |
| 5 | 0.0% | 0.219 | 同上，但接近可用 |
| **20** | **16.0%** | **0.698** | ⭐ **最优** |
| 10 | 10.0% | 0.564 | |
| 50 | 2.0% | 0.547 | 开始开环，失去纠错能力 |
| 100 | 2.0% | 0.354 | 完全开环（= ACT 官方默认行为） |

**曲线是单峰的，峰值在 `n_action_steps = 20`（16.0%）。**

两点值得注意：

1. **`n_action_steps = 100`（ACT 官方默认）在 PushT 上只有 2.0%。**
   ACT 的默认 `chunk_size=100` 是为 ALOHA 双臂任务（控制频率 50 Hz）调的，
   100 步 = 2 秒；而 PushT 是 10 Hz，100 步 = **10 秒**，几乎覆盖
   整个 episode（上限 300 步 = 30 秒）。**同一套超参数换到不同任务上会失效。**
2. **最优的 16.0% 仍然远低于 DP 的 72.0%。** 所以调整
   `n_action_steps` 能改善，但**不足以弥合差距**——差距主要来自
   训练不足（50,000 步 vs LeRobot 官方 DP 的 175,000 步）与配置未对齐。

原始数据：[`evidence/n_action_steps_sweep.json`](evidence/n_action_steps_sweep.json)

---

## 结论

1. **评估管道已验证**：官方 DP 权重在本管道下复现出 72.0%，
   与官方公布的 65.4% 在统计上一致。
2. **在同一协议下，ACT（10.0%）显著差于 DP（72.0%）**，
   Fisher 精确检验 p = 2.1 × 10⁻¹⁰。
3. **`n_action_steps` 是最敏感的超参数**，最优值 20 给出 16.0%，
   而 ACT 官方默认的 100 在 PushT 上只有 2.0%。
4. **但仅调这一个参数不足以弥合差距**，ACT 侧仍需对齐训练预算与配置。

---

## 诚实的局限

本次仍然**不是**一个完备的方法对比，以下问题尚未解决：

| 问题 | 说明 |
|---|---|
| **训练预算不对等** | ACT 训了 50,000 步（batch 32）；LeRobot 官方 DP 用了 200,000 步（batch 64，取 175k 存档）。ACT 少了约 4 倍 |
| **超参数未调优** | ACT 用的 `chunk_size=100` / `dim_model=512` / `dim_feedforward=3200` / `kl_weight=10` 是**照搬 ALOHA 任务**的官方默认值，未针对 PushT 调过 |
| **单种子** | ACT 只训了 1 个种子，DP 用的是官方发布的权重（也是单次训练） |
| **仅 50 个 episode** | 标准误约 6~7 个百分点。要更精确需要 500 个 episode |
| **`n_action_steps` 未取更细的网格** | 峰值在 20 附近，但 15 / 25 / 30 未测 |

**因此本报告的结论应表述为**：
「在**当前配置**下，ACT 显著差于 DP」，
**而不是**「ACT 方法本身差于 DP」。

要得到后一个结论，需要先让 ACT 侧的配置与训练预算对齐。

---

## 复现方式

```bash
# 1. 建环境
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install lerobot==0.4.4 "pymunk<7" gym-pusht

# 2. 下载官方 DP 权重并迁移格式
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download lerobot/diffusion_pusht --local-dir pretrained/diffusion_pusht
.venv/bin/python3 -m lerobot.processor.migrate_policy_normalization \
    --pretrained-path pretrained/diffusion_pusht \
    --output-dir pretrained/diffusion_pusht_migrated

# 3. 评估（两个模型只换 --policy.path）
.venv/bin/lerobot-eval --policy.path=pretrained/diffusion_pusht_migrated \
    --env.type=pusht --eval.n_episodes=50 --eval.batch_size=50 --policy.device=cuda

.venv/bin/lerobot-eval --policy.path=outputs/train/act_pusht_v5/checkpoints/050000/pretrained_model \
    --env.type=pusht --eval.n_episodes=50 --eval.batch_size=50 --policy.device=cuda

# 4. n_action_steps 扫描（只改这一个参数）
for nas in 1 2 5 10 20 50 100; do
  .venv/bin/lerobot-eval --policy.path=outputs/train/act_pusht_v5/checkpoints/050000/pretrained_model \
    --env.type=pusht --policy.n_action_steps=$nas \
    --eval.n_episodes=50 --eval.batch_size=50 --policy.device=cuda
done
```

完整环境版本见 [`evidence/environment_versions.json`](evidence/environment_versions.json)。

---

## 参考

- **Diffusion Policy**: Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- **ACT**: Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- **注意**：这两篇是**同期工作（concurrent work）**，作者无交集，DP 论文**没有**把 ACT 作为基线。本文的对比是自主设计的，不是复现论文中的对比。
- 官方 DP 权重：<https://huggingface.co/lerobot/diffusion_pusht>
