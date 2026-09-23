# ACT 复现：PushT 任务（LeRobot 框架）

复现 **ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**（Zhao et al., RSS 2023）中的 Action Chunking Transformer，在 PushT 任务上训练视觉运动策略。

本仓库基于 [LeRobot](https://github.com/huggingface/lerobot) 运行，**策略代码为官方实现，非从零复现**。目标是把标准 ACT 配置跑通，并把「哪些推理设置真正决定成败」这件事量化清楚。

## 演示视频

![ACT 推 T 块](demo_act.gif)

（模型在 PushT 仿真中自主推 T 块，成功 episode）

---

## 结果

**任务**：PushT（把 T 形块推到目标区域），image 输入版本
**模型**：LeRobot ACT，本仓库训练到 step 200000，**84.0M 参数**
**评估**：`lerobot-eval` + `--env.type=pusht`，50 episodes，seed=1000

### 主结果：同一份权重，只改推理设置

| 配置 | 成功率 | 95% CI (Wilson) | `avg_max_reward` | 评估耗时 |
|---|---:|---|---:|---:|
| **A** 不开时间集成，`n_action_steps=1` | 1/50 = **2.0%** | [0.3%, 10.5%] | 0.1999 | 48.5 s |
| **B** 开时间集成 `coeff=0.01`，`n_action_steps=1` | 0/50 = **0.0%** | [0.0%, 7.1%] | 0.3911 | 48.0 s |
| **C** 不开时间集成，`n_action_steps=20` | 26/50 = **52.0%** | [38.5%, 65.2%] | **0.9218** | **36.1 s** |

Fisher 精确检验（以 C 为基准，双尾）：

| 对比 | 成功率之差 | p 值 |
|---|---:|---:|
| C (52%) vs A (2%) | +50.0 pp | 6.5e-09 |
| C (52%) vs B (0%) | +52.0 pp | 3.5e-10 |
| C (52%) vs `coeff` 扫描各档 | +50 ~ 52 pp | 6.5e-09 ~ 3.5e-10 |

### 时间集成系数扫描（均 `n_action_steps=1`）

| `temporal_ensemble_coeff` | 成功率 | `avg_max_reward` |
|---|---:|---:|
| 0.01（B 组） | 0/50 = 0.0% | 0.3911 |
| 0.05 | 0/50 = 0.0% | 0.3238 |
| 0.1 | 0/50 = 0.0% | 0.2846 |
| 0.5 | 1/50 = 2.0% | 0.2972 |
| 1.0 | 0/50 = 0.0% | 0.2863 |
| —（**不开集成**，即 A 组） | 1/50 = 2.0% | 0.1999 |

### 三点结论

**1. 决定成败的是 `n_action_steps`，不是时间集成。**

`n_action_steps` 从 1 改成 20，成功率从 2% → 52%（+50 个百分点，p ≈ 6e-09）。而时间集成这个开关，无论系数取 0.01 还是 1.0，结果都在 0~2% 徘徊，**和不开集成（2%）没有区别**。

**2. 时间集成在本任务上完全无效，且不是「参数没调好」。**

扫描前曾假设「`coeff=0.01` 衰减太慢导致动作滞后」，扫完发现假设是错的：把系数一路加大到 1.0（几乎等于不用集成）依然是 0%。说明问题不在平均窗口长短，而在**「每隔一步就换一个新计划」这件事本身**。

模型每一步都重新预测一遍动作块，而两次预测并不完全一致，于是执行时动作来回微调、原地打转。**`n_action_steps=20` 相当于要求「规划一旦定下就执行 20 步」**，这个规划一致性才是 52% 的来源。

**3. C 组同时还是最快的。**

评估耗时 36.1 s，比 A/B 的约 48 s 少 25%，因为重规划次数少了 20 倍（`eval_ep_s` 0.72 s vs 0.97 s）。

> ⚠️ 反过来说：B 组的 `avg_max_reward`（0.3911）**高于** A 组（0.1999），说明时间集成**方向是对的、只是速度不够**——50 步一局跑完还没推到位。但成功率上它没有任何体现。

### checkpoint 对比：150K vs 200K（同配置，仅换权重）

上面 7 次用的都是 step 200000 的权重。为了回答「多训的 5 万步值不值」，又用 **step 150000** 的 checkpoint 跑了一次**完全相同的配置**（`n_action_steps=20`、不开时间集成）：

| checkpoint | 成功率 | 95% CI (Wilson) | `avg_max_reward` |
|---|---:|---|---:|
| step 150000 | 18/50 = **36.0%** | [24.1%, 49.9%] | 0.9184 |
| step 200000 | 26/50 = **52.0%** | [38.5%, 65.2%] | 0.9218 |

Fisher 精确检验：差 16.0 个百分点，**p = 0.158 → 不显著**。

**结论：本次实验未能检出后 5 万步带来的提升。** 两者的 `avg_max_reward` 几乎相同（0.9184 vs 0.9218），说明策略质量基本一致。

⚠️ 但措辞要准：这**不能**说「150K 和 200K 一样好」。n=50 时 16 个百分点的差异完全落在噪声范围内，属于**功效不足（underpowered）**——真实差异可能存在，只是本次没测出来。要下确定结论需要更多 episode（或多种子）。

实用含义：loss 早已走平（见下节），成功率也无显著差异，说明**这个配置在 150K 附近已基本收敛**，继续训到 200K 的边际收益未能被本次实验观测到。如果重跑，可以先只训 150K 并观察。

---

## 训练

| 项目 | 值 |
|---|---|
| 数据集 | `lerobot/pusht`（25,650 帧） |
| 步数 × batch | 200,000 × 8 = 1.6M 样本 |
| 学习率 | 1e-5 恒定（AdamW，`weight_decay=1e-4`，`grad_clip_norm=10`） |
| 解码器层数 | **7**（对齐论文 Table III） |
| 编码器层数 | 4（`dim_model=512`，`n_heads=8`，`dim_feedforward=3200`） |
| `chunk_size` | 100 |
| `n_action_steps` | 1（**推理参数，不参与训练损失**；设 1 是为了评估时能自由切换） |
| VAE | `use_vae=True`，`latent_dim=32`，`kl_weight=10.0` |
| 视觉骨干 | `resnet18`（ImageNet 预训练） |
| 训练 seed | 1000 |
| 硬件 | RTX 5060 Laptop (8GB), CUDA 12.8, PyTorch 2.8.0 |
| 总用时 | **4 小时 11 分 27 秒**（约 13.26 step/s） |
| 最终 loss | 0.0423 |
| 最终 grad norm | 1.84 |
| 产物 | 4 个 checkpoint（50K/100K/150K/200K），共 **3.8 GB** |

### 训练时间线（由 checkpoint 目录时间戳还原）

| 事件 | 时间 | 距开始 |
|---|---|---:|
| 训练开始（推算） | 2026-09-22 21:05 | 0:00:00 |
| checkpoint `050000` | 2026-09-22 22:06 | 1:00:26 |
| checkpoint `100000` | 2026-09-22 23:08 | 2:02:52 |
| checkpoint `150000` | 2026-09-23 00:12 | 3:06:26 |
| checkpoint `200000` | 2026-09-23 01:17 | 4:11:27 |

每 5 万步约 62 ~ 65 分钟，后期略慢（62 → 65 分钟），总体吞吐稳定。

### 后期 loss 曲线（部分恢复）

| step | 平均 `loss`（总损失） | 平均 `grad_norm` |
|---:|---:|---:|
| 120K | 0.0523 | 2.209 |
| 130K | 0.0507 | 2.156 |
| 140K | 0.0489 | 2.091 |
| 150K | 0.0475 | 2.033 |
| 160K | 0.0460 | 1.975 |
| 170K | 0.0447 | 1.945 |
| 180K | 0.0435 | 1.906 |
| 190K | 0.0424 | 1.873 |
| 200K | 0.0423 | 1.836 |

⚠️ **这段曲线覆盖范围只有 step 120K–200K，且无法从本仓库复算**：本次训练 stdout 未重定向到文件，原始日志已丢失，数据是从残留的终端会话记录里恢复的。详见 [`metrics/README.md`](metrics/README.md)。

可观察到的两点：loss 在后 3 万步基本走平（0.0435 → 0.0423），grad norm 持续单调下降。

这个「loss 走平」和上面的 **150K vs 200K 对比**相互印证：150K 已经拿到 36.0%、200K 拿到 52.0%，差异在 n=50 下不显著，说明**后期确实进入了收益递减区**。

### 训练命令

```bash
bash scripts/run_act_std_train.sh
```

脚本里包含与论文 Table III 的对齐说明：相对上一轮（v5，1 层解码器 + batch 32），本次只改了两处「不对等」——

1. `n_decoder_layers` 1 → **7**（论文 Table III = 7）
2. `batch_size` 32 → **8**（论文 Table III = 8，batch 与 lr 需配对使用）

并把 `n_action_steps` 设为 1，使评估时能自由切换推理设置。步数取 200K × batch 8 = 1.6M 样本，与上一轮 50K × batch 32 = 1.6M 相同，**让「见过的数据量」不变**。

---

## 模型规模：84.0M 参数（非 52M）

本仓库此前记录的 ACT 参数量为 52M，**本次更新已更正为 84.0M**。按模块拆解（从 `model.safetensors` 表头实测）：

| 模块 | 参数量 |
|---|---:|
| Transformer 解码器（7 层） | 37.75 M |
| Transformer 编码器（4 层） | 17.61 M |
| VAE 编码器（4 层） | 17.42 M |
| ResNet18 视觉骨干 | 11.19 M |
| **合计** | **83.97 M** |

权重文件 `model.safetensors` 为 335.9 MB。

产出「52M」的原因：解码器每层约 5.39 M 参数，1 层解码器时总量 = 84.0 − 6 × 5.39 ≈ **51.7 M** —— 也就是**旧配置（1 层解码器）的参数量**。本次为对齐论文改为 7 层后，参数量增至 84.0M。

---

## 指标说明：`pc_success` 与 `avg_max_reward`

两个数字差很多（52.0% vs 0.9218），容易混淆，在此说明。

### 定义

```python
# LeRobot PushT 环境
coverage = intersection_area / goal_area   # T 块与目标区域的重叠比例
success  = coverage > 0.95                 # 超过 95% 才算成功

# lerobot-eval 聚合
pc_success     = 成功 episode 数 / 总 episode 数 × 100      # 硬门槛
avg_max_reward = mean(每 episode 内达到的最高 coverage)      # 部分成功也算
```

| 指标 | C 组的值 | 含义 |
|---|---|---|
| `pc_success` | **52.0%** | 达到 95% 覆盖率的 episode 占比（硬门槛） |
| `avg_max_reward` | **0.9218** | 平均每回合最高推到 **92.2%** 覆盖率 |

**报告时应同时给出两个数字**，否则会误导：只看 `avg_max_reward` 会以为「模型 92% 都推到了」，实际上有一半的回合卡在 95% 门槛下方。

### 置信区间

成功率的 95% CI 用 **Wilson 区间**（比正态近似更可靠，尤其成功率接近 0 时）。n=50 时单次评估的标准误约 7 个百分点，所以：

- A（1/50）与 B（0/50）的差异**没有统计意义**（CI 重叠，2% 与 0% 的差别就是 1 个 episode）
- A/B 与 C 的差异**有统计意义**（p < 1e-8）

### 只跑了 1 个训练 seed

训练 seed 固定为 1000，因此本仓库给出的是**单次结果**的置信区间（来自 50 个测试 episode），**无法估计训练方差**，不能与多种子聚合的结果比较优劣。

---

## 这次提升混了三个变量（诚实性说明）

本仓库此前记录 ACT 成功率为 16.0%，本次为 52.0%。但**不能把提升归因到单一改动上**，因为相对上一轮（v5）同时改了三处：

| 变量 | 旧（v5） | 新（本次） |
|---|---|---|
| 解码器层数 | 1 | 7 |
| 训练步数 × batch | 50K × 32 | 200K × 8 |
| `n_action_steps` | 10 | **20** |

其中**只有第三项被单独隔离测过**（就是上面的 A/B/C 与系数扫描），前两项没有做控制变量。因此正确的说法是：

- ✅ 「在本次配置下，`n_action_steps=20` 明显优于 `=1`」（有 A/B/C 对照支撑）
- ❌ 「把解码器改成 7 层带来了 X% 提升」（未做消融，无法归因）

另外，此前那个 16.0% 来自当时并不明确的评估协议，已不再引用；统一到 `lerobot-eval` 管道后，同一口径下的历史数字是 10.0%（5/50，1 层解码器配置）。

---

## 可追溯性

README 中的每个数字都能核对：

| 文件 | 内容 | 大小 |
|---|---|---|
| `metrics/raw/eval_*.json` | **8 次评估的原始产物**，含全部 50 个 episode 的逐条分数与成功标记 | 6.6 ~ 7.1 KB × 8 |
| `metrics/eval_results.json` | 汇总结果 + Wilson CI + Fisher 精确检验，由脚本从上面 7 个文件重算 | 25 KB |
| `metrics/build_eval_results.py` | 复算脚本（仅标准库，无第三方依赖） | — |
| `metrics/train_config.json` | 本次训练完整配置（所有命令行覆盖后的最终值） | 5 KB |
| `metrics/train_curve_recovered.csv` | 后期 loss 曲线（⚠️ 不可从仓库复算，见说明） | — |
| `metrics/README.md` | 上述文件的说明与复算方法 | — |

```bash
# 一键复算全部评估指标
python3 metrics/build_eval_results.py
```

**未入库的内容**：模型权重（4 个 checkpoint 共 3.8 GB，可由训练脚本复现）、评估录制的 mp4 视频、训练原始日志（未保存到文件，见上）。

本地也已清理 `checkpoints/050000`、`checkpoints/100000`（评估用不到）；当前保留 `150000` 与 `200000`，`last` 指向 `200000`。要复现上面任意一次评估，只需把 `--policy.path` 指向对应步数的 `pretrained_model`。

---

## 本仓库做了什么 / 没做什么

✅ 已完成

- 在 RTX 5060 Laptop (8GB) 上跑通标准 ACT 训练（7 层解码器 + batch 8），200K 步 / 4 小时 11 分
- 按论文 Table III 对齐架构与 batch，并让「见过的数据量」与上一轮保持一致
- 用 `lerobot-eval` 完成 **8 次评估**（3 组主对照 + 4 档系数扫描 + 1 次 150K checkpoint 对比），均为 50 episodes / seed 1000
- 保存 8 次评估的**逐 episode 原始数据**并提交入库
- 完成 **150K vs 200K 对比**：多训的 5 万步在 n=50 下未能检出显著提升（p = 0.158）
- 定位出**真正影响成功率的是 `n_action_steps`**，并证伪了「时间集成系数没调好」的假设
- 更正参数量：52M → **84.0M**（并说明 52M 对应的其实是旧 1 层解码器配置）
- 提供可复现的 `scripts/` 训练与评估脚本

❌ 未包含

- **`n_action_steps` 的扫描**：只测了 1 与 20 两个点，无法判断 20 是否最优点（100 会怎样？5 会怎样？）
- **150K vs 200K 的差异未定论**：p = 0.158，n=50 功效不足，无法判断那 16 个百分点是真实提升还是噪声
- **架构消融**：解码器 1 层 vs 7 层、步数 50K vs 200K、batch 8 vs 32 均未单独隔离
- **多种子重复**：只跑了 seed 1000，无法估计训练方差
- **`use_vae=false` 消融**：CVAE 那 32 维 latent 值多少分，未测
- 对策略实现的任何修改（本仓库使用 LeRobot 官方代码）
- 模型权重（可由训练复现，未入库）

---

## 历史记录（v5 配置）

仓库里保留的 `train_command.sh` 与 `train_act_v5.log` 属于**上一轮实验（v5）**，配置与本轮不同，仅供追溯：

| | v5 | 本轮（std） |
|---|---|---|
| 解码器层数 | 1 | 7 |
| batch | 32 | 8 |
| 步数 | 50,000 | 200,000 |
| `n_action_steps` | 10 | 1（评估时设 20） |
| 参数 | ~51.7M | 84.0M |
| 统一口径成功率 | 10.0%（5/50） | **52.0%（26/50）** |

v5 的踩坑记录（视频解码相关，本轮沿用相同方案）：

1. pyav 解码器偶发死循环 → 换 torchcodec
2. torchcodec 与 torch 版本不匹配 → 降级到 0.7.0
3. AV1 视频随机 seek 死锁 → 重编码 H.264 + 关键帧间隔 10

---

## 环境配置

```bash
# 依赖：torchcodec==0.7.0（匹配 torch 2.8.0）+ FFmpeg 5
conda create -n ffmpeg5 -c conda-forge ffmpeg=5.1.2
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=~/miniforge3/envs/ffmpeg5/lib:$LD_LIBRARY_PATH

# 视频需重编码为 H.264 + 关键帧间隔 10
ffmpeg -i in.mp4 -c:v libx264 -g 10 out.mp4
```

`scripts/run_act_std_train.sh` 里已包含 `systemd-inhibit`，训练期间阻止系统休眠；并设置
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 以降低 8 GB 显存下 OOM 的风险。

---

## 引用

ACT: *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS 2023
项目主页: https://tonyzhaozh.github.io/aloha/

Diffusion Policy: *Visuomotor Policy Learning via Action Diffusion*, Chi et al., RSS 2023

LeRobot: https://github.com/huggingface/lerobot
