# ACT 复现：PushT 任务（LeRobot 框架）

复现 ACT（Action Chunking with Transformers, RSS 2023），在 PushT 任务上训练视觉运动策略，并与 Diffusion Policy 做对比。

## 演示视频

![ACT 推 T 块](demo_act.gif)

（模型在 PushT 仿真中自主推 T 块，成功 episode）

## 结果对比（同一评估管道，50 episodes）

| 模型 | 成功率 | 95% 置信区间 | 平均最大重叠率 |
|---|---|---|---|
| **官方 Diffusion Policy**<br>（`lerobot/diffusion_pusht`，200k 步） | **72.0%**（36/50） | [59.6%, 84.4%] | 0.9762 |
| **本仓库 ACT**<br>（`act_pusht_v5`，50k 步） | **10.0%**（5/50） | [1.7%, 18.3%] | 0.5645 |

**Fisher 精确检验：p = 2.1 × 10⁻¹⁰**，差距 62 个百分点，
置信区间完全不重叠。

> ⚠️ **正确的解读是**：「在**当前配置**下，ACT 显著差于 DP」，
> **而不是**「ACT 方法本身差于 DP」。两者的训练预算与超参数尚未对齐，
> 详见 [EVAL_COMPARISON.md](EVAL_COMPARISON.md) 的「诚实的局限」一节。

### 校准：先证明评估管道可信

在对比之前，先用**官方发布的、在 LeRobot 里训练好的 DP 权重**验证评估管道：

| 指标 | 官方公布（500 episode） | 本次复现（50 episode） |
|---|---|---|
| 平均最大重叠率 | 0.955 | **0.9762** |
| 成功率 | 65.4% | **72.0%** |

72.0% 的 95% 置信区间 [59.6%, 84.4%] **覆盖**官方公布的 65.4%
→ **评估管道可信，两个数字可比。**

### `n_action_steps` 消融（最有价值的发现）

`n_action_steps` 决定「执行多少步后才重新规划」，即**重规划间隔**。
扫描结果（每次 50 episodes，`chunk_size` 固定 100）：

| `n_action_steps` | 成功率 | 平均最大重叠率 |
|---:|---:|---:|
| 1 | 0.0% | 0.063 |
| 2 | 0.0% | 0.083 |
| 5 | 0.0% | 0.219 |
| 10 | 10.0% | 0.564 |
| **20** | **16.0%** | **0.698** |
| 50 | 2.0% | 0.547 |
| 100 | 2.0% | 0.354 |

**单峰曲线，峰值在 20。** 值得注意：

- **ACT 官方默认的 `n_action_steps = chunk_size = 100` 在 PushT 上只有 2.0%。**
  这个默认值是为 ALOHA 双臂任务（50 Hz）调的，100 步 = 2 秒；
  而 PushT 是 10 Hz，100 步 = **10 秒**，几乎覆盖整个 episode（上限 30 秒）。
- 但最优的 16.0% **仍远低于 DP 的 72.0%**，说明调这一个参数不足以弥合差距。

### 方法对比

| 维度 | Diffusion Policy | ACT |
|---|---|---|
| 生成方式 | 扩散模型（DDPM/DDIM，**生成式**） | CVAE + Transformer（**也是生成式**） |
| 论文原文 | "conditional denoising diffusion process" | "learns a **generative model** over action sequences" |
| 会议 | RSS 2023 | RSS 2023 |
| 关系 | **同期工作（concurrent work）**，作者无交集 | DP 论文**没有**把 ACT 当基线 |
| 动作序列 | 预测 16 步，执行 8 步 | 一次预测 100 步，执行 N 步 |
| 模型参数 | 262M | 52M |

> **两处此前的错误已更正**：
> 1. ACT 曾被写成「判别式」，实际上论文摘要明写 *"learns a **generative model** over
>    action sequences"*，**ACT 也是生成式模型**。
> 2. 曾被列为「DP 93.2% vs ACT 16%」。93.2% 那个数字已废弃——它来自
>    `n_test=1` 的错误评估协议，不可追溯。两个 loss（0.025 vs 0.058）也不可比：
>    DP 是**对噪声的 MSE**，ACT 是**对动作的 L1 + KL 散度**，量纲完全不同。

详细的对比方法、统计检验、复现步骤与局限，见
**[EVAL_COMPARISON.md](EVAL_COMPARISON.md)**。

## 关键配置

- `chunk_size=100`（照搬 ACT 官方默认，为 ALOHA 任务设计）
- `n_action_steps=10`（训练时配置；**评估时扫描发现 20 才是最优**）
- `n_obs_steps=1`（只喂当前 1 帧）
- `use_vae=True`, `latent_dim=32`, `kl_weight=10`
- `optimizer_lr=1e-5`, `batch_size=32`, `steps=50000`（62.4 epoch）
- torchcodec + FFmpeg 5（视频解码）
- num_workers=1（避免多进程解码死锁）

## 踩坑记录

1. pyav 解码器偶发死循环 → 换 torchcodec
2. torchcodec 与 torch 版本不匹配 → 降级到 0.7.0
3. AV1 视频随机 seek 死锁 → 重编码 H.264 + 关键帧间隔 10

## 训练

完整训练日志见 train_act_v5.log

## 引用

ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, RSS 2023
项目主页: https://tonyzhaozh.github.io/aloha/
