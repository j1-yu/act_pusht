# `metrics/` 说明

本目录保存 README 中**每一个评估数字的原始证据**，以及复算方法。

## 文件清单

| 文件 | 内容 | 来源 |
|---|---|---|
| `raw/eval_A_nas1.json` | A 组：不开时间集成，`n_action_steps=1` | `lerobot-eval` 写出的 `eval_info.json`，**原样复制，未做任何修改** |
| `raw/eval_B_te0.01_nas1.json` | B 组：时间集成 `coeff=0.01`，`n_action_steps=1` | 同上 |
| `raw/eval_C_nas20.json` | C 组：不开时间集成，`n_action_steps=20`（step 200000） | 同上 |
| `raw/eval_C150k_nas20.json` | 同 C 配置，但用 **step 150000** 的 checkpoint | 同上 |
| `raw/eval_te0.05_nas1.json` | 时间集成系数扫描 `coeff=0.05` | 同上 |
| `raw/eval_te0.1_nas1.json` | `coeff=0.1` | 同上 |
| `raw/eval_te0.5_nas1.json` | `coeff=0.5` | 同上 |
| `raw/eval_te1.0_nas1.json` | `coeff=1.0` | 同上 |
| `eval_results.json` | 上述 8 次的汇总：成功率、Wilson 95% CI、`avg_max_reward`、**逐 episode 原始分数**、Fisher 精确检验 | 由 `build_eval_results.py` 生成 |
| `build_eval_results.py` | 复算脚本，仅用标准库 | — |
| `train_config.json` | 本次训练（step 200000）的完整配置（所有命令行覆盖后的最终值） | 从 `pretrained_model/train_config.json` 复制 |
| `train_curve_recovered.csv` | 训练后期 loss 曲线（step 120K–200K） | ⚠️ **无法从本仓库复算，见下节** |

## 复算方法

```bash
python3 metrics/build_eval_results.py
```

脚本会重新读取 `raw/` 下的 8 个原始文件，重算成功率、Wilson 置信区间与 Fisher 精确检验，并覆盖写出 `eval_results.json`。输出示例：

```
  A_nas1           step=200000 nas=1   te=None   1/50 =   2.0%  95%CI [0.3, 10.5]  avg_max_reward=0.1999
  B_te0.01_nas1    step=200000 nas=1   te=0.01   0/50 =   0.0%  95%CI [0.0, 7.1]   avg_max_reward=0.3911
  C_nas20          step=200000 nas=20  te=None  26/50 =  52.0%  95%CI [38.5, 65.2]  avg_max_reward=0.9218
  C150k_nas20      step=150000 nas=20  te=None  18/50 =  36.0%  95%CI [24.1, 49.9]  avg_max_reward=0.9184
```

## 原始评估产物里有什么

`eval_info.json` 由 `lerobot-eval` 自动写出，包含两部分：

- **`per_task[0].metrics`**：**逐 episode 原始数据** —— `sum_rewards` / `max_rewards` / `successes` 各 50 个值
  （`successes` 是布尔序列，`True` 表示该 episode 达到 `coverage > 0.95`）
- **`overall`**：聚合值（`avg_sum_reward` / `avg_max_reward` / `pc_success` / `eval_s`）

因此 README 里的 `26/50` 可以逐个 episode 核对，例如用：

```bash
python3 -c "
import json
d = json.load(open('metrics/raw/eval_C_nas20.json'))
s = d['per_task'][0]['metrics']['successes']
print(f'{sum(s)}/{len(s)} = {100*sum(s)/len(s):.1f}%')
"
```

## 评估协议

| 项 | 值 |
|---|---|
| 权重 | `outputs/train/act_pusht_std/checkpoints/<step>/pretrained_model`：8 次中有 7 次用 step 200000，1 次用 step 150000 |
| 环境 | `--env.type=pusht`（LeRobot 内置 PushT 仿真） |
| episode 数 | 50（`--eval.n_episodes=50`） |
| 并行环境数 | 50（`--eval.batch_size=50`） |
| seed | 1000（与训练 seed 相同） |
| 成功的定义 | `coverage > 0.95`（T 块与目标区域重叠率超过 95%） |

**八次评估为什么可比**：同一 seed、同样 50 个 episode。LeRobot 由 `start_seed` 推出逐 episode 初始局面（`start_seed + i`），seed 固定即意味着**每次都面对完全相同的 50 个初始局面**，所以差异只可能来自推理设置或 checkpoint。

## ⚠️ 关于 `train_curve_recovered.csv`

**这个文件无法从本仓库复算，引用时请谨慎。**

原因：本次训练的 stdout **没有重定向到文件**（直接在终端里运行），终端会话结束后原始日志已丢失。该 CSV 是从残留的终端会话记录中正则提取日志行、按 10K 步分桶取均值得到的，**覆盖范围只有 step 120K–200K**，不是完整曲线。

提取时用 `epch ≈ step × batch / 25650`（`25650` 为 `lerobot/pusht` 的帧数）过滤掉了旧运行（`batch=32`）混入的日志行，共剔除 2 行、保留 376 行。

**下次训练请保存日志**：

```bash
bash scripts/run_act_std_train.sh 2>&1 | tee train_act_std.log
```

这样 `loss` / `grad_norm` / 吞吐都能完整留存，才能给出完整曲线。

> 注意：日志里的 `loss` 是**总损失** `L1 + kl_weight × KL`（`kl_weight=10`），不是单纯的 L1 动作误差。
> LeRobot 只把总损失写进 `train_metrics.loss`，`l1_loss` / `kld_loss` 的拆分算在 `loss_dict` 里，
> 只在 wandb 开启时才可见。本次训练 `--wandb.enable=false`，因此**无法从日志区分这两项**。
