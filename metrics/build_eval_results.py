#!/usr/bin/env python3
"""从 metrics/raw/ 里的 LeRobot 评估产物重新生成 metrics/eval_results.json。

数据来源
--------
LeRobot `lerobot-eval` 每次评估会写出一个 `eval_info.json`，其中
`per_task[0].metrics` 保存了**逐 episode 的原始分数**（`sum_rewards` /
`max_rewards` / `successes`），`overall` 保存聚合值。

原始文件位于 `outputs/eval/<日期>/<时间>_pusht_act/eval_info.json`，
已原样复制进 `metrics/raw/`（未做任何修改）。

用法
----
    python3 metrics/build_eval_results.py

仅使用标准库，无第三方依赖。
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_PATH = HERE / "eval_results.json"

# 每次评估的推理设置。
# 生成方式见 scripts/run_act_std_eval.sh（tag A/B/C）与「时间集成系数扫描」部分。
# 三次 A/B/C 与四次扫描共用一个权重、同一个 seed，只改推理参数。
RUNS: list[tuple[str, int, float | None, str]] = [
    ("A_nas1", 1, None, "基线：不开时间集成，每步重新规划"),
    ("B_te0.01_nas1", 1, 0.01, "论文默认做法：开时间集成"),
    ("C_nas20", 20, None, "一次预测执行 20 步后重新规划"),
    ("te0.05_nas1", 1, 0.05, "时间集成系数扫描"),
    ("te0.1_nas1", 1, 0.1, "时间集成系数扫描"),
    ("te0.5_nas1", 1, 0.5, "时间集成系数扫描"),
    ("te1.0_nas1", 1, 1.0, "时间集成系数扫描"),
]

META = {
    "policy_path": "outputs/train/act_pusht_std/checkpoints/last/pretrained_model",
    "checkpoint_step": 200000,
    "env": "pusht",
    "eval_n_episodes": 50,
    "eval_batch_size": 50,
    "seed": 1000,
    "success_definition": "PushT 环境内 coverage > 0.95（即 T 块与目标区域重叠率超过 95%）",
    "comparable": (
        "全部 7 次评估使用同一份权重（step 200000）、同一 seed=1000、同样 50 个 "
        "episode。LeRobot 由 seed 推出逐 episode 初始局面（start_seed + i），"
        "因此各次面对的是同一批初始局面，数字可直接比较。"
    ),
}


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """成功率的 Wilson 95% 置信区间（比正态近似更可靠，尤其 k 接近 0 时）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (center - half, center + half)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """2x2 列联表的 Fisher 精确检验双尾 p 值。

    表结构::

        [[a, b],
         [c, d]]

    即 a = 组1成功, b = 组1失败, c = 组2成功, d = 组2失败。
    双尾定义：所有概率 <= 观测表概率的表的概率之和（标准做法）。
    """
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1 = a + c

    lo = max(0, col1 - row2)
    hi = min(row1, col1)

    def prob(x: int) -> float:
        return comb(row1, x) * comb(row2, col1 - x) / comb(n, col1)

    p_obs = prob(a)
    total = 0.0
    for x in range(lo, hi + 1):
        p = prob(x)
        if p <= p_obs * (1 + 1e-12):
            total += p
    return min(1.0, total)


def load_run(tag: str) -> dict:
    path = RAW_DIR / f"eval_{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少原始评估产物: {path}")

    data = json.loads(path.read_text())
    metrics = data["per_task"][0]["metrics"]
    overall = data["overall"]

    successes = [bool(s) for s in metrics["successes"]]
    max_rewards = [round(float(x), 6) for x in metrics["max_rewards"]]
    sum_rewards = [round(float(x), 6) for x in metrics["sum_rewards"]]

    n = len(successes)
    k = sum(successes)
    lo, hi = wilson_ci(k, n)

    return {
        "tag": tag,
        "raw_file": f"raw/eval_{tag}.json",
        "n_episodes": n,
        "n_success": k,
        "pc_success": round(100.0 * k / n, 2),
        "pc_success_ci95": [round(100 * lo, 2), round(100 * hi, 2)],
        "avg_max_reward": round(overall["avg_max_reward"], 4),
        "avg_sum_reward": round(overall["avg_sum_reward"], 4),
        "eval_s": round(overall["eval_s"], 2),
        "per_episode": {
            "success": successes,
            "max_reward": max_rewards,
            "sum_reward": sum_rewards,
        },
    }


def main() -> None:
    runs: list[dict] = []
    for tag, nas, te, note in RUNS:
        result = load_run(tag)
        result["n_action_steps"] = nas
        result["temporal_ensemble_coeff"] = te
        result["note"] = note
        runs.append(result)

    # 以 C（最佳配置）为基准做 Fisher 精确检验
    best = next(r for r in runs if r["tag"] == "C_nas20")
    comparisons = []
    for r in runs:
        if r is best:
            continue
        p = fisher_exact_2x2(
            best["n_success"], best["n_episodes"] - best["n_success"],
            r["n_success"], r["n_episodes"] - r["n_success"],
        )
        comparisons.append({
            "a": "C_nas20",
            "b": r["tag"],
            "a_success": f"{best['n_success']}/{best['n_episodes']}",
            "b_success": f"{r['n_success']}/{r['n_episodes']}",
            "difference_pp": round(best["pc_success"] - r["pc_success"], 2),
            "fisher_exact_p_two_sided": float(f"{p:.3g}"),
            "significant_at_0.05": bool(p < 0.05),
        })

    payload = {
        "meta": META,
        "runs": runs,
        "comparisons_vs_best": comparisons,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"已写出 {OUT_PATH.relative_to(HERE.parent)}")
    for r in runs:
        ci = r["pc_success_ci95"]
        print(
            f"  {r['tag']:<16} nas={r['n_action_steps']:<3} te={str(r['temporal_ensemble_coeff']):<5} "
            f"{r['n_success']:>2}/{r['n_episodes']} = {r['pc_success']:>5.1f}%  "
            f"95%CI [{ci[0]:.1f}, {ci[1]:.1f}]  avg_max_reward={r['avg_max_reward']:.4f}"
        )


if __name__ == "__main__":
    main()
