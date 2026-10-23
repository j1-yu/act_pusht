#!/usr/bin/env python3
"""Build the reproducible summary for the ACT PushT data-scale experiment."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPLIT_PATH = HERE / "episode_split.json"
OUT_JSON = HERE / "results.json"
OUT_CSV = HERE / "results.csv"
OUT_SVG = HERE / "data_scale.svg"

RUNS = {
    "10pct": HERE / "results" / "raw" / "data_scale_10pct_seed1000.json",
    "25pct": HERE / "results" / "raw" / "data_scale_25pct_seed1000.json",
    "50pct": HERE / "results" / "raw" / "data_scale_50pct_seed1000.json",
    # The existing full-data baseline used the same critical training/eval settings.
    "100pct": REPO / "metrics" / "raw" / "eval_C_nas20.json",
}


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (center - half), 100 * (center + half)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p value for paired binary outcomes."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def load(name: str, path: Path, split: dict) -> dict:
    data = json.loads(path.read_text())
    metrics = data["per_task"][0]["metrics"]
    overall = data["overall"]
    successes = [bool(x) for x in metrics["successes"]]
    k, n = sum(successes), len(successes)
    low, high = wilson(k, n)
    group = split["groups"][name]
    return {
        "data_fraction": name,
        "episode_count": group["episode_count"],
        "frame_count": group["frame_count"],
        "train_seed": 1000,
        "train_steps": 200000,
        "test_seed_start": 1000,
        "test_episodes": n,
        "successes": k,
        "success_rate_pct": round(100 * k / n, 2),
        "wilson_ci95_pct": [round(low, 2), round(high, 2)],
        "avg_max_coverage": round(float(overall["avg_max_reward"]), 4),
        "batched_eval_wall_s": round(float(overall["eval_s"]), 2),
        "source": str(path.relative_to(REPO)),
        "per_episode_success": successes,
    }


def make_svg(rows: list[dict]) -> None:
    width, height = 760, 410
    left, right, top, bottom = 70, 30, 35, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [10, 25, 50, 100]

    def x(v: float) -> float:
        return left + (v - 10) / 90 * plot_w

    def y(v: float) -> float:
        return top + (100 - v) / 100 * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.5}.success{fill:none;stroke:#2563eb;stroke-width:3}.coverage{fill:none;stroke:#f97316;stroke-width:3}</style>',
        '<text x="380" y="23" text-anchor="middle" font-size="18" font-weight="bold">ACT PushT：训练数据量与闭环表现</text>',
    ]
    for tick in range(0, 101, 20):
        yy = y(tick)
        parts += [
            f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>',
            f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" font-size="13">{tick}</text>',
        ]
    parts += [
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
    ]
    for value in xs:
        xx = x(value)
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+25}" text-anchor="middle" font-size="13">{value}%</text>')
    parts += [
        f'<text x="{left + plot_w/2:.1f}" y="{height-14}" text-anchor="middle" font-size="14">训练演示数据比例</text>',
        f'<text x="17" y="{top + plot_h/2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90 17 {top + plot_h/2:.1f})">百分比（%）</text>',
    ]
    success_pts = " ".join(f'{x(v):.1f},{y(r["success_rate_pct"]):.1f}' for v, r in zip(xs, rows))
    coverage_pts = " ".join(f'{x(v):.1f},{y(100*r["avg_max_coverage"]):.1f}' for v, r in zip(xs, rows))
    parts += [f'<polyline class="success" points="{success_pts}"/>', f'<polyline class="coverage" points="{coverage_pts}"/>']
    for value, row in zip(xs, rows):
        xx = x(value)
        for metric, color, offset in [
            (row["success_rate_pct"], "#2563eb", 15),
            (100 * row["avg_max_coverage"], "#f97316", -9),
        ]:
            yy = y(metric)
            parts += [
                f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{color}"/>',
                f'<text x="{xx:.1f}" y="{yy+offset:.1f}" text-anchor="middle" font-size="12" fill="{color}">{metric:.1f}</text>',
            ]
    parts += [
        '<line x1="500" y1="48" x2="530" y2="48" class="success"/><text x="538" y="53" font-size="13">成功率</text>',
        '<line x1="600" y1="48" x2="630" y2="48" class="coverage"/><text x="638" y="53" font-size="13">平均最高 coverage</text>',
        '</svg>',
    ]
    OUT_SVG.write_text("\n".join(parts) + "\n")


def main() -> None:
    split = json.loads(SPLIT_PATH.read_text())
    rows = [load(name, path, split) for name, path in RUNS.items()]
    replication_path = HERE / "results" / "raw" / "data_scale_50pct_seed2000.json"
    replication = load("50pct", replication_path, split)
    replication["data_fraction"] = "50pct_seed2000"
    replication["train_seed"] = 2000
    full_rep_path = HERE / "results" / "raw" / "data_scale_100pct_seed2000.json"
    full_replication = load("100pct", full_rep_path, split)
    full_replication["train_seed"] = 2000
    full = next(row for row in rows if row["data_fraction"] == "100pct")["per_episode_success"]
    for row in rows:
        if row["data_fraction"] == "100pct":
            continue
        cur = row["per_episode_success"]
        cur_only = sum(a and not b for a, b in zip(cur, full))
        full_only = sum((not a) and b for a, b in zip(cur, full))
        row["paired_vs_100pct"] = {
            "subset_only_success": cur_only,
            "full_only_success": full_only,
            "mcnemar_exact_p": round(mcnemar_exact(cur_only, full_only), 6),
        }
    seed_groups = {
        "50pct": [next(row for row in rows if row["data_fraction"] == "50pct"), replication],
        "100pct": [next(row for row in rows if row["data_fraction"] == "100pct"), full_replication],
    }
    aggregates = {}
    for name, group_rows in seed_groups.items():
        rates = [row["success_rate_pct"] for row in group_rows]
        coverages = [row["avg_max_coverage"] for row in group_rows]
        aggregates[name] = {
            "n_training_seeds": len(group_rows),
            "success_rate_mean_pct": round(statistics.mean(rates), 2),
            "success_rate_sample_sd_pp": round(statistics.stdev(rates), 2),
            "avg_max_coverage_mean": round(statistics.mean(coverages), 4),
            "avg_max_coverage_sample_sd": round(statistics.stdev(coverages), 4),
        }

    payload = {
        "question": "在训练更新次数固定为 200K 时，增加不同的演示轨迹是否改善 ACT 的 PushT 闭环表现？",
        "controlled": {
            "policy": "LeRobot ACT",
            "train_seed": 1000,
            "train_steps": 200000,
            "batch_size": 8,
            "n_action_steps_eval": 20,
            "test_seed_start": 1000,
            "test_episodes": 50,
        },
        "runs": rows,
        "seed_replication": replication,
        "full_seed_replication": full_replication,
        "multi_seed_summary": aggregates,
        "limitations": [
            "每个数据量只有一个训练 seed，尚未估计训练随机性。",
            "Wilson 区间只描述 50 个测试场景造成的成功率估计不确定性。",
            "batched_eval_wall_s 是 50 个并行环境的完整 rollout 用时，不是模型单次前向延迟。",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with OUT_CSV.open("w", newline="") as f:
        fields = [
            "data_fraction", "episode_count", "frame_count", "train_seed", "train_steps",
            "test_episodes", "successes", "success_rate_pct", "ci95_low_pct", "ci95_high_pct",
            "avg_max_coverage", "batched_eval_wall_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows + [replication, full_replication]:
            writer.writerow({
                **{key: row[key] for key in fields if key not in {"ci95_low_pct", "ci95_high_pct"}},
                "ci95_low_pct": row["wilson_ci95_pct"][0],
                "ci95_high_pct": row["wilson_ci95_pct"][1],
            })
    make_svg(rows)
    for row in rows:
        print(
            f'{row["data_fraction"]:>6}: {row["successes"]}/{row["test_episodes"]} '
            f'= {row["success_rate_pct"]:.1f}%  CI={row["wilson_ci95_pct"]}  '
            f'coverage={row["avg_max_coverage"]:.4f}'
        )


if __name__ == "__main__":
    main()
