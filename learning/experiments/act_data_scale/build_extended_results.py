#!/usr/bin/env python3
"""Summarize action execution and visual corruption evaluations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def paired_comparison(a_path: Path, b_path: Path) -> dict:
    a = json.loads(a_path.read_text())["per_task"][0]["metrics"]["successes"]
    b = json.loads(b_path.read_text())["per_task"][0]["metrics"]["successes"]
    if len(a) != len(b):
        raise ValueError("Paired evaluations must have the same number of episodes")
    a_only = sum(bool(x) and not bool(y) for x, y in zip(a, b, strict=True))
    b_only = sum(not bool(x) and bool(y) for x, y in zip(a, b, strict=True))
    discordant = a_only + b_only
    if discordant:
        p_value = min(
            1.0,
            2 * sum(math.comb(discordant, i) for i in range(min(a_only, b_only) + 1))
            / (2**discordant),
        )
    else:
        p_value = 1.0
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "mcnemar_exact_p": p_value}


def overall(path: Path) -> dict:
    data = json.loads(path.read_text())
    metrics = data["per_task"][0]["metrics"]
    return {
        "successes": int(sum(metrics["successes"])),
        "episodes": len(metrics["successes"]),
        "success_rate_pct": float(data["overall"]["pc_success"]),
        "avg_max_coverage": float(data["overall"]["avg_max_reward"]),
        "eval_s": float(data["overall"]["eval_s"]),
        "source": str(path.relative_to(HERE.parent.parent.parent)),
    }


def main() -> None:
    raw = HERE / "results" / "raw"
    action = {
        "1": overall(raw / "action_nas1.json"),
        "5": overall(raw / "action_nas5.json"),
        "10": overall(raw / "action_nas10.json"),
        "20": overall(raw / "visual_clean.json"),
    }
    visual_names = {
        "clean": "visual_clean.json",
        "brightness_0.7": "visual_brightness070.json",
        "gaussian_noise_sigma_0.1": "visual_noise_sigma010.json",
        "occlusion_area_0.2": "visual_occlusion020.json",
    }
    visual = {name: overall(raw / filename) for name, filename in visual_names.items()}
    payload = {
        "action_execution_length": {
            "model": "act_pusht_std/checkpoints/200000",
            "train_unchanged": True,
            "eval_seed_start": 1000,
            "runs": action,
            "paired_vs_20": {
                steps: paired_comparison(
                    raw / filename,
                    raw / "visual_clean.json",
                )
                for steps, filename in (("1", "action_nas1.json"), ("5", "action_nas5.json"), ("10", "action_nas10.json"))
            },
        },
        "visual_perturbation": {
            "model": "act_pusht_std/checkpoints/200000",
            "n_action_steps": 20,
            "eval_seed_start": 1000,
            "runs": visual,
            "paired_vs_clean": {
                name: paired_comparison(
                    raw / "visual_clean.json",
                    raw / filename,
                )
                for name, filename in (
                    ("brightness_0.7", "visual_brightness070.json"),
                    ("gaussian_noise_sigma_0.1", "visual_noise_sigma010.json"),
                    ("occlusion_area_0.2", "visual_occlusion020.json"),
                )
            },
            "note": "All conditions use deterministic CUDA algorithms, the same evaluator, environment seeds 1000-1049, n_action_steps as recorded, and image corruption before the saved ACT normalizer; physics is unchanged. Occlusion severity is image area fraction.",
        },
    }
    (HERE / "extended_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with (HERE / "action_execution_results.csv").open("w", newline="") as f:
        fields = ["n_action_steps", "successes", "episodes", "success_rate_pct", "avg_max_coverage", "eval_s"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for steps, row in action.items():
            writer.writerow({"n_action_steps": steps, **{k: row[k] for k in fields[1:]}})
    with (HERE / "visual_perturbation_results.csv").open("w", newline="") as f:
        fields = ["condition", "successes", "episodes", "success_rate_pct", "avg_max_coverage", "eval_s"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for condition, row in visual.items():
            writer.writerow({"condition": condition, **{k: row[k] for k in fields[1:]}})
    for group in (action, visual):
        for name, row in group.items():
            print(name, f'{row["successes"]}/{row["episodes"]}', row["avg_max_coverage"])


if __name__ == "__main__":
    main()
