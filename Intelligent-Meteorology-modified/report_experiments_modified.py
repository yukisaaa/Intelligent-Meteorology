#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_records(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def best_record(records: list[dict]) -> dict:
    return min(records, key=lambda record: record["validation"]["rmse"])


def read_predictions(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with (run_dir / "validation_predictions.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["target"]) for row in rows]),
        np.asarray([float(row["prediction"]) for row in rows]),
    )


def safe_name(config: dict, run_dir: Path) -> str:
    model = config.get("model", run_dir.name)
    head = config.get("head_variant", "full")
    loss = config.get("loss", "unknown")
    return f"{model}-{head}-{loss}-s{config.get('seed', '?')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("#TODO"))
    args = parser.parse_args()
    if str(args.output_dir) == "#TODO":
        parser.error("Replace #TODO or provide --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for run_dir in args.run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        records = read_records(run_dir)
        best = best_record(records)
        targets, predictions = read_predictions(run_dir)
        runs.append(
            {
                "dir": run_dir,
                "config": config,
                "name": safe_name(config, run_dir),
                "records": records,
                "best": best,
                "targets": targets,
                "predictions": predictions,
            }
        )
    baselines = json.loads(
        (runs[0]["dir"] / "baselines.json").read_text(encoding="utf-8")
    )

    summary_path = args.output_dir / "experiment_summary.csv"
    fields = [
        "name",
        "model",
        "head_variant",
        "loss",
        "seed",
        "best_epoch",
        "mae",
        "rmse",
        "holdout_mae",
        "holdout_rmse",
        "r2",
        "pearson",
        "q90_csi",
        "q90_pod",
        "q90_far",
        "q95_csi",
        "q95_pod",
        "q95_far",
    ]
    rows = []
    for run in runs:
        config = run["config"]
        best = run["best"]
        metrics = best["validation"]
        holdout = json.loads((run["dir"] / "holdout_metrics.json").read_text(encoding="utf-8"))
        categorical = metrics.get("categorical", {})
        rows.append(
            {
                "name": run["name"],
                "model": config.get("model"),
                "head_variant": config.get("head_variant", "full"),
                "loss": config.get("loss"),
                "seed": config.get("seed"),
                "best_epoch": best["epoch"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "holdout_mae": holdout["mae"],
                "holdout_rmse": holdout["rmse"],
                "r2": metrics["r2"],
                "pearson": metrics["pearson"],
                "q90_csi": categorical.get("q90", {}).get("csi"),
                "q90_pod": categorical.get("q90", {}).get("pod"),
                "q90_far": categorical.get("q90", {}).get("far"),
                "q95_csi": categorical.get("q95", {}).get("csi"),
                "q95_pod": categorical.get("q95", {}).get("pod"),
                "q95_far": categorical.get("q95", {}).get("far"),
            }
        )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["head_variant"], row["loss"])].append(row)
    seed_summary = []
    for (model, head_variant, loss), items in grouped.items():
        rmse_values = np.asarray([float(item["rmse"]) for item in items])
        mae_values = np.asarray([float(item["mae"]) for item in items])
        holdout_rmse_values = np.asarray([float(item["holdout_rmse"]) for item in items])
        holdout_mae_values = np.asarray([float(item["holdout_mae"]) for item in items])
        seed_summary.append(
            {
                "model": model,
                "head_variant": head_variant,
                "loss": loss,
                "runs": len(items),
                "rmse_mean": float(rmse_values.mean()),
                "rmse_std": float(rmse_values.std(ddof=1)) if len(items) > 1 else 0.0,
                "mae_mean": float(mae_values.mean()),
                "mae_std": float(mae_values.std(ddof=1)) if len(items) > 1 else 0.0,
                "holdout_rmse_mean": float(holdout_rmse_values.mean()),
                "holdout_rmse_std": float(holdout_rmse_values.std(ddof=1)) if len(items) > 1 else 0.0,
                "holdout_mae_mean": float(holdout_mae_values.mean()),
                "holdout_mae_std": float(holdout_mae_values.std(ddof=1)) if len(items) > 1 else 0.0,
            }
        )
    with (args.output_dir / "seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_summary[0]))
        writer.writeheader()
        writer.writerows(seed_summary)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for run in runs:
        epochs = [record["epoch"] for record in run["records"]]
        rmses = [record["validation"]["rmse"] for record in run["records"]]
        axis.plot(epochs, rmses, marker="o", linewidth=1.5, label=run["name"])
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation RMSE")
    axis.set_title("Validation learning curves")
    for baseline_name, baseline in baselines.items():
        axis.axhline(
            baseline["rmse"],
            linestyle="--",
            linewidth=1,
            alpha=0.6,
            label=f"baseline:{baseline_name}",
        )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.savefig(args.output_dir / "learning_curves.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    names = [row["name"] for row in rows]
    positions = np.arange(len(names))
    width = 0.38
    axis.bar(positions - width / 2, [row["mae"] for row in rows], width, label="MAE")
    axis.bar(positions + width / 2, [row["rmse"] for row in rows], width, label="RMSE")
    axis.set_xticks(positions, names, rotation=25, ha="right")
    axis.set_title("Best validation error by run")
    axis.axhline(
        baselines["train_mean"]["rmse"],
        linestyle="--",
        color="black",
        linewidth=1,
        label="train-mean RMSE",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(args.output_dir / "model_comparison.png", dpi=180)
    plt.close(figure)

    columns = min(3, len(runs))
    rows_count = int(np.ceil(len(runs) / columns))
    figure, axes = plt.subplots(rows_count, columns, figsize=(5 * columns, 4 * rows_count), squeeze=False)
    for axis, run in zip(axes.flat, runs):
        targets = run["targets"]
        predictions = run["predictions"]
        maximum = max(float(targets.max()), float(predictions.max()), 1.0)
        axis.scatter(targets, predictions, s=10, alpha=0.5)
        axis.plot([0, maximum], [0, maximum], linestyle="--", color="black", linewidth=1)
        axis.set_title(run["name"], fontsize=9)
        axis.set_xlabel("Observed rainfall")
        axis.set_ylabel("Predicted rainfall")
    for axis in axes.flat[len(runs) :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output_dir / "observed_vs_predicted.png", dpi=180)
    plt.close(figure)

    best_row = min(rows, key=lambda row: float(row["rmse"]))
    best_baseline_name, best_baseline = min(
        baselines.items(), key=lambda item: float(item[1]["rmse"])
    )
    markdown = [
        "# 自动实验报告",
        "",
        f"共汇总 {len(rows)} 个运行。按验证集 RMSE，当前最佳运行是 `{best_row['name']}`，RMSE={float(best_row['rmse']):.4f}。",
        f"该运行在独立留出集（未参与早停或融合权重选择）上的 RMSE={float(best_row['holdout_rmse']):.4f}。各运行的留出指标见 `experiment_summary.csv`。",
        f"最佳简单基线是 `{best_baseline_name}`，RMSE={float(best_baseline['rmse']):.4f}。",
        "",
        "## 输出图表",
        "",
        "- `learning_curves.png`：验证集学习曲线",
        "- `model_comparison.png`：模型 MAE/RMSE 对比",
        "- `observed_vs_predicted.png`：真实值与预测值散点图",
        "- `experiment_summary.csv`：完整可复查指标表",
        "- `seed_summary.csv`：相同配置在不同模型随机种子下的均值与标准差",
        "",
        "## 使用说明",
        "",
        "正式报告应只引用全量训练结果。小样本 smoke test 只能证明代码通路可运行。",
    ]
    (args.output_dir / "experiment_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(f"Wrote experiment report to {args.output_dir}")


if __name__ == "__main__":
    main()
