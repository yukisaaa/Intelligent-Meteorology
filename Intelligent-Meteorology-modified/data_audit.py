#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from tianchi_rainfall.data import load_ids_labels, load_offsets, parse_row

TODO_PATH = Path("#TODO")


def is_todo(path: Path | None) -> bool:
    return path is None or str(path) == "#TODO"


def answer_profile(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    values = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) != 1:
                raise ValueError(f"{path}: expected one column, got {len(row)}")
            values.append(float(row[0]))
    return {"path": path.name, "rows": len(values), "finite": all(math.isfinite(x) for x in values)}


def radar_profile(path: Path, offsets: np.ndarray, count: int) -> dict:
    indices = np.unique(np.linspace(0, len(offsets) - 1, min(count, len(offsets)), dtype=int))
    minima: list[float] = []
    maxima: list[float] = []
    means: list[float] = []
    zero_rates: list[float] = []
    negative_rates: list[float] = []
    malformed: list[int] = []
    nonfinite = 0
    with path.open("rb") as handle:
        for index in indices:
            handle.seek(int(offsets[index]))
            try:
                _, _, radar = parse_row(handle.readline())
            except Exception:
                malformed.append(int(index))
                continue
            nonfinite += int(radar.size - np.isfinite(radar).sum())
            minima.append(float(np.nanmin(radar)))
            maxima.append(float(np.nanmax(radar)))
            means.append(float(np.nanmean(radar)))
            zero_rates.append(float(np.mean(radar == 0)))
            negative_rates.append(float(np.mean(radar < 0)))
    return {
        "sampled_rows": int(len(indices)),
        "malformed_row_indices": malformed,
        "nonfinite_values": nonfinite,
        "minimum": min(minima) if minima else None,
        "maximum": max(maxima) if maxima else None,
        "mean_of_sample_means": float(np.mean(means)) if means else None,
        "mean_zero_rate": float(np.mean(zero_rates)) if zero_rates else None,
        "mean_negative_rate": float(np.mean(negative_rates)) if negative_rates else None,
    }


def dataset_profile(path: Path, index_dir: Path, radar_samples: int, expected_label: float | None) -> dict:
    offsets = load_offsets(path, index_dir)
    sample_ids, labels = load_ids_labels(path, offsets)
    duplicates = [sample_id for sample_id, count in Counter(sample_ids).items() if count > 1]
    profile = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "rows": len(sample_ids),
        "unique_ids": len(set(sample_ids)),
        "duplicate_ids": duplicates[:20],
        "finite_labels": int(np.isfinite(labels).sum()),
        "negative_labels": int((labels < 0).sum()),
        "zero_labels": int((labels == 0).sum()),
        "label_min": float(labels.min()),
        "label_max": float(labels.max()),
        "label_mean": float(labels.mean()),
        "label_std": float(labels.std()),
        "label_quantiles": {
            name: float(np.quantile(labels, quantile))
            for name, quantile in (
                ("q00", 0.0),
                ("q25", 0.25),
                ("q50", 0.50),
                ("q75", 0.75),
                ("q90", 0.90),
                ("q95", 0.95),
                ("q99", 0.99),
                ("q100", 1.0),
            )
        },
        "radar_sample": radar_profile(path, offsets, radar_samples),
    }
    if expected_label is not None:
        profile["unexpected_label_count"] = int(np.sum(labels != expected_label))
    return profile


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# CIKM2017 数据质量审计",
        "",
        "本审计只读取原始数据，不修改任何数据文件。雷达值检查采用等距抽样；ID和标签检查覆盖全部样本。",
        "",
        "| 数据集 | 样本数 | 文件大小 | ID重复 | 标签范围 | 非有限标签 | 雷达负值率 | 雷达解析异常 |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for name in ("train", "test_a", "test_b"):
        item = report.get(name)
        if not item:
            continue
        radar = item["radar_sample"]
        lines.append(
            f"| {name} | {item['rows']} | {item['bytes'] / 1024**3:.2f} GiB | "
            f"{item['rows'] - item['unique_ids']} | {item['label_min']:.3f}–{item['label_max']:.3f} | "
            f"{item['rows'] - item['finite_labels']} | {radar['mean_negative_rate']:.4%} | "
            f"{len(radar['malformed_row_indices']) + int(radar['nonfinite_values'] > 0)} |"
        )
    train = report.get("train")
    if train:
        q = train["label_quantiles"]
        lines.extend(
            [
                "",
                "## 训练标签分布",
                "",
                f"- 零降水占比：{train['zero_labels'] / train['rows']:.2%}",
                f"- 中位数：{q['q50']:.3f}",
                f"- q90/q95/q99：{q['q90']:.3f} / {q['q95']:.3f} / {q['q99']:.3f}",
                f"- 最大值：{q['q100']:.3f}",
                "",
                "## 建模风险与处理",
                "",
                "- 标签为长尾分布时，普通随机回归容易低估强降水；训练代码使用分位数加权损失并报告分阈值指标。",
                "- 数据没有日期、站点或天气过程字段，无法严格按暴雨过程分组；默认采用固定种子的标签分位数分层划分，并保存逐行划分清单。",
                "- Test A/B 标签为 -1 时仅可用于预测，不可用于本地精度评估。",
                "- 雷达负值按缺测哨兵处理；数据加载器只在内存中将其置零，不改写原文件。",
                "- 原始文件保持只读；索引、报告和模型输出写入项目输出目录。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=TODO_PATH)
    parser.add_argument("--test-a-file", type=Path, default=TODO_PATH)
    parser.add_argument("--test-b-file", type=Path, default=TODO_PATH)
    parser.add_argument("--answer-a", type=Path, default=TODO_PATH)
    parser.add_argument("--answer-b", type=Path, default=TODO_PATH)
    parser.add_argument("--index-dir", type=Path, default=PROJECT_DIR / "cache" / "indexes")
    parser.add_argument("--output-dir", type=Path, default=TODO_PATH)
    parser.add_argument("--radar-samples", type=int, default=32)
    args = parser.parse_args()
    if is_todo(args.train_file) or is_todo(args.output_dir):
        parser.error("Provide --train-file and --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "train": dataset_profile(args.train_file, args.index_dir, args.radar_samples, None),
        "test_a": None
        if is_todo(args.test_a_file)
        else dataset_profile(args.test_a_file, args.index_dir, args.radar_samples, -1.0),
        "test_b": None
        if is_todo(args.test_b_file)
        else dataset_profile(args.test_b_file, args.index_dir, args.radar_samples, -1.0),
        "answer_a": None if is_todo(args.answer_a) else answer_profile(args.answer_a),
        "answer_b": None if is_todo(args.answer_b) else answer_profile(args.answer_b),
    }
    (args.output_dir / "data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(args.output_dir / "data_audit.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
