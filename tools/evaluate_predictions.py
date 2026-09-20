"""Evaluate predictions exported by an official external baseline implementation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from drf.metrics import MetricMeter, compute_video_metrics


def _read(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("predictions", payload.get("items"))
    if not isinstance(payload, list):
        raise ValueError("预测文件必须是 CSV、JSONL 或 JSON list")
    return payload


def _first(item: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in item and item[name] not in (None, ""):
            return item[name]
    return default


def _canonical_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _load_expected(values: list[str]):
    expected = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--expected-index 格式必须为 dataset=path.json")
        dataset, raw_path = value.split("=", 1)
        index_path = Path(raw_path).expanduser().resolve()
        items = json.loads(index_path.read_text(encoding="utf-8"))
        dataset_items = {}
        for item in items:
            raw_image = _first(item, ("image_path", "image"))
            image = _canonical_path(str(raw_image), index_path.parent)
            dataset_items[image] = (
                int(item["label"]), str(item.get("video_id", ""))
            )
        expected[dataset] = dataset_items
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset", default=None, help="输入缺少 dataset 字段时使用")
    parser.add_argument(
        "--prediction-root", default=None,
        help="相对 image_path 的基准目录；默认使用预测文件所在目录",
    )
    parser.add_argument("--score-type", choices=["probability", "logit"], default="probability")
    parser.add_argument(
        "--expected-index", action="append", default=[], metavar="DATASET=JSON",
        help="可重复；强制预测路径、标签和 video_id 与测试索引完全一致",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.predictions).expanduser().resolve()
    prediction_root = (
        Path(args.prediction_root).expanduser().resolve()
        if args.prediction_root else source.parent
    )
    rows = _read(source)
    expected = _load_expected(args.expected_index)
    normalized = []
    seen = set()
    for index, item in enumerate(rows):
        dataset = str(_first(item, ("dataset", "dataset_name"), args.dataset or "unknown"))
        label = int(_first(item, ("label", "target", "y_true")))
        score = float(_first(item, ("pred_score", "score", "probability", "prob_fake", "logit")))
        if args.score_type == "logit":
            score = 1.0 / (1.0 + math.exp(-max(min(score, 80.0), -80.0)))
        if label not in (0, 1) or not 0.0 <= score <= 1.0:
            raise ValueError(f"row {index}: label={label}, score={score}")
        raw_image = _first(item, ("image_path", "path", "image"))
        if raw_image in (None, ""):
            raise ValueError(f"row {index}: 缺少 image_path/path/image")
        image_path = _canonical_path(str(raw_image), prediction_root)
        key = (dataset, image_path)
        if key in seen:
            raise ValueError(f"重复预测: {key}")
        seen.add(key)
        normalized.append({
            "dataset": dataset,
            "image_path": image_path,
            "video_id": str(_first(item, ("video_id", "video"), "")),
            "label": label,
            "pred_score": score,
        })

    grouped = defaultdict(list)
    for item in normalized:
        grouped[item["dataset"]].append(item)
    for dataset, expected_items in expected.items():
        actual_items = {item["image_path"]: item for item in grouped.get(dataset, [])}
        missing = set(expected_items) - set(actual_items)
        extra = set(actual_items) - set(expected_items)
        mismatched = []
        for image in set(expected_items) & set(actual_items):
            expected_label, expected_video = expected_items[image]
            actual = actual_items[image]
            if actual["label"] != expected_label or actual["video_id"] != expected_video:
                mismatched.append(image)
        if missing or extra or mismatched:
            raise ValueError(
                f"{dataset} 预测覆盖不一致: missing={len(missing)}, "
                f"extra={len(extra)}, label/video_mismatch={len(mismatched)}"
            )
    results, aucs = {}, []
    for dataset, items in sorted(grouped.items()):
        labels = [item["label"] for item in items]
        scores = [item["pred_score"] for item in items]
        videos = [item["video_id"] for item in items]
        meter = MetricMeter()
        meter.update(labels, scores)
        metrics = meter.compute()
        metrics["num_frames"] = len(items)
        if any(videos):
            metrics["video"] = compute_video_metrics(labels, scores, videos)
        results[dataset] = metrics
        if metrics["auc"] == metrics["auc"]:
            aucs.append(metrics["auc"])

    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "experiment": args.experiment,
        "avg_auc": sum(aucs) / len(aucs) if aucs else float("nan"),
        "per_dataset": results,
        "prediction_source": str(source),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    detailed = output.with_name(f"{output.stem}_detailed.json")
    detailed.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {output} datasets={len(results)} predictions={len(normalized)}")


if __name__ == "__main__":
    main()
