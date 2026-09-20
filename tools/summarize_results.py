"""Merge DRF test_result.json files into a paper-ready Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value, percent: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if number != number:
        return "-"
    return f"{number * 100:.2f}" if percent else f"{number:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="test_result.json files")
    parser.add_argument("--out", default="logs/experiment_summary.md")
    args = parser.parse_args()

    rows = []
    for value in args.results:
        path = Path(value).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        experiment = payload.get("experiment", path.parent.name)
        for dataset, metrics in payload.get("per_dataset", {}).items():
            video = metrics.get("video", {})
            rows.append([
                experiment,
                dataset,
                str(metrics.get("num_frames", "-")),
                _fmt(metrics.get("auc")),
                _fmt(metrics.get("ap")),
                _fmt(metrics.get("eer"), percent=True),
                _fmt(metrics.get("acc")),
                _fmt(video.get("auc")),
                str(video.get("num_videos", "-")),
            ])

    header = ["Experiment", "Dataset", "Frames", "Frame AUC", "AP", "EER(%)", "ACC@0.5", "Video AUC", "Videos"]
    lines = [
        "# DRF Experiment Summary",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
