"""Validate DRF JSON indexes before spending GPU time."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--check-files", action="store_true")
    ap.add_argument("--require-video-id", action="store_true")
    args = ap.parse_args()

    failed = False
    for name in args.jsons:
        path = Path(name).expanduser().resolve()
        samples = json.loads(path.read_text(encoding="utf-8"))
        labels = Counter()
        missing = []
        invalid = 0
        video_ids = set()
        missing_video_id = 0
        for i, sample in enumerate(samples):
            image_path = sample.get("image_path", sample.get("image"))
            label = sample.get("label")
            if not image_path or label not in (0, 1):
                invalid += 1
                continue
            labels[int(label)] += 1
            if sample.get("video_id") not in (None, ""):
                video_ids.add(str(sample["video_id"]))
            else:
                missing_video_id += 1
            resolved = Path(image_path).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            if args.check_files and not resolved.is_file() and len(missing) < 10:
                missing.append(str(resolved))
        print(
            f"{path.name}: samples={len(samples)} labels={dict(labels)} "
            f"video_ids={len(video_ids)} missing_video_id={missing_video_id} "
            f"invalid={invalid} missing_shown={len(missing)}"
        )
        for item in missing:
            print(f"  missing: {item}")
        failed = (
            failed or invalid > 0 or (args.check_files and bool(missing))
            or (args.require_video_id and missing_video_id > 0)
        )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
