"""Build a deterministic frame index with labels and video IDs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _collect(roots: list[str], label: int, dataset: str, extensions: set[str], output: Path, absolute: bool):
    records = []
    for root_index, value in enumerate(roots):
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        files = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        for path in files:
            relative_parent = path.parent.relative_to(root).as_posix()
            if relative_parent == ".":
                relative_parent = path.stem
            video_id = f"{dataset}:{label}:{root_index}:{relative_parent}"
            image_path = str(path) if absolute else os.path.relpath(path, output.parent)
            records.append({
                "image_path": image_path.replace("\\", "/"),
                "label": label,
                "video_id": video_id,
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="例如 wilddeepfake")
    parser.add_argument("--real-root", action="append", required=True)
    parser.add_argument("--fake-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.webp")
    parser.add_argument("--absolute-paths", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    extensions = {
        value.strip().lower() if value.strip().startswith(".") else f".{value.strip().lower()}"
        for value in args.extensions.split(",") if value.strip()
    }
    records = _collect(
        args.real_root, 0, args.dataset, extensions, output, args.absolute_paths
    ) + _collect(
        args.fake_root, 1, args.dataset, extensions, output, args.absolute_paths
    )
    if not records or not any(item["label"] == 0 for item in records) or not any(item["label"] == 1 for item in records):
        raise RuntimeError("索引必须同时包含真实和伪造图像")
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    real_count = sum(item["label"] == 0 for item in records)
    fake_count = len(records) - real_count
    video_count = len({item["video_id"] for item in records})
    print(f"[SAVED] {output} images={len(records)} real={real_count} fake={fake_count} videos={video_count}")


if __name__ == "__main__":
    main()
