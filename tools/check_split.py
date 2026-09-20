"""Reject image- or video-level leakage between training and validation JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _keys(path: Path) -> tuple[set[str], set[str]]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    images, videos = set(), set()
    for index, sample in enumerate(samples):
        image = sample.get("image_path", sample.get("image"))
        video = sample.get("video_id")
        if not image:
            raise ValueError(f"{path}: sample {index} 缺少 image_path/image")
        if video in (None, ""):
            raise ValueError(f"{path}: sample {index} 缺少 video_id，无法排除视频泄漏")
        image_path = Path(image).expanduser()
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        images.add(str(image_path.resolve()))
        videos.add(str(video))
    return images, videos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_json")
    parser.add_argument("val_json")
    args = parser.parse_args()
    train_path = Path(args.train_json).expanduser().resolve()
    val_path = Path(args.val_json).expanduser().resolve()
    train_images, train_videos = _keys(train_path)
    val_images, val_videos = _keys(val_path)
    image_overlap = train_images & val_images
    video_overlap = train_videos & val_videos
    print(
        f"train: images={len(train_images)} videos={len(train_videos)} | "
        f"val: images={len(val_images)} videos={len(val_videos)}"
    )
    if image_overlap or video_overlap:
        print(f"[FAIL] image_overlap={len(image_overlap)} video_overlap={len(video_overlap)}")
        for value in sorted(image_overlap)[:5]:
            print(f"  repeated image: {value}")
        for value in sorted(video_overlap)[:5]:
            print(f"  repeated video_id: {value}")
        raise SystemExit(1)
    print("[OK] train/validation 在图像和视频级均无交集")


if __name__ == "__main__":
    main()
