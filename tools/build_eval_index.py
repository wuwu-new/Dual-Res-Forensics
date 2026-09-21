"""Build metadata-driven evaluation indexes for Celeb-DF v2 and DFDC."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class IndexSummary:
    dataset: str
    images: int
    real_images: int
    fake_images: int
    videos: int
    missing_videos: tuple[str, ...]


def _frames(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _path_for_index(path: Path, output: Path, absolute: bool) -> str:
    value = str(path) if absolute else os.path.relpath(path, output.parent)
    return value.replace("\\", "/")


def _append_video(
    records: list[dict],
    frame_dir: Path,
    label: int,
    video_id: str,
    output: Path,
    extensions: set[str],
    absolute: bool,
) -> bool:
    frames = _frames(frame_dir, extensions)
    if not frames:
        return False
    records.extend(
        {
            "image_path": _path_for_index(path, output, absolute),
            "label": label,
            "video_id": video_id,
        }
        for path in frames
    )
    return True


def _save(
    records: list[dict],
    missing: list[str],
    dataset: str,
    output: Path,
    strict: bool,
) -> IndexSummary:
    if strict and missing:
        shown = "\n".join(f"  {item}" for item in missing[:20])
        raise FileNotFoundError(f"{dataset}: videos without extracted frames:\n{shown}")
    if not records or {item["label"] for item in records} != {0, 1}:
        raise RuntimeError(f"{dataset}: index must contain real and fake frames")
    video_ids = [item["video_id"] for item in records]
    if any(not value for value in video_ids):
        raise RuntimeError(f"{dataset}: empty video_id generated")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    real_images = sum(item["label"] == 0 for item in records)
    return IndexSummary(
        dataset=dataset,
        images=len(records),
        real_images=real_images,
        fake_images=len(records) - real_images,
        videos=len(set(video_ids)),
        missing_videos=tuple(missing),
    )


def build_celebdf_v2(
    root: Path,
    output: Path,
    extensions: set[str] | None = None,
    absolute: bool = False,
    strict: bool = False,
) -> IndexSummary:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    test_list = root / "List_of_testing_videos.txt"
    if not test_list.is_file():
        raise FileNotFoundError(test_list)

    records: list[dict] = []
    missing: list[str] = []
    seen: set[str] = set()
    resolved_extensions = extensions or DEFAULT_EXTENSIONS
    for line_number, raw_line in enumerate(test_list.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] not in {"0", "1"}:
            raise ValueError(f"{test_list}:{line_number}: invalid entry: {raw_line!r}")
        official_label, relative_video = fields
        relative_path = Path(relative_video)
        video_key = relative_path.with_suffix("").as_posix()
        if video_key in seen:
            raise ValueError(f"{test_list}:{line_number}: duplicate video: {relative_video}")
        seen.add(video_key)

        # The official list uses 1=real and 0=fake; DRF uses 0=real and 1=fake.
        label = 0 if official_label == "1" else 1
        frame_dir = root / relative_path.parent / "frames" / relative_path.stem
        if not _append_video(
            records,
            frame_dir,
            label,
            f"celebdfv2:{video_key}",
            output,
            resolved_extensions,
            absolute,
        ):
            missing.append(relative_video)

    return _save(records, missing, "celebdfv2", output, strict)


def build_dfdc(
    root: Path,
    output: Path,
    extensions: set[str] | None = None,
    absolute: bool = False,
    strict: bool = False,
) -> IndexSummary:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: expected an object keyed by video filename")

    records: list[dict] = []
    missing: list[str] = []
    resolved_extensions = extensions or DEFAULT_EXTENSIONS
    for filename in sorted(metadata):
        item = metadata[filename]
        label = item.get("is_fake") if isinstance(item, dict) else None
        if label not in (0, 1, False, True):
            raise ValueError(f"{metadata_path}: {filename} has invalid is_fake={label!r}")
        video_stem = Path(filename).stem
        frame_dir = root / "frames" / video_stem
        if not _append_video(
            records,
            frame_dir,
            int(label),
            f"dfdc:{video_stem}",
            output,
            resolved_extensions,
            absolute,
        ):
            missing.append(filename)

    return _save(records, missing, "dfdc", output, strict)


def _print_summary(summary: IndexSummary, output: Path) -> None:
    print(
        f"[SAVED] {output} images={summary.images} real={summary.real_images} "
        f"fake={summary.fake_images} videos={summary.videos} "
        f"missing_videos={len(summary.missing_videos)}"
    )
    for video in summary.missing_videos[:20]:
        print(f"  [MISSING] {video}")
    if len(summary.missing_videos) > 20:
        print(f"  ... and {len(summary.missing_videos) - 20} more")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("celebdf-v2", "dfdc"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.webp")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    extensions = {
        value.strip().lower() if value.strip().startswith(".") else f".{value.strip().lower()}"
        for value in args.extensions.split(",")
        if value.strip()
    }
    root = Path(args.root)
    output = Path(args.output)
    if args.dataset == "celebdf-v2":
        summary = build_celebdf_v2(root, output, extensions, args.absolute_paths, args.strict)
    else:
        summary = build_dfdc(root, output, extensions, args.absolute_paths, args.strict)
    _print_summary(summary, output.expanduser().resolve())


if __name__ == "__main__":
    main()
