"""Build FF++ C23 frame indexes from the official video-level splits."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitSummary:
    split: str
    images: int
    real_images: int
    fake_images: int
    videos: int
    missing_directories: tuple[str, ...]


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{path}: pair {index} must contain exactly two IDs")
        first, second = (str(value) for value in item)
        if not first or not second or first == second:
            raise ValueError(f"{path}: invalid pair at index {index}: {item!r}")
        pairs.append((first, second))

    source_ids = [value for pair in pairs for value in pair]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{path}: a source video ID occurs more than once")
    return pairs


def _validate_disjoint(splits: dict[str, list[tuple[str, str]]]) -> None:
    source_sets = {
        name: {value for pair in pairs for value in pair}
        for name, pairs in splits.items()
    }
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1 :]:
            overlap = source_sets[first] & source_sets[second]
            if overlap:
                shown = ", ".join(sorted(overlap)[:10])
                raise ValueError(f"source video leakage between {first} and {second}: {shown}")


def _frames(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _image_path(path: Path, output: Path, absolute: bool) -> str:
    value = str(path) if absolute else os.path.relpath(path, output.parent)
    return value.replace("\\", "/")


def _append_video(
    records: list[dict],
    directory: Path,
    label: int,
    video_id: str,
    output: Path,
    extensions: set[str],
    absolute: bool,
) -> bool:
    frames = _frames(directory, extensions)
    if not frames:
        return False
    records.extend(
        {
            "image_path": _image_path(path, output, absolute),
            "label": label,
            "video_id": video_id,
        }
        for path in frames
    )
    return True


def build_split(
    root: Path,
    split: str,
    pairs: list[tuple[str, str]],
    output: Path,
    methods: tuple[str, ...],
    extensions: set[str],
    absolute: bool = False,
    strict: bool = False,
) -> SplitSummary:
    records: list[dict] = []
    missing: list[str] = []
    real_root = root / "original_sequences" / "youtube" / "c23" / "frames"

    for source_id in (value for pair in pairs for value in pair):
        directory = real_root / source_id
        if not _append_video(
            records,
            directory,
            0,
            f"ffpp:real:{source_id}",
            output,
            extensions,
            absolute,
        ):
            missing.append(str(directory))

    for method in methods:
        method_root = root / "manipulated_sequences" / method / "c23" / "frames"
        for first, second in pairs:
            for video_name in (f"{first}_{second}", f"{second}_{first}"):
                directory = method_root / video_name
                if not _append_video(
                    records,
                    directory,
                    1,
                    f"ffpp:{method}:{video_name}",
                    output,
                    extensions,
                    absolute,
                ):
                    missing.append(str(directory))

    if strict and missing:
        shown = "\n".join(f"  {item}" for item in missing[:20])
        raise FileNotFoundError(f"{split}: missing FF++ frame directories:\n{shown}")
    if not records or not {item["label"] for item in records} == {0, 1}:
        raise RuntimeError(f"{split}: index must contain real and fake frames")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    real_images = sum(item["label"] == 0 for item in records)
    video_count = len({item["video_id"] for item in records})
    return SplitSummary(
        split=split,
        images=len(records),
        real_images=real_images,
        fake_images=len(records) - real_images,
        videos=video_count,
        missing_directories=tuple(missing),
    )


def build_indexes(
    root: Path,
    output_dir: Path,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    extensions: set[str] | None = None,
    absolute: bool = False,
    strict: bool = False,
) -> list[SplitSummary]:
    root = root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    split_pairs = {name: _read_pairs(root / f"{name}.json") for name in SPLITS}
    _validate_disjoint(split_pairs)

    resolved_extensions = extensions or DEFAULT_EXTENSIONS
    summaries = []
    for split in SPLITS:
        output = output_dir / f"ffpp_c23_{split}.json"
        summaries.append(
            build_split(
                root,
                split,
                split_pairs[split],
                output,
                methods,
                resolved_extensions,
                absolute,
                strict,
            )
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Extracted FaceForensics++ root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.webp")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of reporting frame directories absent from the release",
    )
    args = parser.parse_args()

    extensions = {
        value.strip().lower() if value.strip().startswith(".") else f".{value.strip().lower()}"
        for value in args.extensions.split(",")
        if value.strip()
    }
    summaries = build_indexes(
        Path(args.root),
        Path(args.output_dir),
        tuple(args.methods),
        extensions,
        args.absolute_paths,
        args.strict,
    )
    for summary in summaries:
        print(
            f"[SAVED] ffpp_c23_{summary.split}.json images={summary.images} "
            f"real={summary.real_images} fake={summary.fake_images} "
            f"videos={summary.videos} missing_dirs={len(summary.missing_directories)}"
        )
        for directory in summary.missing_directories:
            print(f"  [MISSING] {directory}")


if __name__ == "__main__":
    main()
