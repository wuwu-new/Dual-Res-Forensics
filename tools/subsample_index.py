"""Create a deterministic, evenly sampled frame index per video."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def evenly_spaced(records: list[dict], frames_per_video: int) -> list[dict]:
    if frames_per_video < 1:
        raise ValueError("frames_per_video must be positive")
    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for index, record in enumerate(records):
        video_id = str(record.get("video_id", ""))
        if not video_id:
            raise ValueError(f"record {index} has no video_id")
        grouped.setdefault(video_id, []).append(record)

    sampled = []
    for video_records in grouped.values():
        count = len(video_records)
        if count <= frames_per_video:
            sampled.extend(video_records)
            continue
        if frames_per_video == 1:
            positions = [count // 2]
        else:
            positions = [
                (index * (count - 1)) // (frames_per_video - 1)
                for index in range(frames_per_video)
            ]
        sampled.extend(video_records[position] for position in positions)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-per-video", type=int, required=True)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    records = json.loads(source.read_text(encoding="utf-8"))
    sampled = evenly_spaced(records, args.frames_per_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[SAVED] {output} images={len(sampled)} "
        f"videos={len({item['video_id'] for item in sampled})}"
    )


if __name__ == "__main__":
    main()
