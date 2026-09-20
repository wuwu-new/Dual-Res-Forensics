"""Create a reproducible manifest for all experiment JSON indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsons", nargs="+")
    parser.add_argument("--out", default="logs/data_manifest.json")
    args = parser.parse_args()

    files = []
    for value in args.jsons:
        path = Path(value).expanduser().resolve()
        samples = json.loads(path.read_text(encoding="utf-8"))
        labels = Counter(int(item["label"]) for item in samples)
        videos = {str(item.get("video_id")) for item in samples if item.get("video_id") not in (None, "")}
        files.append({
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "frames": len(samples),
            "real_frames": labels.get(0, 0),
            "fake_frames": labels.get(1, 0),
            "videos": len(videos),
        })

    output = Path(args.out).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "files": files,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {output} files={len(files)}")


if __name__ == "__main__":
    main()
