"""Pack only files referenced by dataset indexes into a portable tar archive."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path


def _inside(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"indexed file is outside dataset root: {path}") from exc


def collect_files(root: Path, indexes: list[Path]) -> tuple[list[tuple[Path, Path]], dict]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    files: dict[Path, Path] = {}
    index_stats = []
    for value in indexes:
        index = value.expanduser().resolve()
        index_relative = _inside(index, root)
        records = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{index}: expected a JSON list")
        files[index_relative] = index
        for row_number, record in enumerate(records):
            image_value = record.get("image_path", record.get("image"))
            if not image_value:
                raise ValueError(f"{index}: row {row_number} has no image path")
            image = Path(image_value).expanduser()
            if not image.is_absolute():
                image = index.parent / image
            if not image.is_file():
                raise FileNotFoundError(image)
            files[_inside(image, root)] = image.resolve()
        index_stats.append({"index": index_relative.as_posix(), "records": len(records)})

    ordered = sorted(files.items(), key=lambda item: item[0].as_posix())
    manifest = {
        "format": 1,
        "root_name": root.name,
        "indexes": index_stats,
        "files": len(ordered),
        "payload_bytes": sum(source.stat().st_size for _, source in ordered),
    }
    return [(source, relative) for relative, source in ordered], manifest


def create_archive(root: Path, indexes: list[Path], output: Path) -> dict:
    files, manifest = collect_files(root, indexes)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {output}")

    with tarfile.open(output, mode="w") as archive:
        for source, relative in files:
            archive.add(source, arcname=relative.as_posix(), recursive=False)
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo("PACKAGE_MANIFEST.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root stored at archive root")
    parser.add_argument("--index", action="append", required=True, help="Index inside root; repeatable")
    parser.add_argument("--output", required=True, help="Destination .tar path")
    args = parser.parse_args()

    manifest = create_archive(
        Path(args.root),
        [Path(value) for value in args.index],
        Path(args.output),
    )
    print(
        f"[SAVED] {Path(args.output).expanduser().resolve()} files={manifest['files']} "
        f"payload_gib={manifest['payload_bytes'] / (1024 ** 3):.2f}"
    )


if __name__ == "__main__":
    main()
