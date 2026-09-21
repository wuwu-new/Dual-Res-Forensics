"""List and validate the dataset layout without redistributing licensed data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "datasets.yaml"


def _load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
    datasets = registry.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"dataset registry is empty: {path}")
    return registry


def _selected(registry: dict[str, Any], include_optional: bool):
    for key, item in registry["datasets"].items():
        if include_optional or item.get("tier") == "required":
            yield key, item


def _print_list(registry: dict[str, Any], include_optional: bool) -> None:
    print(f"{'KEY':<22} {'TIER':<10} {'ROLE':<36} NAME")
    for key, item in _selected(registry, include_optional):
        print(
            f"{key:<22} {item['tier']:<10} {item['role']:<36} {item['name']}"
        )


def _print_sources(registry: dict[str, Any], dataset_key: str | None) -> None:
    datasets = registry["datasets"]
    if dataset_key is not None and dataset_key not in datasets:
        raise KeyError(f"unknown dataset: {dataset_key}")
    keys = [dataset_key] if dataset_key else list(datasets)
    for key in keys:
        item = datasets[key]
        print(f"[{key}] {item['name']} ({item['access']})")
        print(f"  scope: {item['contents']}")
        for url in item["official_urls"]:
            print(f"  - {url}")


def _scaffold(registry: dict[str, Any], root: Path, include_optional: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for key, item in _selected(registry, include_optional):
        path = root / item["directory"]
        path.mkdir(parents=True, exist_ok=True)
        print(f"[READY] {key}: {path}")


def _marker_exists(dataset_root: Path, marker: str) -> bool:
    if any(char in marker for char in "*?["):
        return any(dataset_root.glob(marker))
    return (dataset_root / marker).exists()


def _doctor(registry: dict[str, Any], root: Path, include_optional: bool) -> int:
    failures = 0
    for key, item in _selected(registry, include_optional):
        dataset_root = root / item["directory"]
        missing = [
            marker for marker in item.get("markers", [])
            if not _marker_exists(dataset_root, marker)
        ]
        if not dataset_root.is_dir():
            print(f"[MISSING] {key}: {dataset_root}")
            failures += 1
        elif missing:
            print(f"[INCOMPLETE] {key}: missing {', '.join(missing)}")
            failures += 1
        elif not any(dataset_root.iterdir()):
            print(f"[EMPTY] {key}: {dataset_root}")
            failures += 1
        else:
            print(f"[FOUND] {key}: {dataset_root}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DRF dataset registry, source links, and local layout checks"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--include-optional", action="store_true")

    source_parser = subparsers.add_parser("sources")
    source_parser.add_argument("dataset", nargs="?")

    scaffold_parser = subparsers.add_parser("scaffold")
    scaffold_parser.add_argument("--root", type=Path, required=True)
    scaffold_parser.add_argument("--include-optional", action="store_true")

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--root", type=Path, required=True)
    doctor_parser.add_argument("--include-optional", action="store_true")

    args = parser.parse_args()
    registry = _load_registry(args.registry.expanduser().resolve())
    if args.command == "list":
        _print_list(registry, args.include_optional)
    elif args.command == "sources":
        _print_sources(registry, args.dataset)
    elif args.command == "scaffold":
        _scaffold(registry, args.root.expanduser().resolve(), args.include_optional)
    elif args.command == "doctor":
        raise SystemExit(
            1 if _doctor(
                registry, args.root.expanduser().resolve(), args.include_optional
            ) else 0
        )


if __name__ == "__main__":
    main()
