"""Small helpers for reproducible command-line data path overrides."""

from __future__ import annotations


def parse_named_paths(values: list[str] | None, option: str) -> dict[str, str] | None:
    if values is None:
        return None
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must use NAME=PATH, got: {value!r}")
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"{option} must use non-empty NAME=PATH, got: {value!r}")
        if name in parsed:
            raise ValueError(f"{option} contains duplicate dataset name: {name}")
        parsed[name] = path
    if not parsed:
        raise ValueError(f"{option} requires at least one NAME=PATH value")
    return parsed


def apply_data_path_overrides(
    cfg: dict,
    train_json: str | None = None,
    val_jsons: dict[str, str] | None = None,
    test_jsons: dict[str, str] | None = None,
) -> dict:
    data = cfg.setdefault("data", {})
    if train_json:
        data["train_json"] = train_json
    if val_jsons is not None:
        data["val_jsons"] = val_jsons
    if test_jsons is not None:
        data["test_jsons"] = test_jsons
    return cfg
