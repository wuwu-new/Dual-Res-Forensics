"""统一的 DRF 训练入口。

训练阶段只使用 ``data.val_jsons`` 选择 best checkpoint；最终的
``data.test_jsons`` 必须通过 ``tools.test`` 单独评估，避免测试集泄漏。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drf.core import build_model
from drf.data import ForgeryDataset, collate_fn
from drf.engine import Trainer, build_optimizer_and_scheduler, pick_device, set_seed


def _path(value: str) -> str:
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else PROJECT_ROOT / p)


def _loader(json_path: str, data_cfg: dict, mode: str, shuffle: bool) -> DataLoader:
    ds = ForgeryDataset(
        _path(json_path),
        image_size=int(data_cfg["image_size"]),
        mode=mode,
        augment_strength=float(data_cfg.get("augment_strength", 1.0)),
        normalization=str(data_cfg.get("normalization", "clip")),
    )


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return DataLoader(
        ds,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(data_cfg.get("num_workers", 4)) > 0,
        collate_fn=collate_fn,
        drop_last=(mode == "train"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DRF with a validation-only model selection loop")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    set_seed(args.seed, deterministic=True)
    with open(_path(args.config), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg["data"].get("val_jsons"):
        raise ValueError("配置必须提供 data.val_jsons，不能使用测试集选择 best checkpoint")

    experiment_dir = Path(_path(os.path.join(
        cfg["experiment"]["log_dir"], cfg["experiment"]["name"]
    )))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(_path(args.config)).resolve()
    (experiment_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest_path = experiment_dir / "run_manifest.json"
    manifest = {
        "experiment": cfg["experiment"]["name"],
        "status": "running",
        "started_at": _now(),
        "config": str(config_path),
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "resume": str(Path(_path(args.resume)).resolve()) if args.resume else None,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    _save_json(manifest_path, manifest)
    metrics_log = experiment_dir / "metrics.log"

    def log(message) -> None:
        line = f"[{_now()}] {message}"
        print(line)
        with metrics_log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    try:
        device = pick_device()
        model = build_model(cfg).to(device)
        train_loader = _loader(cfg["data"]["train_json"], cfg["data"], "train", True)
        val_loaders = {
            name: _loader(path, cfg["data"], "test", False)
            for name, path in cfg["data"]["val_jsons"].items()
        }
        manifest.update({
            "train_samples": len(train_loader.dataset),
            "validation_samples": {
                name: len(loader.dataset) for name, loader in val_loaders.items()
            },
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        })
        _save_json(manifest_path, manifest)

        accum = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
        steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
        train_steps = steps_per_epoch * int(cfg["train"]["num_epochs"])
        opt, sched = build_optimizer_and_scheduler(
            model.trainable_parameters(),
            lr=float(cfg["optim"]["lr"]),
            weight_decay=float(cfg["optim"]["weight_decay"]),
            total_steps=train_steps,
            warmup_steps=max(1, int(train_steps * float(cfg["optim"].get("warmup_ratio", 0.05)))),
            min_lr_ratio=float(cfg["optim"].get("min_lr_ratio", 0.01)),
        )
        log_dir = str(experiment_dir / "ckpt")
        trainer = Trainer(
            model, opt, sched, device, log_dir, logger=log,
            use_amp=bool(cfg["train"].get("use_amp", True)),
            grad_accum_steps=accum,
            grad_clip_norm=float(cfg["train"].get("grad_clip_norm", 1.0)),
            ema_decay=float(cfg["train"].get("ema_decay", 0.999)),
            contrastive_weight=float(cfg["loss"].get("contrastive_weight", 0.0)),
            contrastive_temperature=float(cfg["loss"].get("contrastive_temperature", 0.1)),
            num_hard_neg=int(cfg["loss"].get("num_hard_neg", 16)),
            label_smoothing=float(cfg["loss"].get("label_smoothing", 0.05)),
        )
        start_epoch = 1
        if args.resume:
            start_epoch = trainer.load_checkpoint(_path(args.resume))
        trainer.fit(train_loader, val_loaders, int(cfg["train"]["num_epochs"]), start_epoch)
    except BaseException as exc:
        manifest.update({"status": "failed", "ended_at": _now(), "error": repr(exc)})
        _save_json(manifest_path, manifest)
        raise
    manifest.update({
        "status": "complete",
        "ended_at": _now(),
        "best_validation_auc": trainer.best_avg_auc,
        "best_checkpoint": str(Path(log_dir) / "ckpt_best.pth"),
    })
    _save_json(manifest_path, manifest)
    log(f"[DONE] validation-selected checkpoint: {Path(log_dir) / 'ckpt_best.pth'}")


if __name__ == "__main__":
    main()
