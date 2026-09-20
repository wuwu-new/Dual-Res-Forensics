"""
DRF v2 评估入口
==================
用法:
  python -m tools.test --config configs/default.yaml --ckpt logs/<name>/ckpt/ckpt_best.pth

行为:
  - 加载 ckpt (优先 ema_state, 没有则用 model_state)
  - 按 config.data.test_jsons 在每个测试集上跑一遍
  - 终端打印 + 保存到 logs/<name>/test_result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drf.core import build_model
from drf.data import ForgeryDataset, collate_fn
from drf.engine import pick_device, safe_load_checkpoint, set_seed
from drf.metrics import MetricMeter, compute_video_metrics
import torchvision.transforms.functional as TF


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_compact_state(model, state: dict, source: str) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Compact checkpoints intentionally omit the frozen CLIP parameters. Every
    # trainable DRF parameter must still be present or the result is invalid.
    bad_missing = [key for key in missing if not key.startswith("clip.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"{source} 与当前配置不兼容: missing={bad_missing[:10]}, "
            f"unexpected={unexpected[:10]}。请确认 checkpoint 与 config 成对使用。"
        )


def _build_test_loaders(cfg):
    d = cfg["data"]
    loaders = {}
    for name, jp in d["test_jsons"].items():
        jp = str(_resolve(jp))
        ts = ForgeryDataset(
            jp,
            image_size=d["image_size"],
            mode="test",
            normalization=str(d.get("normalization", "clip")),
        )
        loaders[name] = DataLoader(
            ts, batch_size=d["batch_size"], shuffle=False,
            num_workers=d["num_workers"], collate_fn=collate_fn, pin_memory=True,
        )
    return loaders


@torch.no_grad()
def _evaluate(model, loader, device, use_amp: bool, name: str, tta: bool = False, save_predictions: bool = False):
    model.eval()
    meter = MetricMeter()
    predictions = [] if save_predictions else None  # Store per-sample predictions
    video_labels, video_scores, video_ids = [], [], []

    for batch in tqdm(loader, desc=f"Eval [{name}]"):
        images = batch["image"].to(device, non_blocking=True)
        B, C, H, W = images.shape
        labels_batch = batch["label"].numpy()

        if not tta:
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                out = model(images)
            prob_fake = torch.sigmoid(out["logit"]).float().cpu().numpy()
            meter.update(labels_batch, prob_fake)
            video_labels.extend(labels_batch.tolist())
            video_scores.extend(prob_fake.tolist())
            batch_video_ids = batch.get("video_id", [""] * B)
            video_ids.extend(batch_video_ids)
            
            if save_predictions:
                for i in range(B):
                    predictions.append({
                        "dataset": name,
                        "image_path": batch["image_path"][i],
                        "video_id": batch_video_ids[i],
                        "label": int(labels_batch[i]),
                        "pred_score": float(prob_fake[i])
                    })
            continue

        # TTA: horizontal flip + 5-crop (4 corners + center) averaged
        probs = []
        for i in range(B):
            img = images[i]
            variants = []

            # original and horizontal flip
            variants.append(img)
            variants.append(TF.hflip(img))

            # five crops on original and hflip
            crop_ratio = 0.9
            ch = max(1, int(H * crop_ratio))
            cw = max(1, int(W * crop_ratio))
            positions = [
                (0, 0),
                (0, W - cw),
                (H - ch, 0),
                (H - ch, W - cw),
                ((H - ch) // 2, (W - cw) // 2),
            ]

            for base in (img, TF.hflip(img)):
                for y, x in positions:
                    try:
                        c = TF.crop(base, y, x, ch, cw)
                    except Exception:
                        # fallback: use full image if crop fails
                        c = base
                    c = TF.resize(c, [H, W])
                    variants.append(c)

            # stack variants and run model
            v = torch.stack(variants, dim=0).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                out_v = model(v)
            logit_v = out_v["logit"].float().cpu()
            prob_v = torch.sigmoid(logit_v).numpy()
            avg_prob = float(prob_v.mean())
            probs.append(avg_prob)

        prob_fake = np.array(probs)
        meter.update(labels_batch, prob_fake)
        video_labels.extend(labels_batch.tolist())
        video_scores.extend(prob_fake.tolist())
        batch_video_ids = batch.get("video_id", [""] * B)
        video_ids.extend(batch_video_ids)
        
        if save_predictions:
            for i in range(B):
                predictions.append({
                    "dataset": name,
                    "image_path": batch["image_path"][i],
                    "video_id": batch_video_ids[i],
                    "label": int(labels_batch[i]),
                    "pred_score": float(prob_fake[i])
                })

    metrics = meter.compute()
    metrics["num_frames"] = len(video_labels)
    if any(video_ids):
        metrics["video"] = compute_video_metrics(video_labels, video_scores, video_ids)
    if save_predictions:
        metrics["_predictions"] = predictions
    return metrics


def main():
    ap = argparse.ArgumentParser(description="DRF v2 Eval Entry")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True, help="checkpoint path (.pth)")
    ap.add_argument("--prefer", choices=["ema", "model"], default="model",
                    help="加载原始模型权重 (默认) 或 EMA 权重")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tta", action="store_true", help="Enable test-time augmentation: flip + 5-crop ensemble")
    ap.add_argument("--out", default=None, help="JSON 结果输出路径 (默认 logs/<name>/test_result.json)")
    ap.add_argument("--save-predictions", action="store_true", 
                    help="Save per-sample predictions for visualization")
    args = ap.parse_args()

    set_seed(args.seed)
    config_path = _resolve(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = pick_device()
    model = build_model(cfg).to(device)

    ckpt_path = _resolve(args.ckpt)
    ckpt = safe_load_checkpoint(str(ckpt_path), map_location=device)
    state_key = "ema_state" if (args.prefer == "ema" and ckpt.get("ema_state")) else "model_state"
    _load_compact_state(model, ckpt[state_key], f"{ckpt_path}:{state_key}")
    print(f"[Load] {state_key} from {ckpt_path}")

    use_amp = cfg["train"].get("use_amp", True)
    loaders = _build_test_loaders(cfg)

    results = {}
    aucs = []
    all_predictions = []
    
    print("\n" + "=" * 72)
    print(f"{'Dataset':<14}{'AUC':>8}{'AP':>8}{'EER(%)':>10}{'ACC':>8}{'F1*':>8}{'thr*':>8}")
    print("-" * 72)
    for name, loader in loaders.items():
        m = _evaluate(model, loader, device, use_amp, name, tta=args.tta, 
                      save_predictions=args.save_predictions)
        
        # Extract predictions if saved
        predictions = m.pop("_predictions", None)
        if predictions:
            all_predictions.extend(predictions)
        
        results[name] = m
        if m["auc"] == m["auc"]:                 # not nan
            aucs.append(m["auc"])
        print(f"{name:<14}{m['auc']:>8.4f}{m['ap']:>8.4f}"
              f"{m['eer']*100:>10.2f}{m['acc']:>8.4f}"
              f"{m['best_f1']:>8.4f}{m['best_thr']:>8.4f}")
    avg_auc = sum(aucs) / len(aucs) if aucs else float("nan")
    print("-" * 72)
    print(f"{'Average AUC':<14}{avg_auc:>8.4f}")
    print("=" * 72)

    out_path = _resolve(args.out) if args.out else _resolve(os.path.join(
        cfg["experiment"]["log_dir"], cfg["experiment"]["name"], "test_result.json"
    ))
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"avg_auc": avg_auc, "per_dataset": results,
                   "experiment": cfg["experiment"]["name"],
                   "ckpt": str(ckpt_path), "config": str(config_path),
                   "loaded_from": state_key}, f, indent=2)
    print(f"\n[Saved] {out_path}")
    
    # Save per-sample predictions if requested
    if args.save_predictions and all_predictions:
        pred_path = out_path.replace(".json", "_detailed.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=2)
        print(f"[Saved] {pred_path} ({len(all_predictions)} samples)")


if __name__ == "__main__":
    main()
