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

from drf.core import DRFModel
from drf.data import ForgeryDataset, collate_fn
from drf.engine import pick_device, safe_load_checkpoint, set_seed
from drf.metrics import MetricMeter
import torchvision.transforms.functional as TF


def _build_test_loaders(cfg):
    d = cfg["data"]
    loaders = {}
    for name, jp in d["test_jsons"].items():
        ts = ForgeryDataset(jp, image_size=d["image_size"], mode="test")
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

    for batch in tqdm(loader, desc=f"Eval [{name}]"):
        images = batch["image"].to(device, non_blocking=True)
        B, C, H, W = images.shape
        labels_batch = batch["label"].numpy()

        if not tta:
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                out = model(images)
            prob_fake = torch.sigmoid(out["logit"]).float().cpu().numpy()
            meter.update(labels_batch, prob_fake)
            
            if save_predictions:
                for i in range(B):
                    predictions.append({
                        "dataset": name,
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
        
        if save_predictions:
            for i in range(B):
                predictions.append({
                    "dataset": name,
                    "label": int(labels_batch[i]),
                    "pred_score": float(prob_fake[i])
                })

    metrics = meter.compute()
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
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = pick_device()
    b = cfg["backbone"]; r = cfg["residual"]; fu = cfg["fusion"]
    model = DRFModel(
        clip_name=b["clip_name"],
        c_token=r["c_token"], token_grid=r["token_grid"],
        fused_dim=fu["fused_dim"], fusion_embed_dim=fu["embed_dim"],
        fusion_num_heads=fu["num_heads"], dropout=fu.get("dropout", 0.1),
        use_boundary_head=cfg["model"].get("use_boundary_head", False),
        freeze_clip=b.get("freeze_clip", True),
        use_srm=r.get("use_srm", True),
        use_gate=fu.get("use_gate", True),
        use_freq=r.get("use_freq", False),
        freq_c_token=r.get("freq_c_token", r["c_token"]),
        freq_token_grid=r.get("freq_token_grid", r["token_grid"]),
    ).to(device)

    ckpt = safe_load_checkpoint(args.ckpt, map_location=device)
    state_key = "ema_state" if (args.prefer == "ema" and ckpt.get("ema_state")) else "model_state"
    missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
    print(f"[Load] {state_key} from {args.ckpt}  "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")

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

    out_path = args.out or os.path.join(
        cfg["experiment"]["log_dir"], cfg["experiment"]["name"], "test_result.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"avg_auc": avg_auc, "per_dataset": results,
                   "ckpt": args.ckpt, "loaded_from": state_key}, f, indent=2)
    print(f"\n[Saved] {out_path}")
    
    # Save per-sample predictions if requested
    if args.save_predictions and all_predictions:
        pred_path = out_path.replace(".json", "_detailed.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=2)
        print(f"[Saved] {pred_path} ({len(all_predictions)} samples)")


if __name__ == "__main__":
    main()
