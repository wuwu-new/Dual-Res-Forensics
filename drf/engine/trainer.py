"""
Trainer v2
============
相对旧版 trainer.py 的关键改动:
  1) 训练循环使用 AMP autocast + GradScaler (混合精度)
  2) 支持梯度累积 (grad_accum_steps)
  3) 每 step 调度学习率 (warmup + cosine), 而非每 epoch
  4) 维护 EMA 权重, 每个 epoch 用 EMA 模型在测试集上评估
  5) 单 logit + BCE (不是 2-class CE), 评估时 sigmoid 输出 fake-prob
  6) 损失日志: total / cls / con; 评估按跨域平均 AUC 选最佳
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .ema import ModelEMA
from .losses import BinaryClsLoss, HardNegSupConLoss
from ..metrics.classification import MetricMeter


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        ckpt_dir: str,
        logger=print,
        *,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        grad_clip_norm: float = 1.0,
        ema_decay: Optional[float] = 0.999,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.1,
        num_hard_neg: int = 16,
        label_smoothing: float = 0.05,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self.logger = logger

        self.use_amp = bool(use_amp) and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.accum = max(1, int(grad_accum_steps))
        self.clip = float(grad_clip_norm)

        self.cls_loss_fn = BinaryClsLoss(label_smoothing=label_smoothing).to(device)
        self.con_w = float(contrastive_weight)
        self.con_loss_fn = HardNegSupConLoss(
            temperature=contrastive_temperature, num_hard_neg=num_hard_neg,
        ).to(device)

        self.ema = ModelEMA(model, decay=ema_decay).to(device) if ema_decay else None
        self.best_avg_auc = 0.0
        self._global_step = 0

    # ---------------------------------------------------------- train
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        running = {"loss": 0.0, "cls": 0.0, "con": 0.0}
        n_seen = 0
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Train [Ep {epoch}]")
        for it, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                out = self.model(images)
                loss_cls = self.cls_loss_fn(out["logit"], labels)
                loss = loss_cls
                loss_con = images.new_zeros(())
                if self.con_w > 0 and images.size(0) > 1:
                    # 优先用独立投影头输出 (proj_feat) 算对比损失;
                    # 没投影头时才回退用 fused_feat (会被 BCE 争夺梯度, 对比几乎不动)
                    feat_for_con = out.get("proj_feat", out["fused_feat"])
                    loss_con = self.con_loss_fn(feat_for_con, labels)
                    loss = loss + self.con_w * loss_con
                loss_to_back = loss / self.accum

            self.scaler.scale(loss_to_back).backward()

            do_step = ((it + 1) % self.accum == 0) or ((it + 1) == len(train_loader))
            if do_step:
                if self.clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)
                self._global_step += 1

            # 日志
            bs = images.size(0)
            running["loss"] += loss.item() * bs
            running["cls"]  += loss_cls.item() * bs
            running["con"]  += float(loss_con.item()) * bs
            n_seen += bs

            cur_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             cls=f"{loss_cls.item():.3f}",
                             con=f"{float(loss_con.item()):.3f}",
                             lr=f"{cur_lr:.2e}")

        return {k: v / max(n_seen, 1) for k, v in running.items()}

    # ---------------------------------------------------------- eval
    @torch.no_grad()
    def evaluate(self, test_loader, name: str = "test") -> Dict[str, float]:
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        meter = MetricMeter()
        for batch in tqdm(test_loader, desc=f"Eval [{name}]"):
            images = batch["image"].to(self.device, non_blocking=True)
            with autocast("cuda", enabled=self.use_amp):
                out = eval_model(images)
            prob_fake = torch.sigmoid(out["logit"]).float().cpu().numpy()
            meter.update(batch["label"].numpy(), prob_fake)
        return meter.compute()

    # ---------------------------------------------------------- ckpt
    def _trainable_state(self, module: torch.nn.Module) -> dict:
        """只导出可训练参数 + 所有 buffer (跳过冻结的 CLIP, 显著瘦身)。"""
        trainable_names = {n for n, p in module.named_parameters() if p.requires_grad}
        sd = module.state_dict()
        kept = {k: v for k, v in sd.items()
                if (k in trainable_names) or (k not in {n for n, _ in module.named_parameters()})}
        return kept

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool):
        ckpt = {
            "epoch": epoch,
            "model_state": self._trainable_state(self.model),
            "ema_state": (self._trainable_state(self.ema.ema)
                          if self.ema is not None else None),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (self.scheduler.state_dict()
                                if self.scheduler is not None else None),
            "scaler_state": self.scaler.state_dict(),
            "global_step": self._global_step,
            "best_avg_auc": self.best_avg_auc,
            "metrics": metrics,
        }
        # 只滚动保存 last + best, 不再每 epoch 留独立文件 (省 10GB+ 磁盘)
        torch.save(ckpt, os.path.join(self.ckpt_dir, "ckpt_last.pth"))
        if is_best:
            torch.save(ckpt, os.path.join(self.ckpt_dir, "ckpt_best.pth"))
            self.logger(f"[BEST] Ep {epoch} avg_AUC={metrics['avg_auc']:.4f}")

    def load_checkpoint(self, path: str, strict: bool = False) -> int:
        """加载 ckpt 用于续训; 返回下一个要训练的 epoch。
        默认 strict=False 因为新 ckpt 不再保存冻结的 CLIP 权重。
        """
        from .utils import safe_load_checkpoint
        ckpt = safe_load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"], strict=strict)
        if self.ema is not None and ckpt.get("ema_state") is not None:
            self.ema.ema.load_state_dict(ckpt["ema_state"], strict=strict)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler is not None and ckpt.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt.get("scaler_state") is not None:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state"])
            except Exception:
                pass
        self._global_step = int(ckpt.get("global_step", 0))
        self.best_avg_auc = float(ckpt.get("best_avg_auc", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.logger(f"[RESUME] from {path}  start_epoch={start_epoch}  "
                    f"best_avg_auc={self.best_avg_auc:.4f}")
        return start_epoch

    # ---------------------------------------------------------- fit
    def fit(self, train_loader, test_loaders, num_epochs: int, start_epoch: int = 1):
        for epoch in range(start_epoch, num_epochs + 1):
            tr = self.train_epoch(train_loader, epoch)
            self.logger(
                f"[Train Ep {epoch}] loss={tr['loss']:.4f} | "
                f"cls={tr['cls']:.4f} | con={tr['con']:.4f}"
            )

            aucs = []
            metrics_per = {}
            for name, loader in test_loaders.items():
                m = self.evaluate(loader, name=name)
                metrics_per[name] = m
                self.logger(
                    f"[Eval {name} @ Ep {epoch}] "
                    f"AUC={m['auc']:.4f} | AP={m['ap']:.4f} | "
                    f"EER={m['eer']*100:.2f}% | ACC={m['acc']:.4f} | "
                    f"F1*={m['best_f1']:.4f}"
                )
                if not np.isnan(m["auc"]):
                    aucs.append(m["auc"])

            avg_auc = float(np.mean(aucs)) if aucs else 0.0
            is_best = avg_auc > self.best_avg_auc
            if is_best:
                self.best_avg_auc = avg_auc
            self.save_checkpoint(
                epoch, {"avg_auc": avg_auc, "per_dataset": metrics_per}, is_best=is_best,
            )
