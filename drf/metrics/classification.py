"""
分类指标
==========
相对旧版 compute_metrics 函数:
  - 改成累积式 MetricMeter 类 (支持多 batch 累积一起算)
  - 在 AUC/AP/EER 之外, 增加 ACC@0.5 和 Best-F1 (扫描阈值取最大 F1)
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
)


class MetricMeter:
    def __init__(self):
        self._y: list = []
        self._s: list = []

    def update(self, y_true, y_score):
        self._y.append(np.asarray(y_true).reshape(-1).astype(int))
        self._s.append(np.asarray(y_score).reshape(-1).astype(float))

    def _stack(self):
        return (np.concatenate(self._y) if self._y else np.zeros(0, int),
                np.concatenate(self._s) if self._s else np.zeros(0, float))

    def compute(self) -> Dict[str, float]:
        y, s = self._stack()
        nan = float("nan")
        if y.size == 0 or len(np.unique(y)) < 2:
            return {"auc": nan, "ap": nan, "eer": nan,
                    "acc": nan, "best_f1": nan, "best_thr": nan}

        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)

        fpr, tpr, _ = roc_curve(y, s)
        fnr = 1.0 - tpr
        idx = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)

        acc = float(((s >= 0.5) == (y == 1)).mean())

        prec, rec, thr = precision_recall_curve(y, s)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        f1 = f1[:-1]                                       # 与 thr 对齐
        if f1.size > 0:
            j = int(np.nanargmax(f1))
            best_f1 = float(f1[j])
            best_thr = float(thr[j])
        else:
            best_f1, best_thr = nan, nan

        return {"auc": float(auc), "ap": float(ap), "eer": eer,
                "acc": acc, "best_f1": best_f1, "best_thr": best_thr}
