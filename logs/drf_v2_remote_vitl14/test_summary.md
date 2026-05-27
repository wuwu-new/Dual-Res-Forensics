# remote_vitl14 — Evaluation Summary

## model (默认 `--prefer model`)

| Dataset | AUC | AP | EER(%) | ACC | Best-F1 | Best-thr |
|---:|---:|---:|---:|---:|---:|---:|
| cdfv2 | 0.8701 | 0.8920 | 20.90 | 0.7728 | 0.7920 | 0.6665 |
| dfdc  | 0.6433 | 0.6806 | 39.79 | 0.6069 | 0.6673 | 0.0488 |

**Average AUC:** 0.7567

---

## TTA (`--tta` + `--prefer model`)

| Dataset | AUC | AP | EER(%) | ACC | Best-F1 | Best-thr |
|---:|---:|---:|---:|---:|---:|---:|
| cdfv2 | 0.8694 | 0.8885 | 21.18 | 0.7717 | 0.7916 | 0.5703 |
| dfdc  | 0.6483 | 0.6749 | 39.47 | 0.6019 | 0.6675 | 0.0485 |

**Average AUC:** 0.7589

---

Files produced on remote during evaluation:
- logs/drf_v2_remote_vitl14/test_result_model.json
- logs/drf_v2_remote_vitl14/test_result_tta.json

Notes:
- `model_state` 被证明为此次 checkpoint 的正确加载项（EMA 在本次 ckpt 下不适配）。
- 可将 `ckpt_best.pth` 上传到共享存储并在 `README.md` 中补充下载链接。