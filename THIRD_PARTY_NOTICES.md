# Third-Party Notices

本项目 (DRF v2) 在科研思想与部分工程实践上参考/借鉴了以下开源工作。
本文件用于声明归因；除非特别说明，本仓库未直接复制下列项目的源代码。

---

## 1. Forensics Adapter (CVPR 2025)
- 仓库: https://github.com/OUC-VAS/ForensicsAdapter
- 论文: Cui et al., "Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection", CVPR 2025
- 借鉴: 冻结 CLIP 主干 + 轻量 Adapter 辅助通道 + 样本级对比辅助损失 的整体思想。
- 本项目实现: SRM 残差通道、深度可分离残差编码器、门控残差引导交叉注意力 (GRCA)、Hard-Negative SupCon 均为重新实现，非源码拷贝。

## 2. DeepfakeBench (NeurIPS 2023)
- 仓库: https://github.com/SCLBD/DeepfakeBench
- 论文: Yan et al., "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection", NeurIPS 2023
- 借鉴: 训练 / 评估的指标口径 (frame-level AUC / AP / EER)、数据切分与训练配置 (lr=2e-4, wd=5e-4, batch=16, 10 epochs) 的取值参考。
- 本项目实现: `MetricMeter` 类、JSON-driven `ForgeryDataset` 均为重新实现。

## 3. OpenAI CLIP
- 仓库: https://github.com/openai/CLIP
- 许可: MIT
- 使用方式: 通过 HuggingFace `transformers.CLIPVisionModel` 加载预训练权重，仅作为冻结视觉主干使用，本仓库不分发任何 CLIP 权重。

## 4. HuggingFace Transformers
- 仓库: https://github.com/huggingface/transformers
- 许可: Apache-2.0

## 5. Albumentations
- 仓库: https://github.com/albumentations-team/albumentations
- 许可: MIT

## 6. scikit-learn / PyTorch / NumPy / Pillow / tqdm / PyYAML
- 标准 BSD / MIT / Apache-2.0 许可的常见科学计算依赖, 用法均为标准 API 调用。

---

如发现任何遗漏归因，请在 issue 中告知，我们会在下一个版本补正。
