# Dual-Res Forensics (DRF): 边界感知与双通道残差深度伪造检测

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/Based%20on-DeepfakeBench-4A90D9" />
  <img src="https://img.shields.io/badge/CLIP-ViT--L%2F14-green" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-orange" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" />
</p>

> 本项目在 **Forensics Adapter (CVPR'25)** 与 **DeepfakeBench (NeurIPS'23)** 基础上进行了独立的模块化重构与算法升级。  
> 核心思想：冻结 CLIP 主干提取全局语义 × SRM 高频残差通道捕捉局部篡改细节 × 门控残差引导交叉注意力融合（GRCA）× Hard-Negative 监督对比辅助损失。

---

## 📌 项目背景与创新点

本项目致力于解决通用人脸伪造检测（Generalizable Face Forgery Detection）任务中的泛化性难题。针对传统高通滤波方法在细节提取上的局限性，以及现有模型容易过拟合特定操作痕迹的问题，本项目提出了 **DRF（Dual-Res Forensics）** 框架，包含以下三个核心创新：

1. **双通道特征融合架构（Dual-Channel Feature Fusion）**  
   采用冻结的 CLIP-ViT 作为主通道提取全局语义，同时并联 SRM 高频残差滤波（3 组固定噪声残差核，输出 9 通道）+ 轻量级 MobileNet 风格 DW 可分离残差编码器，捕捉局部高频篡改细节。

2. **门控残差引导交叉注意力机制（GRCA）**  
   以 SRM 残差 token 为 Query，CLIP 全局 token 序列为 Key/Value，手写 Scaled-Dot-Product Attention（Pre-LayerNorm + SwiGLU FFN + 可学习 [CLS] AttnPool + 可学习标量门控），大幅提升对隐写痕迹的检测敏感度。

3. **Hard-Negative 监督对比辅助损失（HardNeg SupCon）**  
   在训练阶段引入 A/B 图像集对比思路（FF++ 真脸为全局一致集，Deepfake 为局部不一致集），每个 anchor 只挖掘 top-k 相似度最高的难负例，迫使模型忽略压缩等常规干扰，精准锁定图像融合边缘的「属性差异过渡带」。

---

## 🧭 Pipeline

```
输入图像 (B×3×224×224)
   │
   ├──→ CLIP-ViT-L/14（冻结）─────────────────────── clip_tokens (B, N_c, 1024)
   │                                                           │ K / V
   └──→ SRMResidual（固定 9ch 高频核）                        ↓
            └──→ DWResidualEncoder（DW 可分离残差块）─ residual_tokens (B, 64, 192)
                                                         │ Q
                                                         ↓
                              GatedResidualCrossAttention（GRCA）
                              Pre-LN + SDP-Attn + SwiGLU + AttnPool + 标量门
                                                         │
                                               fused_feat (B, 128)
                                                    │           │
                              BinaryClassifierHead           HardNeg SupCon
                                (单 logit, BCE+LS)          （辅助对比损失）
                                        │
                                   真/伪 二分类
```

---

## ✨ DRF v2 vs v1 核心改动对比

| 维度 | DRF v1（中期版） | DRF v2（当前版） |
|---|---|---|
| 高频滤波 | 单个 Laplacian（3ch） | **SRM 3 组固定噪声残差核（9ch）** |
| 残差编码器 | 朴素 Conv→ReLU→AvgPool | **MobileNet 风格 DW 可分离残差块** |
| 融合层 | `nn.MultiheadAttention` + Post-LN + mean-pool | **手写 SDP-Attention + Pre-LN + SwiGLU + 可学习 [CLS] AttnPool + 标量门控** |
| 分类头 | 2-class CrossEntropy | **单 logit + BCEWithLogits + label smoothing** |
| 对比损失 | 全负例 SupCon | **Hard-Negative SupCon（top-k 难负例挖掘）** |
| 优化器 | Adam + StepLR | **AdamW + Cosine warmup（per-step）** |
| 训练循环 | 朴素 fwd/bwd | **AMP + GradScaler + 梯度累积 + EMA** |
| 数据增强 | albumentations 固定 7 项 | spatial → photo → noise → jpeg → **Cutout** → norm，可调 `augment_strength` |
| 评估指标 | 函数 `compute_metrics` | **`MetricMeter` 类**：AUC / AP / EER / ACC / Best-F1 |
| 目录结构 | `training/ data_processing/` 平铺 | `drf/{core,engine,data,metrics}/` + `tools/` + `configs/` |

---

## 📊 实验结果

训练集：FaceForensics++（FF++ C23），跨域测试：CelebDF-v2 + DFDC（零样本泛化）

| 配置 | CelebDF-v2 AUC | DFDC AUC | 平均 AUC |
|---|---|---|---|
| DRF v2 baseline（ViT-B/32） | 0.7701 | 0.6852 | 0.7327 |
| DRF v2 + proj_head ablation | 0.7650 | 0.6803 | 0.7227 |
| DRF v2 + DCT 频域分支 ablation | 0.7589 | 0.6759 | 0.7174 |
| **DRF v2（ViT-L/14 backbone）** | **0.8701** | **0.6433** | **0.7567** |
| DRF v2（ViT-L/14 + TTA） | 0.8694 | 0.6483 | **0.7589** ⬆️ |

> **完全确定性评估** ✅：测试集 `shuffle=False`，eval 模式无随机操作，多次运行结果完全相同（详见 [stability_summary.md](logs/drf_v2_remote_vitl14/stability_summary.md)）。  
> **发布结果对应配置**：`configs/remote_vitl14.yaml`（backbone: `openai/clip-vit-large-patch14`）。

### 消融实验说明

本项目提供三组现成消融配置，证明每个模块的独立贡献：

| 配置文件 | 关闭的部件 | 结果影响 |
|---|---|---|
| `configs/ablation_no_srm.yaml` | SRM 高频残差 → 退化为 RGB 3ch | avg AUC 下降 |
| `configs/ablation_no_gate.yaml` | GRCA 标量门控 → 纯 cross-attention | 融合稳定性降低 |
| `configs/ablation_no_supcon.yaml` | Hard-Neg SupCon → 纯 BCE | 难样本区分能力下降 |

---

## 📁 模块化目录结构

```
Dual-Res-Forensics/
├── README.md
├── LICENSE                        # Apache-2.0
├── THIRD_PARTY_NOTICES.md         # 第三方归因声明
├── setup.py                       # pip install -e . 支持
├── main.py                        # 快速冒烟测试入口
├── configs/                       # 所有训练/消融配置
│   ├── remote_vitl14.yaml         # 发布结果对应配置（ViT-L/14）
│   ├── ablation_no_srm.yaml
│   ├── ablation_no_gate.yaml
│   └── ablation_no_supcon.yaml
├── data/                          # 测试集 JSON 索引
│   ├── cdfv2_test.json
│   └── dfdc_test.json
├── data_processing/
│   └── preprocess.py              # 基于 DeepfakeBench 的人脸检测与 256×256 对齐
├── tools/
│   └── test.py                    # 离线评估入口（支持 --tta）
├── drf/                           # 核心库
│   ├── core/
│   │   ├── filters.py             # SRMResidual（3 组固定噪声残差核）
│   │   ├── residual_encoder.py    # DWResidualEncoder（DW 可分离残差块）
│   │   ├── fusion.py              # GatedResidualCrossAttention（GRCA）★ 核心
│   │   ├── freq_branch.py         # DCTFrequencyBranch（可选频域分支）
│   │   ├── heads.py               # BinaryClassifierHead + BoundaryDecoder
│   │   └── model.py               # DRFModel（orchestrator）
│   ├── engine/
│   │   ├── losses.py              # BinaryClsLoss + HardNegSupConLoss
│   │   ├── optim.py               # AdamW + cosine warmup scheduler
│   │   ├── ema.py                 # ModelEMA（指数移动平均）
│   │   ├── trainer.py             # Trainer（AMP / 梯度累积 / EMA / resume）
│   │   └── utils.py               # set_seed / pick_device / safe_load_checkpoint
│   ├── data/
│   │   ├── transforms.py          # 数据增强流水线
│   │   └── dataset.py             # JSON 格式数据集加载器
│   └── metrics/
│       └── classification.py      # MetricMeter（AUC/AP/EER/ACC/Best-F1）
└── logs/
    └── drf_v2_remote_vitl14/      # ViT-L/14 实验结果与可视化
        ├── test_summary.md
        ├── stability_summary.md
        └── visualizations/
```

---

## ⚡ 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision transformers opencv-python numpy albumentations
# 可选：安装为本地包
pip install -e .
```

### 2. 冒烟测试（无需数据集）

```bash
python main.py
```

成功后打印模型参数量并完成随机数据前向与反向传播验证。

### 3. 数据准备

参照 [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) 完成裁脸/抽帧后，整理为 JSON 格式：

```json
[
  {"image_path": "/abs/path/real_0001.jpg", "label": 0},
  {"image_path": "/abs/path/fake_0001.jpg", "label": 1}
]
```

### 4. 评估（使用发布 checkpoint）

```bash
python -m tools.test \
    --config configs/remote_vitl14.yaml \
    --ckpt logs/drf_v2_remote_vitl14/ckpt/ckpt_best.pth \
    --prefer model

# 开启测试时增强（TTA）
python -m tools.test \
    --config configs/remote_vitl14.yaml \
    --ckpt logs/drf_v2_remote_vitl14/ckpt/ckpt_best.pth \
    --tta
```

> **注意**：发布结果使用 `backbone.clip_name: openai/clip-vit-large-patch14`，首次运行会自动下载约 1.1GB 权重。

---

## 🔬 技术细节

### SRM 高频残差滤波

使用 SRM（Spatial Rich Model）论文中 3 组经典噪声残差核（5×5 KV 核、5×5 SQUARE 核、3×3 EDGE 核），对 RGB 三通道分别做组卷积，输出 9 通道高频残差图。所有卷积核权重固定，不参与训练。

### GRCA 门控残差引导交叉注意力

- **Pre-LayerNorm**：相比 Post-LN 训练更稳定
- **SwiGLU FFN**：`y = Linear(SiLU(W1·x) × W2·x)`，参数效率高于 ReLU FFN
- **可学习 [CLS] AttnPool**：替代 mean-pool，让模型自适应地聚合 token 序列
- **标量门控 g**：`output = g × attended_feat + (1-g) × residual_global`，可学习地平衡局部与全局特征

### 训练细节

- **优化器**：AdamW，lr=2e-4，weight_decay=1e-3
- **调度器**：Cosine decay with warmup（warmup_ratio=0.05）
- **混合精度**：AMP + GradScaler
- **EMA**：decay=0.999，评估时使用 EMA 模型
- **损失**：BCE+LS（label_smoothing=0.05）+ Hard-Neg SupCon（weight=0.3，temperature=0.07，top-16 难负例）
- **数据增强**：augment_strength=1.3（spatial → photo → noise → jpeg → Cutout → normalize）

---

## 📝 引用

本项目以 **Apache-2.0** 许可证开源，科研思想与依赖的第三方归因详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

```bibtex
@InProceedings{Cui_2025_CVPR,
    author    = {Cui, Xinjie and Li, Yuezun and Luo, Ao and Zhou, Jiaran and Dong, Junyu},
    title     = {Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection},
    booktitle = {CVPR},
    year      = {2025},
    pages     = {19207-19217}
}
```

---

## 👥 贡献者

感谢所有为本项目做出贡献的同学！详见 [CONTRIBUTORS.md](CONTRIBUTORS.md)。

指导教师：**杨榆**（北京邮电大学 网络空间安全学院）
