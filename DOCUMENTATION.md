# DRF v2 代码文档速查表

## 核心模型架构

### DRFModel (drf/core/model.py)
**作用**: DRF v2 的主编排器（Orchestrator），组织所有子模块的前向传播

**初始化参数**:
```python
DRFModel(
    clip_name='openai/clip-vit-large-patch14',      # CLIP 骨干网络名称
    c_token=192,                                     # 残差编码维度
    token_grid=8,                                    # 残差分块大小
    fused_dim=128,                                   # 融合层输出维度
    fusion_embed_dim=256,                            # 融合注意力中间维度
    fusion_num_heads=8,                              # 融合注意力头数
    dropout=0.1,                                     # dropout 率
    freeze_clip=True,                                # 是否冻结 CLIP 主干
    use_srm=True,                                    # 是否使用 SRM 高频残差
    use_gate=True,                                   # 是否使用门控融合
    use_boundary_head=False,                         # 是否使用边界预测头
    use_freq=False,                                  # 是否使用频域分支（DCT）
    freq_c_token=192,                                # 频域分支维度
    freq_token_grid=8,                               # 频域分块大小
)
```

**前向传播**:
```python
output = model(x)  # x: (B, 3, H, W)
# 返回: {
#   'logit': (B,),           # 二分类 logit（未激活）
#   'fused_feat': (B, fused_dim),  # 融合特征（用于对比损失）
#   'proj_feat': (B, proj_dim),    # 可选投影特征（如果有投影头）
# }
```

**架构管线**:
```
输入 (B,3,H,W)
  ├→ CLIP-ViT (frozen)     → clip_tokens (B, N_c, C_c)
  └→ SRM 残差 (fixed)      → (B, 9, H, W)
       → DW 编码器         → residual_tokens (B, N_r, C_r), fmap
  → 门控交叉注意力(Q=residual, K/V=clip)  → fused_feat (B, fused_dim)
  ├→ 分类头                → logit (B,)
  └→ 可选投影头/边界头
```

---

## 高频残差模块

### SRMResidual (drf/core/filters.py)
**作用**: 固定的高频残差提取（使用 SRM 核心），输出 9 通道

**特点**:
- 使用 3 个 SRM 残差核（固定，不可训练）
- 每个核生成 3 通道，共 9 通道
- 直接应用于输入，不参与反向传播

**输入/输出**:
```python
srm = SRMResidual()
residual = srm(x)  # x: (B, 3, H, W) → (B, 9, H, W)
```

### DWResidualEncoder (drf/core/residual_encoder.py)
**作用**: 深度可分离残差编码器，编码高频残差特征为 token

**特点**:
- MobileNet 风格的可分离卷积残差块
- 输入 (B, 9, H, W) → 输出 (B, C_token, N_token)
- 同时提供特征图（用于边界预测）

**参数**:
- `c_token`: 输出 token 维度（通常 192）
- `token_grid`: 分块大小（通常 8，即 224/8 = 28×28）

---

## 融合模块

### GatedResidualCrossAttention (drf/core/fusion.py)
**作用**: 门控残差引导交叉注意力融合，融合残差和 CLIP 特征

**核心机制**:
1. **Query**: 残差 tokens（高频特征）
2. **Key/Value**: CLIP tokens（语义特征）
3. **融合方式**: 
   - 手写 SDP-Attention (Scaled Dot-Product)
   - Pre-LayerNorm + SwiGLU FFN
   - 可学习 [CLS] Attention Pooling
   - 标量门控融合（可学习权重）

**输入/输出**:
```python
grca = GatedResidualCrossAttention(
    embed_dim=256,      # 中间维度
    num_heads=8,        # 注意力头数
    fused_dim=128,      # 输出维度
    dropout=0.1,
)
fused = grca(residual_tokens, clip_tokens, freq_tokens=None)  
# residual_tokens: (B, N_r, C_r)
# clip_tokens: (B, N_c, C_c)
# 输出: (B, fused_dim)
```

---

## 分类头与输出

### BinaryClassifierHead (drf/core/heads.py)
**作用**: 单 logit 二分类头（真伪分类）

**特点**:
- 输入融合特征 (B, fused_dim)
- 输出单个 logit (B,)（未激活）
- 评估时：`prob_fake = sigmoid(logit)`

### BoundaryDecoder (可选)
**作用**: 可选的边界预测头，用于生成伪造概率图

**参数**:
- `use_boundary_head=True` 时激活
- 基于编码器特征图生成 (B, 1, h, w) 边界图

---

## 损失函数

### BinaryClsLoss (drf/engine/losses.py)
**作用**: 二分类损失（BCEWithLogits + 标签平滑）

**特点**:
- 替代旧版 2-class CrossEntropy
- 支持标签平滑（防过拟合）
- 支持正例权重（处理不平衡）

**使用**:
```python
cls_loss_fn = BinaryClsLoss(label_smoothing=0.05, pos_weight=None)
loss = cls_loss_fn(logit, label)  # logit: (B,), label: (B,) ∈ {0,1}
```

### HardNegSupConLoss (drf/engine/losses.py)
**作用**: 难负例挖掘的监督对比损失

**特点**:
- 每个 anchor 仅保留 top-k 难负例
- 正例不变（同类所有样本）
- 相对于全量 SupCon：更稳定，避免噪声负例

**使用**:
```python
con_loss_fn = HardNegSupConLoss(
    temperature=0.1,      # 相似度温度
    num_hard_neg=16,      # 每个样本保留多少难负例
)
loss_con = con_loss_fn(feat, label)  # feat: (B, D), label: (B,)
```

---

## 训练引擎

### Trainer (drf/engine/trainer.py)
**作用**: v2 版本的训练器，支持 AMP、梯度累积、EMA、跨 epoch 学习率调度

**关键特性**:
1. **混合精度 (AMP)**: `torch.amp.autocast` + `GradScaler`
2. **梯度累积**: 支持 `grad_accum_steps > 1`
3. **每 step 调度**: Warmup + Cosine 衰减（而非每 epoch）
4. **EMA 权重**: 维护指数移动平均模型用于评估
5. **损失组合**: BCE + Hard-Neg SupCon（可配置权重）

**初始化**:
```python
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    ckpt_dir='logs/exp_name/ckpt',
    use_amp=True,
    grad_accum_steps=2,
    grad_clip_norm=1.0,
    ema_decay=0.999,
    contrastive_weight=0.3,
    contrastive_temperature=0.07,
    num_hard_neg=16,
    label_smoothing=0.05,
)
```

**主要方法**:
```python
# 训练一个 epoch
metrics = trainer.train_epoch(train_loader, epoch=0)
# 返回 {'loss': avg_loss, 'cls': cls_loss, 'con': con_loss, 'lr': learning_rate}

# 评估（使用 EMA 模型）
results = trainer.eval_epoch(test_loaders, epoch=0)
# 返回 {'cdfv2': auc, 'dfdc': auc, 'avg_auc': (cdfv2+dfdc)/2, ...}

# 获取 checkpoint dict
ckpt_dict = trainer.state_dict()
# 包含: model_state, ema_state（如果启用）, optimizer_state, ...
```

---

## 数据处理

### ForgeryDataset (drf/data/dataset.py)
**作用**: 深伪检测数据集（支持训练和测试模式）

**初始化**:
```python
ds = ForgeryDataset(
    json_path='data/ffpp_c23_train.json',  # 数据集 JSON 文件
    image_size=224,                         # 目标图像大小
    mode='train',                           # 'train' 或 'test'
    augment_strength=1.3,                   # 数据增强强度（仅训练模式）
)
```

**数据格式** (JSON):
```json
[
  {
    "image": "path/to/real.jpg",
    "label": 0
  },
  {
    "image": "path/to/fake.jpg",
    "label": 1
  }
]
```

**输出**:
```python
batch = ds[0]
# {
#   'image': (3, 224, 224)，已归一化
#   'label': 0 或 1
#   'image_path': str
# }
```

### MetricMeter (drf/metrics/classification.py)
**作用**: 计算分类指标（AUC、AP、EER、ACC、F1、阈值）

**使用**:
```python
meter = MetricMeter()
meter.update(y_true, y_score)  # y_true: (N,), y_score: (N,) ∈ [0,1]
metrics = meter.compute()
# {'auc': float, 'ap': float, 'eer': float, 'acc': float, 'best_f1': float, 'best_thr': float}
```

---

## 评估脚本

### tools/test.py
**作用**: 离线评估入口（加载 checkpoint 并在测试集上计算指标）

**使用**:
```bash
python -m tools.test \
  --config configs/remote_vitl14.yaml \
  --ckpt logs/drf_v2_remote_vitl14/ckpt/ckpt_best.pth \
  --prefer model \           # 'model' 或 'ema'（加载哪个权重）
  --seed 42 \                # 固定随机种子（test 无影响）
  --tta \                    # 启用测试时增强（5-crop + flip）
  --out logs/test_result.json
```

**参数说明**:
- `--prefer model`: 加载原始模型权重（推荐）；`--prefer ema` 加载 EMA 权重
- `--tta`: 启用测试时增强（平均 5-crop + 水平翻转的预测）
- `--seed`: 固定随机种子（在 shuffle=False 的 test 模式下无实际效果）

**输出** (JSON):
```json
{
  "avg_auc": 0.7567,
  "per_dataset": {
    "cdfv2": {"auc": 0.8701, "ap": 0.8920, "eer": 20.90, ...},
    "dfdc": {"auc": 0.6433, "ap": 0.6806, "eer": 39.79, ...}
  },
  "ckpt": "logs/...",
  "loaded_from": "model_state"
}
```

---

## 配置文件结构

### configs/remote_vitl14.yaml
**关键配置项**:

```yaml
experiment:
  name: drf_v2_remote_vitl14              # 实验名称

backbone:
  clip_name: openai/clip-vit-large-patch14   # CLIP 模型
  freeze_clip: true                      # 冻结 CLIP

residual:
  c_token: 192                           # 残差编码维度
  token_grid: 8                          # 分块大小
  use_srm: true                          # 使用 SRM
  use_freq: false                        # 是否启用频域分支

fusion:
  embed_dim: 256                         # 注意力中间维度
  num_heads: 8                           # 注意力头数
  fused_dim: 128                         # 输出维度
  dropout: 0.1
  use_gate: true                         # 门控融合

model:
  use_boundary_head: false               # 边界预测

data:
  image_size: 224
  batch_size: 32
  num_workers: 4
  augment_strength: 1.3                  # 增强强度
  train_json: ./data/ffpp_c23_train.json
  test_jsons:
    cdfv2: ./data/cdfv2_test.json
    dfdc:  ./data/dfdc_test.json

loss:
  label_smoothing: 0.05
  contrastive_weight: 0.3                # 对比损失权重
  contrastive_temperature: 0.07
  num_hard_neg: 16                       # 难负例数

optim:
  lr: 0.0002
  weight_decay: 0.001
  warmup_ratio: 0.05                     # 预热比例
  min_lr_ratio: 0.01                     # 最小学习率

train:
  num_epochs: 8
  use_amp: true                          # 混合精度
  grad_accum_steps: 2                    # 梯度累积步数
  grad_clip_norm: 1.0
  ema_decay: 0.999                       # EMA 衰减
```

---

## 快速开始

### 1. 安装
```bash
cd DRF_release
pip install -r requirements.txt
```

### 2. 数据准备（JSON 格式）
```bash
python scripts/prepare_json.py --input-dir /path/to/images --output data/train.json
```

### 3. 训练
```bash
python -m tools.train \
  --config configs/remote_vitl14.yaml \
  --seed 42 \
  --resume-ckpt logs/exp_name/ckpt/ckpt_last.pth
```

### 4. 评估
```bash
python -m tools.test \
  --config configs/remote_vitl14.yaml \
  --ckpt logs/exp_name/ckpt/ckpt_best.pth \
  --prefer model \
  --tta
```

---

## 关键性能指标 (v2 remote_vitl14)

| 配置 | cdfv2 AUC | dfdc AUC | Avg AUC | 备注 |
|---|---|---|---|---|
| Baseline | 0.8701 | 0.6433 | **0.7567** | ViT-L/14 + SRM + GRCA |
| + TTA | 0.8694 | 0.6483 | **0.7589** | 5-crop + flip 平均 |

---

**最后更新**: 2026-05-27  
**维护者**: DRF v2 Team
