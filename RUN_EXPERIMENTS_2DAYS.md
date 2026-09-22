# DRF 两天实验、分工与交稿执行手册

## 0. 本轮实验范围

训练统一使用 FaceForensics++ C23，验证集只用于选择 checkpoint；最终跨域测试统一使用 Celeb-DF v2、DFDC 和 WildDeepfake。现有 CDF-v2/DFDC 索引是各 10k 帧且没有 `video_id` 的旧版索引，必须重建；FF++ train/val 和 WildDeepfake 索引目前也需要生成。

两天内必须先完成老师要求的六组核心实验。公开对比的最低主表由严格 CLIP 语义基线、Xception、官方 Forensics Adapter 和 Full DRF 组成；单 GPU 情况下，Xception和Forensics Adapter排在六组核心实验之后，不能牺牲核心消融。

## 1. 统一协议

- FF++必须按原始视频 ID 划分train/validation，禁止同一视频的帧跨集合。
- CDF-v2、DFDC、WildDeepfake必须使用相同的人脸检测、裁剪边距和每视频抽帧策略。
- 所有方法使用相同JSON样本集合；网络需要的归一化和原生输入尺寸可以不同，但必须记录。
- checkpoint只按FF++ validation AUC选择。跨域测试集不得用于提前停止、调参或挑checkpoint。
- 主表报告video AUC；同时报告frame AUC、AP、EER、ACC和视频/帧数量。
- 两天核心实验统一seed 2027；投稿版补2027、3407、4096三个种子并报告均值±标准差。
- 显存不足时，六组DRF实验必须一起调整batch和梯度累积，不能只改其中一组。

## 2. 实验清单

### 2.1 主对比实验

| 方法 | 实现 | 作用 | 两天优先级 |
|---|---|---|---|
| Strict CLIP baseline | `configs/baseline_semantic.yaml` | 纯语义基线 | 2 |
| Xception | `configs/baseline_xception.yaml` | 经典空间检测基线 | 7 |
| Forensics Adapter | 官方仓库 + `tools.evaluate_predictions` | 同类CLIP泛化方法 | 8 |
| Full DRF | `configs/full_drf.yaml` | 本文完整方法 | 1 |

SPSL/F3-Net或ReCLIP-SFD属于投稿增强项，不进入两天核心队列。禁止把协议不同的论文数字当作公平复现结果；引用数字只能单独标成“reported”。

### 2.2 核心消融

| 方法 | 配置 | 唯一目标变化 |
|---|---|---|
| No residual | `ablation_no_residual.yaml` | 关闭残差路径 |
| No SRM | `ablation_no_srm.yaml` | 残差编码器改收RGB |
| No gate | `ablation_no_gate.yaml` | 关闭自适应门控 |
| No Hard-Neg SupCon | `ablation_no_supcon.yaml` | 对比损失权重设0 |

## 3. 三人分工

三人各自主跑两组核心实验。公开基线属于核心六组完成后的追加任务，不计入“每人两种方法”，避免某个人同时背负三组正式训练。

### 3.1 林雨泓：总控、完整方法与语义基线

核心实验：

```bash
python -m tools.train --config configs/full_drf.yaml --seed 2027
python -m tools.train --config configs/baseline_semantic.yaml --seed 2027
```

在Colab或服务器上无需修改YAML，可直接覆盖索引路径：

```bash
python -m tools.train --config configs/full_drf.yaml --seed 2027 \
  --train-json /data/FaceForensics++/ffpp_c23_train.json \
  --val-json ffpp_val=/data/FaceForensics++/ffpp_c23_val.json
```

对应输出目录：`logs/drf_full/`、`logs/drf_baseline_semantic/`。

具体责任：

1. Day 1 08:30前确认统一代码包、依赖版本、GPU编号和公共seed，禁止三人使用不同代码副本。
2. 负责FF++ C23训练/验证索引，运行`tools.check_split`，提交train/val视频数、帧数和零交集截图或文本记录。
3. 先用Full DRF完成小批次前向、反向、保存和续训测试；通过后通知另外两人启动正式训练。
4. 每两小时记录Full DRF和Strict CLIP的epoch、validation AUC、预计完成时间；训练失败时保留`run_manifest.json`和错误信息。
5. Day 2负责Full DRF的TP、TN、FP、FN和困难样本选择，生成CLIP语义、SRM残差与GRCA融合四联图。
6. 汇总三人的结果，形成主对比表、摘要中的核心数字和最终论文目录。

必须交付：两组`config_snapshot.yaml`、`run_manifest.json`、`metrics.log`、best/last checkpoint、三个测试集的结果与逐样本预测、至少8张代表性可视化。

### 3.2 李婉铃：残差机制消融与Xception

核心实验：

```bash
python -m tools.train --config configs/ablation_no_residual.yaml --seed 2027
python -m tools.train --config configs/ablation_no_srm.yaml --seed 2027
```

对应输出目录：`logs/drf_ablation_no_residual/`、`logs/drf_v2_ablation_no_srm/`。

具体责任：

1. Day 1 08:30–10:30负责CDF-v2与WildDeepfake完整视频索引；统一每视频抽帧数量、裁脸边距和路径格式。
2. 对两份索引运行`tools.check_data --check-files --require-video-id`，提交真假帧数、视频数和缺失文件数。
3. 在林雨泓的Full DRF冒烟测试通过后，分别启动No residual和No SRM；不得独自修改公共batch、epoch、增强或学习率。
4. 比较`Full → No residual`和`Full → No SRM`在三个数据集上的变化，分别撰写“残差分支贡献”和“SRM先验贡献”结论；不能把二者混成一个结论。
5. 六组核心实验全部进入稳定运行后，如有空闲GPU，再负责Xception训练和测试；Xception不挤占核心消融资源。
6. 负责论文中的数据集统计表、训练/测试协议表和残差相关消融段落。

必须交付：两组核心结果及运行材料、CDF-v2/WildDeepfake新索引与统计；有算力时再交`logs/baseline_xception/`完整结果。

### 3.3 李佳乐：融合/损失消融与外部基线

核心实验：

```bash
python -m tools.train --config configs/ablation_no_gate.yaml --seed 2027
python -m tools.train --config configs/ablation_no_supcon.yaml --seed 2027
```

对应输出目录：`logs/drf_v2_ablation_no_gate/`、`logs/drf_v2_ablation_no_supcon/`。

具体责任：

1. Day 1 08:30–10:30负责DFDC完整视频索引，并复核三份跨域测试索引使用一致的抽帧与裁脸协议。
2. 负责生成数据清单：每份JSON的SHA256、总帧数、真假帧数和视频数；三个人训练前必须使用同一份清单：

```bash
python -m tools.build_data_manifest data/ffpp_c23_train.json data/ffpp_c23_val.json \
  data/cdfv2_test.json data/dfdc_test.json data/wilddeepfake_test.json \
  --out logs/data_manifest.json
```
3. 启动No gate和No SupCon，分别回答“门控是否有效”和“Hard-Neg SupCon是否有效”；重点核对配置只改变对应目标因素。
4. Day 2负责收集八组`test_result.json`，运行`summarize_results`并检查每组的frames/videos数量完全一致。
5. 核心六组完成后，用官方Forensics Adapter在同一测试JSON上推理；通过`tools.evaluate_predictions --expected-index`导入，不允许直接抄论文数字进入公平主表。
6. 负责论文消融表、公开基线说明、指标定义和复现附录。

必须交付：两组核心结果及运行材料、DFDC新索引、数据SHA256清单、消融汇总表；有条件时再交官方Forensics Adapter预测与统一评价结果。

### 3.4 交叉验收关系

| 被检查人 | 检查人 | 必查内容 |
|---|---|---|
| 林雨泓 | 李佳乐 | Full/CLIP配置、best epoch、三个测试集样本数 |
| 李婉铃 | 林雨泓 | No residual/No SRM是否严格控制变量、索引统计 |
| 李佳乐 | 李婉铃 | No gate/No SupCon配置、外部预测覆盖是否完整 |

检查人必须在Pull Request评论中写“通过”或明确问题，不能只口头确认。任何公共超参变化都要同步六份核心配置，并更换实验名重新运行，禁止覆盖旧结果。

### 3.5 每名成员的GitHub提交操作

统一集成分支为`ccf-c-experiments-20260920`。三人不得直接向`main`或集成分支提交，必须从集成分支创建自己的功能分支，并通过Pull Request合并回集成分支。

#### 林雨泓：`exp/lin-full-clip`

需要加入个人分支的内容：

- Full DRF和Strict CLIP涉及的代码修复及配置：`configs/full_drf.yaml`、`configs/baseline_semantic.yaml`。
- FF++划分/检查工具的必要修复；不提交FF++原始数据和含本机绝对路径的索引。
- 两种方法的轻量最终指标：`results/lin/full_drf.json`、`results/lin/baseline_semantic.json`。
- 选定的论文可视化：`results/lin/figures/`，只放最终图，不放所有中间图。
- 汇总表或说明：`results/lin/README.md`，简述seed、best epoch、验证AUC、三测试集frame/video AUC及共享存储链接。

首次创建分支：

```bash
git fetch origin
git switch ccf-c-experiments-20260920
git pull --ff-only origin ccf-c-experiments-20260920
git switch -c exp/lin-full-clip

# 完成一个可验证阶段后，只添加本次相关文件
git add configs/full_drf.yaml configs/baseline_semantic.yaml results/lin/
git commit -m "result(lin): add full DRF and CLIP baseline results"
git push -u origin exp/lin-full-clip
```

后续继续提交：

```bash
git switch exp/lin-full-clip
git pull --rebase origin exp/lin-full-clip
git add <本次实际修改的文件>
git commit -m "fix(lin): <具体修改>"
git push origin exp/lin-full-clip
```

完成后创建PR：`exp/lin-full-clip` → `ccf-c-experiments-20260920`，由李佳乐审核。

#### 李婉铃：`exp/wan-residual-srm`

需要加入个人分支的内容：

- No residual和No SRM配置/必要代码修复：`configs/ablation_no_residual.yaml`、`configs/ablation_no_srm.yaml`。
- CDF-v2、WildDeepfake数据索引生成工具的必要修改；不提交数据、帧或含绝对路径的大索引。
- 两项消融的轻量最终指标：`results/wan/no_residual.json`、`results/wan/no_srm.json`。
- 残差/SRM对比表或图：`results/wan/figures/`。
- 结果摘要：`results/wan/README.md`。
- 核心六组结束后，如果完成Xception，再加入`configs/baseline_xception.yaml`与`results/wan/xception.json`。

首次创建分支：

```bash
git fetch origin
git switch ccf-c-experiments-20260920
git pull --ff-only origin ccf-c-experiments-20260920
git switch -c exp/wan-residual-srm

git add configs/ablation_no_residual.yaml configs/ablation_no_srm.yaml results/wan/
git commit -m "result(wan): add residual and SRM ablation results"
git push -u origin exp/wan-residual-srm
```

完成后创建PR：`exp/wan-residual-srm` → `ccf-c-experiments-20260920`，由林雨泓审核。

#### 李佳乐：`exp/jia-gate-supcon`

需要加入个人分支的内容：

- No gate和No SupCon配置/必要代码修复：`configs/ablation_no_gate.yaml`、`configs/ablation_no_supcon.yaml`。
- DFDC索引工具、数据清单或统一汇总工具的必要修改。
- 两项消融的轻量最终指标：`results/jia/no_gate.json`、`results/jia/no_supcon.json`。
- 完整消融汇总表：`results/jia/ablation_summary.md`。
- 结果摘要：`results/jia/README.md`。
- 核心六组结束后，如果完成Forensics Adapter，再加入统一指标文件`results/jia/forensics_adapter.json`；不提交逐样本预测。

首次创建分支：

```bash
git fetch origin
git switch ccf-c-experiments-20260920
git pull --ff-only origin ccf-c-experiments-20260920
git switch -c exp/jia-gate-supcon

git add configs/ablation_no_gate.yaml configs/ablation_no_supcon.yaml results/jia/
git commit -m "result(jia): add gate and SupCon ablation results"
git push -u origin exp/jia-gate-supcon
```

完成后创建PR：`exp/jia-gate-supcon` → `ccf-c-experiments-20260920`，由李婉铃审核。

#### 三人共同的提交规则

开始新的工作前，先把集成分支的最新修改同步到个人分支：

```bash
git fetch origin
git switch <自己的个人分支>
git merge --no-edit origin/ccf-c-experiments-20260920
git push origin <自己的个人分支>
```

提交信息格式：

```text
result(name): add <method> cross-dataset metrics
fix(name): correct <specific problem>
docs(name): update <specific table or explanation>
```

提交前统一执行：

```bash
git status --short
python -m compileall -q drf tools
git diff --check
```

禁止提交原始数据、人脸帧、`*.pth`、逐样本预测、完整训练日志、含本机绝对路径的大索引和个人密钥。上述大文件放共享存储，GitHub只加入轻量结果JSON、最终图表和README中的共享链接。

PR审核通过后使用Squash and merge。三人的PR全部合并且核心表格验收完成后，才由林雨泓从`ccf-c-experiments-20260920`向`main`发起最终PR。

## 4. 单GPU执行顺序

1. Full DRF
2. Strict CLIP baseline
3. No residual
4. No SRM
5. No gate
6. No Hard-Neg SupCon
7. Xception
8. 官方Forensics Adapter
9. TTA、三随机种子、额外数据集或额外公开方法

如果两天内第6项仍未结束，不启动TTA或多seed。多GPU时三人可并行各跑自己的两组核心实验，但依然先通过同一个小规模冒烟测试。

## 5. Day 1：数据与训练

08:30–10:30完成索引。通用索引工具假定每个视频的帧位于独立子目录：

```bash
python -m tools.build_ffpp_index \
  --root /data/FaceForensics++ \
  --output-dir /data/FaceForensics++

python -m tools.build_dataset_index --dataset wilddeepfake \
  --real-root /data/WildDeepfake/real --fake-root /data/WildDeepfake/fake \
  --output data/wilddeepfake_test.json

python -m tools.build_eval_index celebdf-v2 \
  --root /data/Celeb-DF-v2 --output /data/Celeb-DF-v2/cdfv2_test.json --strict

python -m tools.build_eval_index dfdc \
  --root /data/DFDC/test --output /data/DFDC/test/dfdc_test.json
```

FF++专用工具读取官方`train.json`、`val.json`和`test.json`，同时加入四种伪造方法，
并检查三个集合的源视频ID零交集。默认报告但跳过官方发布包中不存在的目录；只有在使用
自行补齐的完整副本时才增加`--strict`。索引与数据集根目录一起迁移到服务器可保持相对路径有效。

Celeb-DF v2工具会把官方测试清单的`1=real, 0=fake`转换为项目统一的
`0=real, 1=fake`。DFDC工具读取`metadata.json`中的`is_fake`，并报告未成功提取
人脸帧的视频；这些视频不能用空样本替代。

向Colab或服务器传输时，不要上传数十万个散文件。按索引制作无压缩tar包，PNG/JPEG
本身已压缩，无压缩打包速度更快，也不会改变数据：

```bash
python -m tools.package_indexed_data \
  --root /data/FaceForensics++ \
  --index /data/FaceForensics++/ffpp_c23_train.json \
  --index /data/FaceForensics++/ffpp_c23_val.json \
  --output /transfer/ffpp_train_val.tar
```

解包后索引仍位于数据根目录，索引中的相对图片路径无需改写。

网络受限时可先制作保留全部视频的应急小索引，用于冒烟测试或明确标注的预实验：

```bash
python -m tools.subsample_index --input /data/FaceForensics++/ffpp_c23_train.json \
  --output /data/FaceForensics++/ffpp_c23_train_8f.json --frames-per-video 8
```

该结果改变了既定的每视频32帧协议，不能冒充完整正式实验；正式论文结果仍须用完整索引复跑。

CDF-v2和DFDC使用同一命令，分别指定真实/伪造帧根目录。生成后强制检查：

```bash
python -m tools.check_data data/ffpp_c23_train.json data/ffpp_c23_val.json \
  data/cdfv2_test.json data/dfdc_test.json data/wilddeepfake_test.json \
  --check-files --require-video-id
python -m tools.check_split data/ffpp_c23_train.json data/ffpp_c23_val.json
```

10:30–12:00：每个配置先跑一个小批次，确认前向、反向、验证、保存和续训均正常。随后启动正式任务：

```bash
python -m tools.train --config configs/full_drf.yaml --seed 2027
python -m tools.train --config configs/baseline_semantic.yaml --seed 2027
```

每两小时登记epoch、train loss、validation AUC、峰值显存、已用时间和预计结束时间。不得查看跨域测试结果决定是否继续训练。

## 6. Day 2：测试与公开基线

每组只使用validation选择的`ckpt_best.pth`，一次性评测三个跨域集合：

```bash
python -m tools.test --config configs/full_drf.yaml \
  --ckpt logs/drf_full/ckpt/ckpt_best.pth --prefer ema --save-predictions
```

跨域测试索引同样通过命令行覆盖，避免提交本机路径：

```bash
python -m tools.test --config configs/full_drf.yaml \
  --ckpt logs/drf_full/ckpt/ckpt_best.pth --prefer ema \
  --test-json cdfv2=/data/Celeb-DF-v2/cdfv2_test.json \
  --test-json dfdc=/data/DFDC/test/dfdc_test.json \
  --test-json wilddeepfake=/data/WildDeepfake/wilddeepfake_test.json
```

六组核心实验的测试对应关系如下，负责人逐项替换上面命令中的配置和checkpoint：

| 负责人 | 配置 | checkpoint |
|---|---|---|
| 林雨泓 | `full_drf.yaml` | `logs/drf_full/ckpt/ckpt_best.pth` |
| 林雨泓 | `baseline_semantic.yaml` | `logs/drf_baseline_semantic/ckpt/ckpt_best.pth` |
| 李婉铃 | `ablation_no_residual.yaml` | `logs/drf_ablation_no_residual/ckpt/ckpt_best.pth` |
| 李婉铃 | `ablation_no_srm.yaml` | `logs/drf_v2_ablation_no_srm/ckpt/ckpt_best.pth` |
| 李佳乐 | `ablation_no_gate.yaml` | `logs/drf_v2_ablation_no_gate/ckpt/ckpt_best.pth` |
| 李佳乐 | `ablation_no_supcon.yaml` | `logs/drf_v2_ablation_no_supcon/ckpt/ckpt_best.pth` |

Xception使用相同命令替换配置与checkpoint：

```bash
python -m tools.train --config configs/baseline_xception.yaml --seed 2027
python -m tools.test --config configs/baseline_xception.yaml \
  --ckpt logs/baseline_xception/ckpt/ckpt_best.pth --prefer ema --save-predictions
```

Forensics Adapter必须用官方实现和相同JSON样本运行，并导出字段`dataset,image_path,video_id,label,pred_score`。统一评价命令：

```bash
python -m tools.evaluate_predictions \
  --predictions external_results/forensics_adapter_predictions.csv \
  --experiment forensics_adapter_official \
  --expected-index cdfv2=data/cdfv2_test.json \
  --expected-index dfdc=data/dfdc_test.json \
  --expected-index wilddeepfake=data/wilddeepfake_test.json \
  --out logs/forensics_adapter_official/test_result.json
```

导入前核对三个数据集的预测条数、路径、标签和video ID与JSON完全一致。缺预测、重复预测或使用不同裁脸结果时，该结果不得进入公平主表。

## 7. 汇总与可视化

```bash
python -m tools.summarize_results \
  logs/drf_full/test_result.json \
  logs/drf_baseline_semantic/test_result.json \
  logs/baseline_xception/test_result.json \
  logs/forensics_adapter_official/test_result.json \
  logs/drf_ablation_no_residual/test_result.json \
  logs/drf_v2_ablation_no_srm/test_result.json \
  logs/drf_v2_ablation_no_gate/test_result.json \
  logs/drf_v2_ablation_no_supcon/test_result.json \
  --out logs/experiment_summary.md
```

从Full DRF逐样本结果中选择TP、TN、FP、FN和高难样本各2–3例，生成输入、CLIP语义相关性、SRM残差能量、GRCA注意力和门值图。不得只展示成功案例。

## 8. 每组交付与验收

- 原始YAML、seed、代码版本、环境版本和GPU型号。
- `ckpt_best.pth`、`ckpt_last.pth`、完整训练日志和每epoch validation指标。
- `test_result.json`和`test_result_detailed.json`。
- frame/video指标、帧数、视频数、训练时长和峰值显存。
- 失败运行保留原因；修改超参后的运行必须使用新实验名，不得覆盖。

## 9. 论文交稿条件

两天目标是形成可信的核心表格和图，不等于立即投稿。CCF-C投稿前还需：公开基线至少两种、完整视频级测试、三随机种子、多数据集一致收益、效率表、相关工作定位和可复现附录。如果Full DRF只在个别数据集提升，应缩小泛化主张并分析失败案例，不能选择性汇报。
