# DRF 数据集获取与团队共享规范

## 1. 本轮论文实验需要的数据集

| 数据集 | 用途 | 本项目采用的范围 | 官方获取入口 |
|---|---|---|---|
| FaceForensics++ (FF++) | 唯一训练集、验证集和可选同域测试 | C23；original + Deepfakes + Face2Face + FaceSwap + NeuralTextures；使用官方视频级划分 | [官方仓库](https://github.com/ondyari/FaceForensics) / [访问申请](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform) |
| Celeb-DF v2 | 跨域测试 | 官方 `List_of_testing_videos.txt` 中的完整测试列表，不使用当前仓库的旧 10k 帧抽样 | [官方仓库](https://github.com/yuezunli/celeb-deepfakeforensics) / [Google 申请](https://forms.gle/2jYBby6y1FBU3u6q9) / [腾讯申请](https://wj.qq.com/s2/8540155/b5d9/) |
| DFDC | 跨域测试 | 使用 DeepfakeBench 发布的 DFDC test data 与标准 JSON；论文中必须注明是 full DFDC 的标准测试划分，不能把 DFDC Preview 或竞赛无标签测试视频混写为 DFDC | [Kaggle 官方数据](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) / [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) |
| WildDeepfake | 跨域测试 | 官方 test 部分；以官方 face-sequence 目录作为 `video_id`/`sequence_id`，正文明确报告 sequence-level 聚合口径 | [官方仓库](https://github.com/deepfakeinthewild/deepfake-in-the-wild) / [使用协议](https://forms.gle/o8vy9Q8fQ5mQZ4Qk6) / [Hugging Face](https://huggingface.co/datasets/xingjunm/WildDeepfake) |

两天核心实验不需要额外下载 FaceShifter、DeepFakeDetection、UADFV、DF40、FF++ C40 或 FF++ raw/c0。Xception 和 Forensics Adapter 必须使用上表相同的样本，不为基线单独准备另一份测试集。

## 2. 推荐的最快获取路线

1. 三名成员先分别接受各数据集原始提供方的条款。申请表必须填写真实姓名、学校、学术邮箱和研究用途。
2. FF++ 只下载 C23 compressed videos，不下载 raw/c0、C40、masks、models 或全量 PNG。
3. 为与公开基线对齐，优先使用 DeepfakeBench 发布的 32 frames/video 预处理 RGB 数据和标准 JSON。下载前仍需遵守每个原始数据集的许可条款。DeepfakeBench 的入口见其 README 中的 `Rgb-format Datasets` 和 `Json Configurations`。
4. WildDeepfake 官方只发布 face sequences，没有完整原视频；使用官方 test face sequences，并在论文中披露其聚合单位。
5. 先下载每个数据集的少量样本跑通流程，再开始完整下载。少量样本只用于冒烟测试，不能进入正式表格。

官方规模较大。免费 Google Drive 的 15 GB 无法同时容纳 CLIP 缓存、FF++、三个测试集、处理后帧和 checkpoint。团队应准备至少 50 GB，建议 100 GB 以上的受控共享存储或 GPU 服务器磁盘。

## 3. 统一数据协议

- 训练：FF++ C23 官方 train split。
- 选择 checkpoint：仅使用 FF++ C23 官方 validation split。
- 最终测试：FF++ test（同域，可选主表）以及 Celeb-DF v2、DFDC、WildDeepfake（跨域）。
- FF++ 必须按原始视频 ID 划分；同一原始视频及其操纵配对不得跨 train/validation/test。
- FF++、Celeb-DF v2 和 DFDC 统一采用 DeepfakeBench 的人脸检测、对齐、裁剪和每视频 32 帧协议。
- 所有方法共享完全相同的帧索引。只允许模型所需的输入归一化不同。
- JSON 每条记录必须包含 `image_path`、`label` 和非空 `video_id`。
- JSON 使用相对于 JSON 文件的路径，禁止写 `/root/...`、`D:\\...` 等机器绝对路径。
- 训练前必须运行 `tools.check_data --check-files --require-video-id` 和 `tools.check_split`。

仓库现有的 `data/cdfv2_test.json` 与 `data/dfdc_test.json` 是旧版各 10,000 帧抽样，包含其他机器的绝对路径且没有 `video_id`，只能用于追溯旧结果，不能用于本轮正式实验。

## 4. GitHub 只共享复现材料，不共享数据集

不得提交到 GitHub：

- 原始视频、抽帧图片、人脸裁剪图、mask 或 landmarks；
- ZIP、TAR、7Z 等数据压缩包；
- FF++ 私人下载脚本或下载链接、Kaggle API token、Google/Hugging Face token；
- checkpoint、逐样本预测、完整训练日志；
- 含本机绝对路径的 JSON，或几十万条帧记录的大型 JSON。

需要提交到 GitHub：

- 下载入口和版本说明（本文件）；
- 预处理、划分、索引生成和检查代码；
- 官方 split 文件或由官方 split 派生的轻量视频 ID 列表，前提是许可证允许；
- 每份本地索引的 SHA256、真假帧数、视频数和处理参数；
- 轻量最终指标 JSON、论文表格和选定图片。

原始数据只能放在获授权的学校服务器、团队私有存储或各成员本地。不要通过公开 GitHub、公开网盘或匿名直链二次分发。若数据许可不允许团队内转存，每名成员应分别从官方入口获取。

## 5. 三人复现同一数据的方法

1. 每名成员在自己的存储中保持相同的目录层级和文件名。
2. 林雨泓生成 FF++ train/validation 视频 ID 清单和索引统计。
3. 李婉铃生成 Celeb-DF v2 与 WildDeepfake 测试索引统计。
4. 李佳乐生成 DFDC 测试索引统计和总数据清单。
5. 三人比较 `tools.build_data_manifest` 生成的样本数、视频数、标签数和 SHA256；全部一致后才启动正式训练。
6. GitHub PR 只提交脚本、轻量清单和统计结果，目标分支为 `ccf-c-experiments-20260920`。

若成员使用同一台服务器，可共享只读数据目录，但每人仍使用自己的 Git 分支和实验输出目录。
