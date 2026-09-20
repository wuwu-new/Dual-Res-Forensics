# GitHub协作规范

## 分支结构

- `main`：已验收、可发布版本。成员禁止直接push。
- `ccf-c-experiments-20260920`：本轮CCF-C论文实验集成分支。
- `exp/lin-full-clip`：林雨泓的Full DRF与Strict CLIP工作分支。
- `exp/wan-residual-srm`：李婉铃的残差/SRM消融与Xception工作分支。
- `exp/jia-gate-supcon`：李佳乐的门控/SupCon消融与外部基线工作分支。

所有PR以`ccf-c-experiments-20260920`为base。集成分支稳定并通过验收后，再单独向`main`提交发布PR。

## 每日操作

开始工作前：

```bash
git fetch origin
git switch <自己的分支>
git rebase origin/ccf-c-experiments-20260920
```

完成一个可验证节点后：

```bash
git status --short
python -m compileall -q drf tools
git diff --check
git add <本次相关文件>
git commit -m "<类型>(<姓名>): <具体结果>"
git push --force-with-lease origin <自己的分支>
```

只有在执行过rebase且远程个人分支已存在时才使用`--force-with-lease`，禁止使用`--force`。

## 实验结果管理

GitHub保存：代码、YAML配置、数据清单、轻量指标JSON、实验记录和图表。共享存储保存：原始数据、帧、checkpoint、完整日志和逐样本预测。

每次结果提交必须在个人实验记录中填写：commit、配置、seed、数据清单SHA256、best epoch、validation AUC、三个测试集指标、运行状态和外部文件链接。只提交最好数字但不保留失败记录视为未通过验收。

## PR审核

- 林雨泓PR由李佳乐审核。
- 李婉铃PR由林雨泓审核。
- 李佳乐PR由李婉铃审核。
- 至少一名审核人批准，且自动/手动检查通过后才能Squash and merge。
- 修改公共协议、数据划分或指标定义的PR必须三人共同确认。
