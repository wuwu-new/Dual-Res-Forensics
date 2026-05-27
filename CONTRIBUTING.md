# Contributing to DRF v2

感谢你对 Dual-Res-Forensics (DRF v2) 项目的兴趣！本文档提供贡献指南。

## 开发环境设置

### 1. Fork 并 Clone 仓库
```bash
git clone https://github.com/YOUR_USERNAME/Dual-Res-Forensics.git
cd Dual-Res-Forensics
git remote add upstream https://github.com/wuwu-new/Dual-Res-Forensics.git
```

### 2. 创建虚拟环境
```bash
# 使用 conda
conda create -n drf-dev python=3.12
conda activate drf-dev

# 或使用 venv
python -m venv venv
source venv/bin/activate  # 或 venv\Scripts\activate (Windows)
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
pip install -e .  # 开发模式安装
```

## 工作流程

### 1. 创建特性分支
```bash
git checkout -b feature/your-feature-name
# 或修复 bug
git checkout -b bugfix/issue-description
```

### 2. 编码规范
- **Python**: PEP 8
- **类名**: CapitalCase (如 `DRFModel`)
- **函数名**: snake_case (如 `forward_pass`)
- **常量**: UPPER_CASE (如 `NUM_HEADS`)

### 3. 提交信息格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

类型:
- `feat`: 新特性
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码风格
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/依赖

示例:
```
feat(model): add flash attention support

- Implements FlashAttention-2 for improved performance
- Reduces memory usage by 30% on large batches
- Backward compatible with standard attention

Closes #123
```

### 4. 测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_model.py

# 生成覆盖率报告
pytest --cov=drf tests/
```

### 5. Push 并创建 Pull Request
```bash
git push origin feature/your-feature-name
```

在 GitHub 上创建 PR，描述：
- 修改了什么
- 为什么修改
- 如何测试

## 代码审查流程

1. **自动检查** (CI/CD)
   - 代码风格检查 (flake8)
   - 类型检查 (mypy)
   - 单元测试

2. **代码审查**
   - 至少 1 个维护者审查
   - 遵循项目架构
   - 包含文档更新

3. **合并**
   - 所有检查通过
   - 至少 1 个 approval
   - 分支已同步 main

## 报告 Bug

创建 Issue 时包含：
- DRF 版本
- Python 版本
- PyTorch 版本
- 复现步骤
- 期望行为
- 实际行为
- 错误堆栈

## 提出特性建议

描述：
- 功能概要
- 使用场景
- 可能的实现方案
- 相关论文/参考

## 文档贡献

- 更新 README.md 中的说明
- 添加 docstring 到新函数
- 更新 DOCUMENTATION.md API 参考
- 添加使用示例到 examples/

## 许可证

通过提交代码，你同意在 MIT 许可证下发布。

## 联系方式

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: wuwu@example.com

---

感谢你的贡献！🙏
