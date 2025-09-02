# Pre-commit 设置指南

## 快速开始

### 1. 安装 pre-commit

```bash
# 使用 uv 安装
uv add pre-commit

# 或使用 pip 安装
pip install pre-commit
```

### 2. 安装 git hooks

```bash
# 在项目根目录运行
pre-commit install
```

### 3. 可选：安装 pre-commit 到 git 模板（所有新项目）

```bash
pre-commit install --install-hooks
```

## 使用方法

### 手动运行所有检查

```bash
# 检查所有文件
pre-commit run --all-files

# 检查特定文件
pre-commit run black main.py

# 检查暂存的文件
pre-commit run
```

### 跳过特定检查

```bash
# 跳过单个检查
pre-commit run --skip black

# 跳过多个检查
pre-commit run --skip black --skip flake8
```

### 更新 hooks

```bash
# 更新到最新版本
pre-commit autoupdate

# 查看更新内容
pre-commit autoupdate --dry-run
```

## 配置的检查工具

### 代码质量

- **black**: 代码格式化 (88字符行长度)
- **isort**: import 排序 (与 black 兼容)
- **flake8**: 代码风格检查 (扩展规则)
- **mypy**: 静态类型检查
- **pyupgrade**: Python 语法升级

### 安全检查

- **bandit**: 安全漏洞检查
- **safety**: 依赖安全检查
- **detect-private-key**: 私钥检测

### 文件检查

- **end-of-file-fixer**: 文件结尾换行符
- **trailing-whitespace**: 行尾空格检查
- **check-yaml**: YAML 语法检查
- **check-json**: JSON 语法检查
- **yamllint**: YAML 格式检查
- **markdownlint**: Markdown 格式检查

### 文档检查

- **interrogate**: 文档覆盖率检查
- **flake8-docstrings**: docstring 格式检查

## 故障排除

### 常见问题

1. **mypy 找不到类型定义**

   ```bash
   # 安装类型存根
   uv add types-requests types-urllib3
   ```

2. **black 与其他工具冲突**

   ```bash
   # 确保 isort 使用 black 配置
   pre-commit run isort --all-files
   ```

3. **权限问题**

   ```bash
   # 给予执行权限
   chmod +x .git/hooks/pre-commit
   ```

### 手动修复问题

```bash
# 自动修复可修复的问题
pre-commit run --all-files

# 查看详细错误信息
pre-commit run --all-files --verbose

# 只运行失败的检查
pre-commit run --all-files --hook-stage manual
```

## CI/CD 集成

### GitHub Actions

```yaml
name: Pre-commit
on: [push, pull_request]
jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install pre-commit
        run: pip install pre-commit
      - name: Run pre-commit
        run: pre-commit run --all-files
```

### 跳过 CI 检查

```bash
# 提交时跳过 pre-commit
git commit -m "feat: add new feature" --no-verify

# 或者在特定文件上跳过
git commit -m "feat: add new feature" --no-verify -- file1.py file2.py
```

## 最佳实践

1. **在提交前运行**: `pre-commit run --all-files`
2. **定期更新**: `pre-commit autoupdate`
3. **团队协作**: 将 `.pre-commit-config.yaml` 加入版本控制
4. **CI 集成**: 在 CI/CD 流程中运行 pre-commit
5. **逐步采用**: 可以先启用部分检查，逐步增加

## 配置自定义

### 排除特定文件或目录

在 `.pre-commit-config.yaml` 中添加 `exclude` 配置：

```yaml
exclude: |
  (?x)^(
      docs/.*\.md|
      tests/test_.*\.py|
      \.venv/.*
  )$
```

### 自定义参数

每个 hook 都可以通过 `args` 自定义参数：

```yaml
- id: black
  args: [--line-length=100, --target-version=py312]
```

### 添加自定义检查

```yaml
- repo: local
  hooks:
    - id: custom-check
      name: Custom Check
      entry: python custom_check.py
      language: python
```
