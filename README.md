# weakagent

## 手动打包

在**项目根目录**（含 `pyproject.toml` 的目录）执行以下步骤。

### 1. 环境

- Python **3.11+**
- 建议使用虚拟环境，避免污染系统 Python

```bash
python -m venv .venv
# Windows (cmd):
# .venv\Scripts\activate
# Windows (Git Bash) / Linux / macOS:
source .venv/Scripts/activate   # Git Bash on Windows
# 或: source .venv/bin/activate
```

### 2. 安装打包工具

任选其一：

**方式 A：`build`（官方推荐，生成 sdist + wheel）**

```bash
pip install build
```

**方式 B：`uv`（若已安装 uv）**

```bash
# 无需单独装 build，直接构建
uv build
```

### 3. 执行构建

**使用 `build`：**

```bash
python -m build
```

默认会生成：

- `dist/<项目名>-<版本>.tar.gz`：源码包（sdist）
- `dist/<项目名>-<版本>-py3-none-any.whl`：内置发行包（wheel）

**仅打 wheel 或仅打 sdist（可选）：**

```bash
python -m build --wheel      # 只要 wheel
python -m build --sdist      # 只要 sdist
```

**使用 `uv`：**

```bash
uv build
```

产物同样在 `dist/` 目录下。

### 4. 清理后重新打包（可选）

若曾构建过，可先删除再打包，避免混用旧文件：

```bash
rm -rf build dist *.egg-info
# Windows PowerShell: Remove-Item -Recurse -Force build,dist,*egg-info -ErrorAction SilentlyContinue
```

然后重复第 3 步。

### 5. 本地验证安装

从 wheel 安装到当前环境：

```bash
pip install dist/weakagent-0.1.0-py3-none-any.whl
```

版本号以 `pyproject.toml` 里 `[project]` 的 `version` 为准，若已修改，请把文件名换成实际生成的 wheel 名。

或从源码包安装：

```bash
pip install dist/weakagent-0.1.0.tar.gz
```

开发阶段也可**可编辑安装**（不生成 `dist`，直接链到源码）：

```bash
pip install -e .
```

---

**说明：** 发布到 PyPI 时，除上述构建外，还需配置 PyPI 账号与 `twine upload dist/*` 等步骤；仅本地或内网分发时，分发 `dist/` 下的文件即可。
