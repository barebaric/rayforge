# 环境设置

本指南介绍如何设置 Rayforge 的开发环境。

## Linux

### 前提条件

请参阅 [安装指南](../../getting-started/installation#linux-pixi) 了解 Pixi 安装说明。

### 预提交钩子（可选）

要在每次提交前自动格式化和检查代码，您可以安装预提交钩子：

```bash
pixi run pre-commit-install
```

### 常用命令

所有命令通过 `pixi run` 运行：

-   `pixi run rayforge`：运行应用程序。
    -   添加 `--loglevel=DEBUG` 获取更详细的输出。
-   `pixi run test`：使用 `pytest` 运行完整测试套件。
-   `pixi run format`：使用 `ruff` 格式化所有代码。
-   `pixi run lint`：运行所有检查器（`flake8`、`pyflakes`、`pyright`）。

## Windows

### 前提条件

请参阅 [安装指南](../../getting-started/installation#windows-developer) 了解详细的 MSYS2 开发环境设置说明。

### 快速开始

Windows 上的开发任务通过 `run.bat` 脚本管理，它是 MSYS2 shell 的包装器。

克隆仓库并完成 MSYS2 设置后，您可以从标准 Windows 命令提示符或 PowerShell 使用以下命令：

```batch
.\run.bat setup
```

这将执行 `scripts/win/win_setup.sh` 以将所有必要的系统和 Python 包安装到您的 MSYS2/MinGW64 环境中。

### 预提交钩子（可选）

要在每次提交前自动格式化和检查代码，请从 MSYS2 MINGW64 shell 运行：

```bash
bash scripts/win/win_setup_dev.sh
```

:::note

预提交钩子需要在 MSYS2 MINGW64 shell 中运行 git 命令，而不是在 PowerShell 或命令提示符中。

:::

### 常用命令

所有命令通过 `run.bat` 脚本运行：

-   `run app`：从源代码运行应用程序。
    -   添加 `--loglevel=DEBUG` 获取更详细的输出。
-   `run test`：使用 `pytest` 运行完整测试套件。
-   `run lint`：运行所有检查器（`flake8`、`pyflakes`、`pyright`）。
-   `run format`：使用 `ruff` 格式化并自动修复代码。
-   `run build`：构建最终的 Windows 可执行文件（`.exe`）。

或者，您可以直接从 MSYS2 MINGW64 shell 运行脚本：

-   `bash scripts/win/win_run.sh`：运行应用程序。
-   `bash scripts/win/win_test.sh`：运行测试套件。
-   `bash scripts/win/win_lint.sh`：运行所有检查器。
-   `bash scripts/win/win_format.sh`：格式化并自动修复代码。
-   `bash scripts/win/win_build.sh`：构建 Windows 可执行文件。
