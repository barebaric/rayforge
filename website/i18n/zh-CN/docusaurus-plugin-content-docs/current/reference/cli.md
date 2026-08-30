---
description: "Rayforge 命令行界面参考。"
---

# 命令行

Rayforge 命令行选项的完整参考。

```
rayforge [选项] [文件名...]
```

---

## 位置参数

| 参数     | 描述                            |
| -------- | ------------------------------- |
| `文件名` | 启动时要打开的 SVG 或图像文件。 |

---

## 选项

| 选项              | 描述                     |
| ----------------- | ------------------------ |
| `--version`       | 打印版本并退出。         |
| `-h`, `--help`    | 显示帮助并退出。         |
| `--loglevel 级别` | 日志级别。默认：`INFO`。 |
| `--config 目录`   | 自定义配置目录。         |
| `--exit`          | 导入后退出。             |
| `--vector`        | 强制作为直接矢量导入。   |
| `--trace`         | 强制通过位图描摹导入。   |
| `--script 脚本`   | 早期启动脚本。           |
| `--uiscript 脚本` | UI 脚本（加载后）。      |

---

## 示例

### 打开文件

```bash
rayforge 我的项目.ryp
```

### 打开多个文件

```bash
rayforge 零件1.svg logo.png 设计.ryp
```

### 使用描摹导入

```bash
rayforge --trace 照片.png
```

### 运行早期脚本并退出

```bash
rayforge --exit --script 注册.py \
    我的项目.ryp
```

### UI 脚本（自动化）

```bash
rayforge --exit --uiscript 截图.py \
    我的项目.ryp
```

### 批处理

```bash
rayforge --exit --vector 输入.svg
```

---

## 早期脚本 (`--script`)

`--script` 标志在**启动期间同步运行** Python 脚本，在加载插件之前、在创建主窗口之前。适合用于：

- 向 `pluggy` 插件管理器注册插件
- 配置应用上下文
- 为文本框注册模板函数
- 在启动前设置环境变量

脚本可以通过 `get_context()` 访问上下文：

```python
from rayforge.context import get_context

ctx = get_context()
```

### 示例：注册自定义模板函数

```python
"""为文本框表达式注册自定义函数。

运行：rayforge --script 注册_fn.py
"""
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function("我的id", lambda: "零件-001")
```

现在 `{我的id()}` 在任何文本框中都有效。

参见 [自定义模板函数](../features/sketcher/expressions.md#custom-template-functions) 了解完整教程。

---

## UI 脚本 (`--uiscript`)

`--uiscript` 标志在**主窗口完全加载后**在后台线程中运行 Python 脚本。适合用于：

- 自动化 UI 测试
- 截取应用程序屏幕截图
- 端到端工作流

脚本可以直接导入应用程序和窗口：

```python
from rayforge.uiscript import app, win
```

脚本在**后台线程**中运行 — 访问 GTK 小部件时请注意线程安全（使用 `GLib.idle_add` 进行 GTK 操作）。

### 示例：截取屏幕截图

```python
"""截取主窗口的屏幕截图。"""
from rayforge.uiscript import app, win

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib

def capture():
    surface = win.get_surface()
    if surface:
        surface.write_to_png("/tmp/rayforge_screenshot.png")
    return GLib.SOURCE_REMOVE

GLib.idle_add(capture)
```

---

## 同时使用两个标志

`--script` 和 `--uiscript` 可以一起使用。 `--script` 先运行（同步），然后加载窗口，然后运行
`--uiscript`：

```bash
rayforge --script 早期设置.py \
    --uiscript 自动化.py \
    我的项目.ryp
```

当你需要先注册插件然后再驱动 UI 时，这非常有用。
