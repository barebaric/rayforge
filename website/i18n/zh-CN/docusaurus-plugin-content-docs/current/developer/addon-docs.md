# Rayforge 扩展开发指南

Rayforge 使用基于 [pluggy](https://pluggy.readthedocs.io/) 的扩展系统，允许开发者扩展功能、添加新机器驱动程序或集成自定义逻辑，而无需修改核心代码库。

## 1. 快速入门

最快的方式是使用官方模板。

1. **Fork 或克隆** [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template)。
2. **重命名**目录并更新元数据。

## 2. 扩展结构

`AddonManager` 扫描 `addons` 目录。有效的扩展必须是一个至少包含两个文件的目录：

1. `rayforge_addon.yaml`（元数据）
2. Python 入口点（例如 `addon.py`）

**目录布局：**

```text
my-rayforge-addon/
├── rayforge_addon.yaml  <-- 必需的清单
├── addon.py             <-- 入口点（逻辑）
├── assets/              <-- 可选资源
└── README.md
```

## 3. 清单（`rayforge_addon.yaml`）

此文件告诉 Rayforge 如何加载您的扩展。

```yaml
# rayforge_addon.yaml

# 扩展的唯一标识符
name: my_custom_addon

# 人类可读的显示名称
display_name: "My Custom Addon"

# UI 中显示的描述
description: "Adds support for the XYZ laser cutter."

# 依赖项（扩展和版本约束）
depends:
  - rayforge>=0.27.0,~0.27

# 要加载的 python 文件（相对于扩展文件夹）
entry_point: addon.py

# 作者元数据
author: Jane Doe
url: https://github.com/username/my-custom-addon
```

## 4. 编写扩展代码

Rayforge 使用 `pluggy` 钩子。要挂钩到 Rayforge，请定义用 `@pluggy.HookimplMarker("rayforge")` 装饰的函数。

### 基本样板（`addon.py`）

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

# 定义钩子实现标记
hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    在 Rayforge 完全初始化时调用。
    这是您访问管理器的主要入口点。
    """
    logger.info("My Custom Addon 已启动！")

    # 通过 context 访问核心系统
    machine = context.machine
    camera = context.camera_mgr

    if machine:
        logger.info(f"扩展运行在机器上：{machine.id}")

@hookimpl
def register_machines(machine_manager):
    """
    在启动期间调用以注册新的机器驱动程序。
    """
    # from .my_driver import MyNewMachine
    # machine_manager.register("my_new_machine", MyNewMachine)
    pass
```

### 可用钩子

定义在 `rayforge/core/hooks.py` 中：

**`rayforge_init`**（`context`）
: **主入口点。** 在配置、摄像头和硬件加载后调用。用于逻辑、UI 注入或监听器。

**`register_machines`**（`machine_manager`）**
: 在引导过程早期调用。用于注册新的硬件类/驱动程序。

## 5. 访问 Rayforge 数据

`rayforge_init` 钩子提供 **`RayforgeContext`**。通过此对象，您可以访问：

- **`context.machine`**：当前活动的机器实例。
- **`context.config`**：全局配置设置。
- **`context.camera_mgr`**：访问摄像头源和计算机视觉工具。
- **`context.material_mgr`**：访问材料库。
- **`context.recipe_mgr`**：访问加工配方。

## 6. 开发和测试

要在本地测试您的扩展而无需发布：

1.  **找到您的配置目录：**
    Rayforge 使用 `platformdirs`。

    - **Windows：** `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`
    - **macOS：** `~/Library/Application Support/rayforge/addons`
    - **Linux：** `~/.config/rayforge/addons`
      _（检查启动日志中的 `Config dir is ...`）_

2.  **符号链接您的扩展：**
    不要来回复制文件，而是从您的开发文件夹创建符号链接到 Rayforge 扩展文件夹。

    _Linux/macOS：_

    ```bash
    ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
    ```

3.  **重启 Rayforge：**
    应用程序在启动时扫描目录。检查控制台日志：
    > `Loaded addon: my_custom_addon`

## 7. 发布

要与社区分享您的扩展：

1.  **托管在 Git 上：** 将代码推送到公共 Git 仓库（GitHub、GitLab 等）。
2.  **提交到注册表：**
    - 访问 [rayforge-registry](https://github.com/barebaric/rayforge-registry)。
    - Fork 仓库。
    - 将您扩展的 Git URL 和元数据添加到注册表列表。
    - 提交 Pull Request。

一旦接受，用户可以直接通过 Rayforge UI 或使用 Git URL 安装您的扩展。
