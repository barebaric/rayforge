# 扩展开发

Rayforge 使用基于 [pluggy](https://pluggy.readthedocs.io/) 的扩展系统，允许开发者扩展功能、添加新机器驱动程序或集成自定义逻辑，而无需修改核心代码库。

## 1. 快速入门

最快的方式是使用官方模板。

1. **Fork 或克隆** [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template)。
2. **重命名**目录并更新元数据。

## 2. 扩展结构

`AddonManager` 扫描 `addons` 目录。有效的扩展必须是一个包含清单文件的目录：

**目录布局：**

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- 必需的清单
├── my_addon/            <-- Python 包
│   ├── __init__.py
│   ├── backend.py       <-- 后端入口点
│   └── frontend.py      <-- 前端入口点（可选）
├── assets/              <-- 可选资源
├── locales/             <-- 可选翻译（.po 文件）
└── README.md
```

## 3. 清单（`rayforge-addon.yaml`）

此文件告诉 Rayforge 如何加载您的扩展。

```yaml
# rayforge-addon.yaml

# 扩展的唯一标识符（目录名称）
name: my_custom_addon

# 人类可读的显示名称
display_name: "My Custom Addon"

# UI 中显示的描述
description: "Adds support for the XYZ laser cutter."

# API 版本（必须与 Rayforge 的 PLUGIN_API_VERSION 匹配）
api_version: 1

# Rayforge 版本依赖
depends:
  - rayforge>=0.27.0,<2.0.0

# 可选：其他扩展依赖
requires:
  - some-other-addon>=1.0.0

# 扩展提供的内容
provides:
  # 后端模块（在主进程和 worker 进程中加载）
  backend: my_addon.backend
  # 前端模块（仅在主进程中加载，用于 UI）
  frontend: my_addon.frontend
  # 可选的 asset 文件
  assets:
    - path: assets/profiles.json
      type: profiles

# 作者元数据
author:
  name: Jane Doe
  email: jane@example.com

url: https://github.com/username/my-custom-addon
```

### 必填字段

- `name`：唯一标识符（应与目录名称匹配）
- `display_name`：UI 中显示的可读名称
- `description`：扩展功能的简要描述
- `api_version`：必须为 `1`（与 Rayforge 的 `PLUGIN_API_VERSION` 匹配）
- `depends`：Rayforge 的版本约束列表
- `author`：包含 `name`（必填）和 `email`（可选）的对象

### 可选字段

- `requires`：其他扩展依赖列表
- `provides`：入口点和 assets
- `url`：项目主页或仓库

## 4. 入口点

扩展可以提供两种类型的入口点：

### 后端（`provides.backend`）

在主进程和 worker 进程中都加载。用于：
- 机器驱动程序
- 步骤类型
- Ops 生产者
- 没有 UI 依赖的核心功能

### 前端（`provides.frontend`）

仅在主进程中加载。用于：
- UI 组件
- GTK 小部件
- 菜单项
- 需要主窗口的操作

入口点指定为点分隔的模块路径（例如 `my_addon.backend`）。

## 5. 编写扩展代码

Rayforge 使用 `pluggy` 钩子。要挂钩到 Rayforge，请定义用 `@pluggy.HookimplMarker("rayforge")` 装饰的函数。

### 基本样板（`backend.py`）

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    在 Rayforge 完全初始化时调用。
    这是您访问管理器的主要入口点。
    """
    logger.info("My Custom Addon 已启动！")

    machine = context.machine
    if machine:
        logger.info(f"扩展运行在机器上：{machine.id}")

@hookimpl
def on_unload():
    """
    当扩展被禁用或卸载时调用。
    清理资源、关闭连接、取消注册处理程序。
    """
    logger.info("My Custom Addon 正在关闭")

@hookimpl
def register_machines(machine_manager):
    """
    在启动期间调用以注册新的机器驱动程序。
    """
    from .my_driver import MyNewMachine
    machine_manager.register("my_new_machine", MyNewMachine)

@hookimpl
def register_steps(step_registry):
    """
    调用以注册自定义步骤类型。
    """
    from .my_step import MyCustomStep
    step_registry.register("my_custom_step", MyCustomStep)

@hookimpl
def register_producers(producer_registry):
    """
    调用以注册自定义 ops 生产者。
    """
    from .my_producer import MyProducer
    producer_registry.register("my_producer", MyProducer)

@hookimpl
def register_step_widgets(widget_registry):
    """
    调用以注册自定义步骤设置小部件。
    """
    from .my_widget import MyStepWidget
    widget_registry.register("my_custom_step", MyStepWidget)

@hookimpl
def register_menu_items(menu_registry):
    """
    调用以注册菜单项。
    """
    from .menu_items import register_menus
    register_menus(menu_registry)

@hookimpl
def register_commands(command_registry):
    """
    调用以注册编辑器命令。
    """
    from .commands import register_commands
    register_commands(command_registry)

@hookimpl
def register_actions(window):
    """
    调用以注册窗口操作。
    """
    from .actions import setup_actions
    setup_actions(window)
```

### 可用钩子

定义在 `rayforge/core/hooks.py` 中：

**`rayforge_init`**（`context`）
: **主入口点。** 在配置、摄像头和硬件加载后调用。用于逻辑、UI 注入或监听器。

**`on_unload`**（）
: 当扩展被禁用或卸载时调用。用于清理资源、关闭连接、取消注册处理程序等。

**`register_machines`**（`machine_manager`）
: 在启动期间调用以注册新的机器驱动程序。

**`register_steps`**（`step_registry`）
: 调用以允许插件注册自定义步骤类型。

**`register_producers`**（`producer_registry`）
: 调用以允许插件注册自定义 ops 生产者。

**`register_step_widgets`**（`widget_registry`）
: 调用以允许插件注册自定义步骤设置小部件。

**`register_menu_items`**（`menu_registry`）
: 调用以允许插件注册菜单项。

**`register_commands`**（`command_registry`）
: 调用以允许插件注册编辑器命令。

**`register_actions`**（`window`）
: 调用以允许插件注册窗口操作。

## 6. 访问 Rayforge 数据

`rayforge_init` 钩子提供 **`RayforgeContext`**。通过此对象，您可以访问：

- **`context.machine`**：当前活动的机器实例。
- **`context.config`**：全局配置设置。
- **`context.config_mgr`**：配置管理器。
- **`context.machine_mgr`**：机器管理器（所有机器）。
- **`context.camera_mgr`**：访问摄像头源和计算机视觉工具。
- **`context.material_mgr`**：访问材料库。
- **`context.recipe_mgr`**：访问加工配方。
- **`context.dialect_mgr`**：G-code 方言管理器。
- **`context.language`**：本地化内容的当前语言代码。
- **`context.addon_mgr`**：扩展管理器实例。
- **`context.plugin_mgr`**：插件管理器实例。
- **`context.debug_dump_manager`**：调试转储管理器。
- **`context.artifact_store`**：管道工件存储。

## 7. 本地化

扩展可以使用 `.po` 文件提供翻译：

```text
my-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── my_addon.po
│   └── zh_CN/
│       └── LC_MESSAGES/
│           └── my_addon.po
```

`.po` 文件在扩展安装或加载时自动编译为 `.mo` 文件。

## 8. 开发和测试

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

## 9. 发布

要与社区分享您的扩展：

1.  **托管在 Git 上：** 将代码推送到公共 Git 仓库（GitHub、GitLab 等）。
2.  **提交到注册表：**
    - 访问 [rayforge-registry](https://github.com/barebaric/rayforge-registry)。
    - Fork 仓库。
    - 将您扩展的 Git URL 和元数据添加到注册表列表。
    - 提交 Pull Request。

一旦接受，用户可以直接通过 Rayforge UI 或使用 Git URL 安装您的扩展。
