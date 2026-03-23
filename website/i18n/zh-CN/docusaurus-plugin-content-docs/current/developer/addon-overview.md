# 插件开发概述

Rayforge 使用基于 [pluggy](https://pluggy.readthedocs.io/) 的插件系统，让您可以扩展功能、添加新的机器驱动程序或集成自定义逻辑，而无需修改核心代码库。

## 快速入门

最快的入门方式是使用官方的 [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template)。复刻或克隆它，重命名目录，并更新元数据以匹配您的插件。

## 插件的工作原理

`AddonManager` 会扫描 `addons` 目录以查找有效的插件。插件只是一个包含 `rayforge-addon.yaml` 清单文件以及您的 Python 代码的目录。

以下是典型插件的结构：

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- 必需的清单文件
├── my_addon/            <-- 您的 Python 包
│   ├── __init__.py
│   ├── backend.py       <-- 后端入口点
│   └── frontend.py      <-- 前端入口点（可选）
├── assets/              <-- 可选资源
├── locales/             <-- 可选翻译（.po 文件）
└── README.md
```

## 创建您的第一个插件

让我们创建一个简单的插件来注册自定义机器驱动程序。首先，创建清单文件：

```yaml title="rayforge-addon.yaml"
name: my_laser_driver
display_name: "My Laser Driver"
description: "Adds support for the XYZ laser cutter."
api_version: 9

author:
  name: Jane Doe
  email: jane@example.com

provides:
  backend: my_addon.backend
```

现在创建注册您的驱动程序的后端模块：

```python title="my_addon/backend.py"
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def register_machines(machine_manager):
    """Register our custom machine driver."""
    from .my_driver import MyLaserMachine
    machine_manager.register("my_laser", MyLaserMachine)
```

就这样！您的插件将在 Rayforge 启动时加载，您的机器驱动程序将可供用户使用。

[清单文件](./addon-manifest.md)文档涵盖了所有可用的配置选项。

## 理解入口点

插件可以提供两个入口点，每个入口点在不同时间加载：

**后端**（backend）入口点在主进程和工作进程中都会加载。将其用于机器驱动程序、步骤类型、ops 生成器和转换器，或任何不需要 UI 依赖的核心功能。

**前端**（frontend）入口点仅在主进程中加载。这是您放置 UI 组件、GTK 小部件、菜单项以及任何需要访问主窗口的内容的地方。

两者都指定为点分模块路径，如 `my_addon.backend`。

## 通过钩子连接到 Rayforge

Rayforge 使用 `pluggy` 钩子让插件与应用程序集成。只需用 `@pluggy.HookimplMarker("rayforge")` 装饰您的函数：

```python
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context: RayforgeContext):
    """Called when Rayforge is fully initialized."""
    # Your setup code here
    pass

@hookimpl
def on_unload():
    """Called when the addon is being disabled or unloaded."""
    # Clean up resources here
    pass
```

[钩子](./addon-hooks.md)文档描述了每个可用的钩子及其调用时机。

## 注册您的组件

大多数钩子接收一个注册表对象，您可以使用它来注册自定义组件：

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)

@hookimpl
def register_actions(action_registry):
    from .actions import setup_actions
    setup_actions(action_registry)
```

[注册表](./addon-registries.md)文档解释了每个注册表及其使用方法。

## 访问 Rayforge 的数据

`rayforge_init` 钩子让您可以访问 `RayforgeContext` 对象。通过此上下文，您可以访问 Rayforge 中的所有内容：

您可以通过 `context.machine` 获取当前活动的机器，或通过 `context.machine_mgr` 访问所有机器。`context.config` 对象保存全局设置，而 `context.camera_mgr` 提供对摄像头源的访问。对于材料，使用 `context.material_mgr`，对于处理配方，使用 `context.recipe_mgr`。G-code 方言管理器可用作 `context.dialect_mgr`，AI 功能通过 `context.ai_provider_mgr` 访问。对于本地化，检查 `context.language` 获取当前语言代码。插件管理器本身可用作 `context.addon_mgr`，如果您正在构建付费插件，`context.license_validator` 处理许可证验证。

## 添加翻译

插件可以使用标准的 `.po` 文件提供翻译。按以下方式组织：

```text
my-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── my_addon.po
│   └── es/
│       └── LC_MESSAGES/
│           └── my_addon.po
```

Rayforge 会在加载您的插件时自动将 `.po` 文件编译为 `.mo` 文件。

## 开发期间测试

要在本地测试您的插件，请从您的开发文件夹创建一个符号链接到 Rayforge 的插件目录。

首先，找到您的配置目录。在 Windows 上，它是 `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`。在 macOS 上，查看 `~/Library/Application Support/rayforge/addons`。在 Linux 上，它是 `~/.config/rayforge/addons`。

然后创建符号链接：

```bash
ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
```

重启 Rayforge 并检查控制台是否有类似 `Loaded addon: my_laser_driver` 的消息。

## 分享您的插件

当您准备好分享您的插件时，将其推送到 GitHub 或 GitLab 上的公共 Git 仓库。然后通过复刻仓库、添加插件元数据并打开拉取请求，将其提交到 [rayforge-registry](https://github.com/barebaric/rayforge-registry)。

一旦被接受，用户可以直接通过 Rayforge 的插件管理器安装您的插件。
