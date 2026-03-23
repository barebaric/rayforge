# 插件注册表

注册表是 Rayforge 管理可扩展性的方式。每个注册表保存一组相关组件——步骤、生成器、操作等。当您的插件注册某些内容时，它将在整个应用程序中可用。

## 注册表的工作原理

所有注册表都遵循类似的模式。它们提供 `register()` 方法来添加项目，以及各种查找方法来检索它们。大多数注册表还会跟踪哪个插件注册了每个项目，以便在插件卸载时进行清理。

以下是一般模式：

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

`addon_name` 参数是可选的，但建议使用。它确保如果用户禁用您的插件，您的组件会被正确移除。

## 步骤注册表

步骤注册表（`StepRegistry`）管理出现在操作面板中的步骤类型。每个步骤代表用户可以添加到其作业中的操作类型。

### 注册步骤

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

步骤的类名用作注册表键。您的步骤类应该继承自 `Step` 并定义 `TYPELABEL`、`HIDDEN` 等属性，并实现 `create()` 类方法。

### 检索步骤

注册表提供了几种查找步骤的方法：

```python
# Get a step by its class name
step_class = step_registry.get("MyCustomStep")

# Get a step by its TYPELABEL (for backward compatibility)
step_class = step_registry.get_by_typelabel("My Custom Step")

# Get all registered steps
all_steps = step_registry.all_steps()

# Get factory methods for UI menus (excludes hidden steps)
factories = step_registry.get_factories()
```

## 生成器注册表

生成器注册表（`ProducerRegistry`）管理 ops 生成器。生成器为步骤生成刀具路径操作——本质上，它们将您的工作件转换为机器指令。

### 注册生成器

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyCustomProducer
    producer_registry.register(MyCustomProducer, addon_name="my_addon")
```

默认情况下，类名成为注册表键。您可以指定自定义名称：

```python
producer_registry.register(MyCustomProducer, name="custom_name", addon_name="my_addon")
```

### 检索生成器

```python
# Get a producer by name
producer_class = producer_registry.get("MyCustomProducer")

# Get all producers
all_producers = producer_registry.all_producers()
```

## 转换器注册表

转换器注册表（`TransformerRegistry`）管理 ops 转换器。转换器在生成器生成操作后对其进行后处理——想想路径优化、平滑或添加固定标签等任务。

### 注册转换器

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyCustomTransformer
    transformer_registry.register(MyCustomTransformer, addon_name="my_addon")
```

### 检索转换器

```python
# Get a transformer by name
transformer_class = transformer_registry.get("MyCustomTransformer")

# Get all transformers
all_transformers = transformer_registry.all_transformers()
```

## 操作注册表

操作注册表（`ActionRegistry`）管理窗口操作。操作是您添加菜单项、工具栏按钮和键盘快捷键的方式。这是功能较丰富的注册表之一。

### 注册操作

```python
from gi.repository import Gio
from rayforge.ui_gtk.action_registry import MenuPlacement, ToolbarPlacement

@hookimpl
def register_actions(action_registry):
    # Create the action
    action = Gio.SimpleAction.new("my-action", None)
    action.connect("activate", lambda a, p: do_something())
    
    # Register with optional menu and toolbar placement
    action_registry.register(
        action_name="my-action",
        action=action,
        addon_name="my_addon",
        label="My Action",
        icon_name="document-new-symbolic",
        shortcut="<Ctrl><Alt>m",
        menu=MenuPlacement(menu_id="tools", priority=50),
        toolbar=ToolbarPlacement(group="main", priority=50),
    )
```

### 操作参数

注册操作时，您可以提供：

- `action_name`：操作的标识符（不带 "win." 前缀）
- `action`：`Gio.SimpleAction` 实例
- `addon_name`：您的插件名称，用于清理
- `label`：用于菜单和工具提示的可读文本
- `icon_name`：工具栏的图标标识符
- `shortcut`：使用 GTK 加速器语法的键盘快捷键
- `menu`：指定菜单和优先级的 `MenuPlacement` 对象
- `toolbar`：指定工具栏组和优先级的 `ToolbarPlacement` 对象

### 菜单位置

`MenuPlacement` 类接受：

- `menu_id`：要添加到哪个菜单（例如 "tools"、"arrange"）
- `priority`：数字越小越靠前

### 工具栏位置

`ToolbarPlacement` 类接受：

- `group`：工具栏组标识符（例如 "main"、"arrange"）
- `priority`：数字越小越靠前

### 检索操作

```python
# Get action info
info = action_registry.get("my-action")

# Get all actions for a specific menu
menu_items = action_registry.get_menu_items("tools")

# Get all actions for a toolbar group
toolbar_items = action_registry.get_toolbar_items("main")

# Get all actions with keyboard shortcuts
shortcuts = action_registry.get_all_with_shortcuts()
```

## 命令注册表

命令注册表（`CommandRegistry`）管理编辑器命令。命令扩展文档编辑器的功能。

### 注册命令

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand, addon_name="my_addon")
```

命令类应该在其构造函数中接受 `DocEditor` 实例。

### 检索命令

```python
# Get a command by name
command_class = command_registry.get("my_command")

# Get all commands
all_commands = command_registry.all_commands()
```

## 资源类型注册表

资源类型注册表（`AssetTypeRegistry`）管理可以存储在文档中的资源类型。这启用了动态反序列化——当 Rayforge 加载包含您的自定义资源的文档时，它知道如何重建它。

### 注册资源类型

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(
        MyCustomAsset,
        type_name="my_asset",
        addon_name="my_addon"
    )
```

`type_name` 是在序列化文档中用于标识您的资源类型的字符串。

### 检索资源类型

```python
# Get an asset class by type name
asset_class = asset_type_registry.get("my_asset")

# Get all registered asset types
all_types = asset_type_registry.all_types()
```

## 布局策略注册表

布局策略注册表（`LayoutStrategyRegistry`）管理用于在文档编辑器中排列内容的布局策略。

### 注册布局策略

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(
        MyLayoutStrategy,
        name="my_layout",
        addon_name="my_addon"
    )
```

请注意，标签和快捷键等 UI 元数据应通过操作注册表注册，而不是在这里。

### 检索布局策略

```python
# Get a strategy by name
strategy_class = layout_registry.get("my_layout")

# Get all strategy classes
all_strategies = layout_registry.list_all()

# Get all strategy names
strategy_names = layout_registry.list_names()
```

## 导入器注册表

导入器注册表（`ImporterRegistry`）管理文件导入器。导入器处理将外部文件加载到 Rayforge 中。

### 注册导入器

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter, addon_name="my_addon")
```

您的导入器类应该定义 `extensions` 和 `mime_types` 类属性，以便注册表知道它处理哪些文件。

### 检索导入器

```python
# Get importer by file extension
importer_class = importer_registry.get_by_extension(".xyz")

# Get importer by MIME type
importer_class = importer_registry.get_by_mime_type("application/x-xyz")

# Get importer by class name
importer_class = importer_registry.get_by_name("MyCustomImporter")

# Get appropriate importer for a file path
importer_class = importer_registry.get_for_file(Path("file.xyz"))

# Get all supported file extensions
extensions = importer_registry.get_supported_extensions()

# Get all file filters for file dialogs
filters = importer_registry.get_all_filters()

# Get importers that support a specific feature
importers = importer_registry.by_feature(ImporterFeature.SOME_FEATURE)
```

## 导出器注册表

导出器注册表（`ExporterRegistry`）管理文件导出器。导出器处理将 Rayforge 文档或操作保存为外部格式。

### 注册导出器

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter, addon_name="my_addon")
```

您的导出器类应该定义 `extensions` 和 `mime_types` 类属性。

### 检索导出器

```python
# Get exporter by file extension
exporter_class = exporter_registry.get_by_extension(".xyz")

# Get exporter by MIME type
exporter_class = exporter_registry.get_by_mime_type("application/x-xyz")

# Get all file filters for file dialogs
filters = exporter_registry.get_all_filters()
```

## 渲染器注册表

渲染器注册表（`RendererRegistry`）管理资源渲染器。渲染器在 UI 中显示资源。

### 注册渲染器

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer(), addon_name="my_addon")
```

请注意，您注册的是渲染器实例，而不是类。渲染器的类名用作注册表键。

### 检索渲染器

```python
# Get renderer by class name
renderer = renderer_registry.get("MyAssetRenderer")

# Get renderer by name (same as get)
renderer = renderer_registry.get_by_name("MyAssetRenderer")

# Get all renderers
all_renderers = renderer_registry.all()
```

## 库管理器

库管理器（`LibraryManager`）管理材料库。虽然技术上不是注册表，但它遵循类似的模式来注册插件提供的库。

### 注册材料库

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

注册的库默认是只读的。用户可以查看和使用材料，但不能通过 UI 修改它们。
