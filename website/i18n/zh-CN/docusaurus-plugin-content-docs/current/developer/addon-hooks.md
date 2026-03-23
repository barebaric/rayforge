# 插件钩子

钩子是您的插件与 Rayforge 之间的连接点。当应用程序中发生某些事情——创建步骤、打开对话框或窗口初始化时——Rayforge 会调用任何已注册的钩子，以便您的插件可以响应。

## 钩子的工作原理

Rayforge 使用 [pluggy](https://pluggy.readthedocs.io/) 作为其钩子系统。要实现钩子，请使用 `@pluggy.HookimplMarker("rayforge")` 装饰函数：

```python
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context):
    # Your code runs when Rayforge finishes initializing
    pass
```

您不必实现每个钩子——只需实现您需要的钩子。所有钩子都是可选的。

## 生命周期钩子

这些钩子处理插件的整体生命周期。

### `rayforge_init(context)`

这是您的主要入口点。Rayforge 在应用程序上下文完全初始化后调用此钩子，这意味着所有管理器、配置和硬件都已准备就绪。将其用于常规设置、日志记录或注入 UI 元素。

`context` 参数是一个 `RayforgeContext` 实例，让您可以访问 Rayforge 中的所有内容。有关详细信息，请参阅[访问 Rayforge 数据](./addon-overview.md#accessing-rayforges-data)。

```python
@hookimpl
def rayforge_init(context):
    logger.info("My addon is starting up!")
    machine = context.machine
    if machine:
        logger.info(f"Running on machine: {machine.id}")
```

### `on_unload()`

当您的插件被禁用或卸载时，Rayforge 会调用此钩子。使用它来清理资源、关闭连接或注销处理程序。

```python
@hookimpl
def on_unload():
    logger.info("My addon is shutting down")
    # Clean up any resources here
```

### `main_window_ready(main_window)`

此钩子在主窗口完全初始化时触发。它对于注册 UI 页面、命令或其他需要主窗口先存在的组件很有用。

`main_window` 参数是 `MainWindow` 实例。

```python
@hookimpl
def main_window_ready(main_window):
    # Add a custom page to the main window
    from .my_page import MyCustomPage
    main_window.add_page("my-page", MyCustomPage())
```

## 注册钩子

这些钩子让您可以向 Rayforge 的各种注册表注册自定义组件。

### `register_machines(machine_manager)`

使用此钩子注册新的机器驱动程序。`machine_manager` 是管理所有机器配置的 `MachineManager` 实例。

```python
@hookimpl
def register_machines(machine_manager):
    from .my_driver import MyCustomMachine
    machine_manager.register("my_custom_machine", MyCustomMachine)
```

### `register_steps(step_registry)`

注册出现在操作面板中的自定义步骤类型。`step_registry` 是 `StepRegistry` 实例。

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)
```

### `register_producers(producer_registry)`

注册生成刀具路径的自定义 ops 生成器。`producer_registry` 是 `ProducerRegistry` 实例。

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyProducer
    producer_registry.register(MyProducer)
```

### `register_transformers(transformer_registry)`

注册用于后处理操作的自定义 ops 转换器。转换器在生成器生成操作后修改操作。`transformer_registry` 是 `TransformerRegistry` 实例。

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyTransformer
    transformer_registry.register(MyTransformer)
```

### `register_commands(command_registry)`

注册扩展文档编辑器功能的编辑器命令。`command_registry` 是 `CommandRegistry` 实例。

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand)
```

### `register_actions(action_registry)`

注册带有可选菜单和工具栏位置的窗口操作。操作是您添加按钮、菜单项和键盘快捷键的方式。`action_registry` 是 `ActionRegistry` 实例。

```python
from gi.repository import Gio
from rayforge.ui_gtk.action_registry import MenuPlacement, ToolbarPlacement

@hookimpl
def register_actions(action_registry):
    action = Gio.SimpleAction.new("my-action", None)
    action.connect("activate", on_my_action_activated)
    
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

### `register_layout_strategies(layout_registry)`

注册用于在文档中排列内容的自定义布局策略。`layout_registry` 是 `LayoutStrategyRegistry` 实例。请注意，标签和快捷键等 UI 元数据应通过 `register_actions` 注册，而不是在这里。

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(MyLayoutStrategy, name="my_layout")
```

### `register_asset_types(asset_type_registry)`

注册可以存储在文档中的自定义资源类型。这启用了插件提供的资源的动态反序列化。`asset_type_registry` 是 `AssetTypeRegistry` 实例。

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(MyCustomAsset, type_name="my_asset")
```

### `register_renderers(renderer_registry)`

注册在 UI 中显示您的资源类型的自定义渲染器。`renderer_registry` 是 `RendererRegistry` 实例。

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer())
```

### `register_exporters(exporter_registry)`

注册自定义导出格式的文件导出器。`exporter_registry` 是 `ExporterRegistry` 实例。

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter)
```

### `register_importers(importer_registry)`

注册自定义导入格式的文件导入器。`importer_registry` 是 `ImporterRegistry` 实例。

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter)
```

### `register_material_libraries(library_manager)`

注册额外的材料库。调用 `library_manager.add_library_from_path(path)` 来注册包含材料 YAML 文件的目录。默认情况下，注册的库是只读的。

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

## UI 扩展钩子

这些钩子让您扩展现有的 UI 组件。

### `step_settings_loaded(dialog, step, producer)`

当步骤设置对话框正在填充时，Rayforge 会调用此钩子。您可以根据步骤的生成器类型向对话框添加自定义小部件。

`dialog` 是 `GeneralStepSettingsView` 实例。`step` 是正在配置的 `Step`。`producer` 是 `OpsProducer` 实例，如果不可用则为 `None`。

```python
@hookimpl
def step_settings_loaded(dialog, step, producer):
    # Only add widgets for specific producer types
    if producer and producer.__class__.__name__ == "MyCustomProducer":
        from .my_widget import create_custom_widget
        dialog.add_widget(create_custom_widget(step))
```

### `transformer_settings_loaded(dialog, step, transformer)`

当后处理设置正在填充时调用。在这里为您的转换器添加自定义小部件。

`dialog` 是 `PostProcessingSettingsView` 实例。`step` 是正在配置的 `Step`。`transformer` 是 `OpsTransformer` 实例。

```python
@hookimpl
def transformer_settings_loaded(dialog, step, transformer):
    if transformer.__class__.__name__ == "MyCustomTransformer":
        from .my_widget import create_transformer_widget
        dialog.add_widget(create_transformer_widget(transformer))
```

## API 版本历史

钩子是有版本的，以保持向后兼容性。当添加新钩子或现有钩子发生变化时，API 版本会递增。您的插件的 `api_version` 字段必须至少是最低支持的版本。

当前 API 版本是 9。以下是最近版本的变更内容：

**版本 9** 添加了 `main_window_ready`、`register_exporters`、`register_importers` 和 `register_renderers`。

**版本 8** 添加了 `register_asset_types` 用于自定义资源类型。

**版本 7** 添加了 `register_material_libraries`。

**版本 6** 添加了 `register_transformers`。

**版本 5** 用 `step_settings_loaded` 和 `transformer_settings_loaded` 替换了 `register_step_widgets`。

**版本 4** 移除了 `register_menu_items` 并将操作注册整合到 `register_actions` 中。

**版本 2** 添加了 `register_layout_strategies`。

**版本 1** 是初始版本，包含插件生命周期、资源注册和 UI 集成的核心钩子。
