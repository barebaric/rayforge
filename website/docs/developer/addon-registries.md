---
description: "Publish and discover add-ons through Rayforge registries. Share your extensions with the laser cutting community."
---

# Addon Registries

Registries are how Rayforge manages extensibility. Each registry holds a collection of related components—steps, producers, actions, and so on. When your addon registers something, it becomes available throughout the application.

## How Registries Work

All registries follow a similar pattern. They provide a `register()` method to add items, and various lookup methods to retrieve them. Most registries also track which addon registered each item, so they can clean up when an addon is unloaded.

Here's the general pattern:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

The `addon_name` parameter is optional but recommended. It ensures your components are properly removed if the user disables your addon.

## Step Registry

The step registry (`StepRegistry`) manages step types that appear in the operations panel. Each step represents a type of operation users can add to their job.

### Registering a Step

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

The step's class name is used as the registry key. Your step class should inherit from `Step` and define attributes like `TYPELABEL`, `HIDDEN`, and implement the `create()` class method.

### Retrieving Steps

The registry provides several methods for looking up steps:

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

## Producer Registry

The producer registry (`ProducerRegistry`) manages ops producers. Producers generate the toolpath operations for a step—essentially, they convert your workpiece into machine instructions.

### Registering a Producer

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyCustomProducer
    producer_registry.register(MyCustomProducer, addon_name="my_addon")
```

By default, the class name becomes the registry key. You can specify a custom name:

```python
producer_registry.register(MyCustomProducer, name="custom_name", addon_name="my_addon")
```

### Retrieving Producers

```python
# Get a producer by name
producer_class = producer_registry.get("MyCustomProducer")

# Get all producers
all_producers = producer_registry.all_producers()
```

## Transformer Registry

The transformer registry (`TransformerRegistry`) manages ops transformers. Transformers post-process operations after producers generate them—think of tasks like path optimization, smoothing, or adding holding tabs.

### Registering a Transformer

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyCustomTransformer
    transformer_registry.register(MyCustomTransformer, addon_name="my_addon")
```

### Retrieving Transformers

```python
# Get a transformer by name
transformer_class = transformer_registry.get("MyCustomTransformer")

# Get all transformers
all_transformers = transformer_registry.all_transformers()
```

## Action Registry

The action registry (`ActionRegistry`) manages window actions. Actions are how you add menu items, toolbar buttons, and keyboard shortcuts. This is one of the more feature-rich registries.

### Registering an Action

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

### Action Parameters

When registering an action, you can provide:

- `action_name`: The action's identifier (without the "win." prefix)
- `action`: The `Gio.SimpleAction` instance
- `addon_name`: Your addon's name for cleanup
- `label`: Human-readable text for menus and tooltips
- `icon_name`: Icon identifier for toolbars
- `shortcut`: Keyboard shortcut using GTK accelerator syntax
- `menu`: `MenuPlacement` object specifying which menu and priority
- `toolbar`: `ToolbarPlacement` object specifying toolbar group and priority

### Menu Placement

The `MenuPlacement` class takes:

- `menu_id`: Which menu to add to (e.g., "tools", "arrange")
- `priority`: Lower numbers appear first

### Toolbar Placement

The `ToolbarPlacement` class takes:

- `group`: Toolbar group identifier (e.g., "main", "arrange")
- `priority`: Lower numbers appear first

### Retrieving Actions

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

## Command Registry

The command registry (`CommandRegistry`) manages editor commands. Commands extend the document editor's functionality.

### Registering a Command

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand, addon_name="my_addon")
```

Command classes should accept a `DocEditor` instance in their constructor.

### Retrieving Commands

```python
# Get a command by name
command_class = command_registry.get("my_command")

# Get all commands
all_commands = command_registry.all_commands()
```

## Asset Type Registry

The asset type registry (`AssetTypeRegistry`) manages asset types that can be stored in documents. This enables dynamic deserialization—when Rayforge loads a document containing your custom asset, it knows how to reconstruct it.

### Registering an Asset Type

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

The `type_name` is the string used in serialized documents to identify your asset type.

### Retrieving Asset Types

```python
# Get an asset class by type name
asset_class = asset_type_registry.get("my_asset")

# Get all registered asset types
all_types = asset_type_registry.all_types()
```

## Layout Strategy Registry

The layout strategy registry (`LayoutStrategyRegistry`) manages layout strategies for arranging content in the document editor.

### Registering a Layout Strategy

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

Note that UI metadata like labels and shortcuts should be registered via the action registry, not here.

### Retrieving Layout Strategies

```python
# Get a strategy by name
strategy_class = layout_registry.get("my_layout")

# Get all strategy classes
all_strategies = layout_registry.list_all()

# Get all strategy names
strategy_names = layout_registry.list_names()
```

## Importer Registry

The importer registry (`ImporterRegistry`) manages file importers. Importers handle loading external files into Rayforge.

### Registering an Importer

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter, addon_name="my_addon")
```

Your importer class should define `extensions` and `mime_types` class attributes so the registry knows which files it handles.

### Retrieving Importers

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

## Exporter Registry

The exporter registry (`ExporterRegistry`) manages file exporters. Exporters handle saving Rayforge documents or operations to external formats.

### Registering an Exporter

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter, addon_name="my_addon")
```

Your exporter class should define `extensions` and `mime_types` class attributes.

### Retrieving Exporters

```python
# Get exporter by file extension
exporter_class = exporter_registry.get_by_extension(".xyz")

# Get exporter by MIME type
exporter_class = exporter_registry.get_by_mime_type("application/x-xyz")

# Get all file filters for file dialogs
filters = exporter_registry.get_all_filters()
```

## Renderer Registry

The renderer registry (`RendererRegistry`) manages asset renderers. Renderers display assets in the UI.

### Registering a Renderer

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer(), addon_name="my_addon")
```

Note that you register a renderer instance, not a class. The renderer's class name is used as the registry key.

### Retrieving Renderers

```python
# Get renderer by class name
renderer = renderer_registry.get("MyAssetRenderer")

# Get renderer by name (same as get)
renderer = renderer_registry.get_by_name("MyAssetRenderer")

# Get all renderers
all_renderers = renderer_registry.all()
```

## Library Manager

The library manager (`LibraryManager`) manages material libraries. While not technically a registry, it follows similar patterns for registering addon-provided libraries.

### Registering a Material Library

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

Registered libraries are read-only by default. Users can view and use the materials but cannot modify them through the UI.
