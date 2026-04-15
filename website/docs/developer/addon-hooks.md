---
description: "Add-on hooks in Rayforge - lifecycle events and extension points for integrating custom functionality into the laser cutting workflow."
---

# Addon Hooks

Hooks are the connection points between your addon and Rayforge. When something happens in the application—a step is created, a dialog opens, or the window initializes—Rayforge calls any registered hooks so your addon can respond.

## How Hooks Work

Rayforge uses [pluggy](https://pluggy.readthedocs.io/) for its hook system. To implement a hook, decorate a function with `@pluggy.HookimplMarker("rayforge")`:

```python
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context):
    # Your code runs when Rayforge finishes initializing
    pass
```

You don't have to implement every hook—just the ones you need. All hooks are optional.

## Lifecycle Hooks

These hooks handle the overall lifecycle of your addon.

### `rayforge_init(context)`

This is your main entry point. Rayforge calls this hook after the application context is fully initialized, meaning all managers, configs, and hardware are ready. Use this for general setup, logging, or injecting UI elements.

The `context` parameter is a `RayforgeContext` instance that gives you access to everything in Rayforge. See [Accessing Rayforge Data](./addon-overview.md#accessing-rayforges-data) for details.

```python
@hookimpl
def rayforge_init(context):
    logger.info("My addon is starting up!")
    machine = context.machine
    if machine:
        logger.info(f"Running on machine: {machine.id}")
```

### `on_unload()`

Rayforge calls this when your addon is being disabled or unloaded. Use it to clean up resources, close connections, or unregister handlers.

```python
@hookimpl
def on_unload():
    logger.info("My addon is shutting down")
    # Clean up any resources here
```

### `main_window_ready(main_window)`

This hook fires when the main window is fully initialized. It's useful for registering UI pages, commands, or other components that need the main window to exist first.

The `main_window` parameter is the `MainWindow` instance.

```python
@hookimpl
def main_window_ready(main_window):
    # Add a custom page to the main window
    from .my_page import MyCustomPage
    main_window.add_page("my-page", MyCustomPage())
```

## Registration Hooks

These hooks let you register custom components with Rayforge's various registries.

### `register_machines(machine_manager)`

Use this to register new machine drivers. The `machine_manager` is a `MachineManager` instance that manages all machine configurations.

```python
@hookimpl
def register_machines(machine_manager):
    from .my_driver import MyCustomMachine
    machine_manager.register("my_custom_machine", MyCustomMachine)
```

### `register_steps(step_registry)`

Register custom step types that appear in the operations panel. The `step_registry` is a `StepRegistry` instance.

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)
```

### `register_producers(producer_registry)`

Register custom ops producers that generate toolpaths. The `producer_registry` is a `ProducerRegistry` instance.

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyProducer
    producer_registry.register(MyProducer)
```

### `register_transformers(transformer_registry)`

Register custom ops transformers for post-processing operations. Transformers modify operations after producers generate them. The `transformer_registry` is a `TransformerRegistry` instance.

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyTransformer
    transformer_registry.register(MyTransformer)
```

### `register_commands(command_registry)`

Register editor commands that extend the document editor's functionality. The `command_registry` is a `CommandRegistry` instance.

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand)
```

### `register_actions(action_registry)`

Register window actions with optional menu and toolbar placement. Actions are how you add buttons, menu items, and keyboard shortcuts. The `action_registry` is an `ActionRegistry` instance.

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

Register custom layout strategies for arranging content in the document. The `layout_registry` is a `LayoutStrategyRegistry` instance. Note that UI metadata like labels and shortcuts should be registered via `register_actions`, not here.

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(MyLayoutStrategy, name="my_layout")
```

### `register_asset_types(asset_type_registry)`

Register custom asset types that can be stored in documents. This enables dynamic deserialization of addon-provided assets. The `asset_type_registry` is an `AssetTypeRegistry` instance.

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(MyCustomAsset, type_name="my_asset")
```

### `register_renderers(renderer_registry)`

Register custom renderers for displaying your asset types in the UI. The `renderer_registry` is a `RendererRegistry` instance.

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer())
```

### `register_exporters(exporter_registry)`

Register file exporters for custom export formats. The `exporter_registry` is an `ExporterRegistry` instance.

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter)
```

### `register_importers(importer_registry)`

Register file importers for custom import formats. The `importer_registry` is an `ImporterRegistry` instance.

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter)
```

### `register_material_libraries(library_manager)`

Register additional material libraries. Call `library_manager.add_library_from_path(path)` to register directories containing material YAML files. By default, registered libraries are read-only.

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

## UI Extension Hooks

These hooks let you extend existing UI components.

### `step_settings_loaded(dialog, step, producer)`

Rayforge calls this when a step settings dialog is being populated. You can add custom widgets to the dialog based on the step's producer type.

The `dialog` is a `GeneralStepSettingsView` instance. The `step` is the `Step` being configured. The `producer` is the `OpsProducer` instance, or `None` if not available.

```python
@hookimpl
def step_settings_loaded(dialog, step, producer):
    # Only add widgets for specific producer types
    if producer and producer.__class__.__name__ == "MyCustomProducer":
        from .my_widget import create_custom_widget
        dialog.add_widget(create_custom_widget(step))
```

### `transformer_settings_loaded(dialog, step, transformer)`

Called when post-processing settings are being populated. Add custom widgets for your transformers here.

The `dialog` is a `PostProcessingSettingsView` instance. The `step` is the `Step` being configured. The `transformer` is the `OpsTransformer` instance.

```python
@hookimpl
def transformer_settings_loaded(dialog, step, transformer):
    if transformer.__class__.__name__ == "MyCustomTransformer":
        from .my_widget import create_transformer_widget
        dialog.add_widget(create_transformer_widget(transformer))
```

## API Version History

Hooks are versioned to maintain backwards compatibility. When new hooks are added or existing ones change, the API version is incremented. Your addon's `api_version` field must be at least the minimum supported version.

The current API version is 9. Here's what changed in recent versions:

**Version 9** added `main_window_ready`, `register_exporters`, `register_importers`, and `register_renderers`.

**Version 8** added `register_asset_types` for custom asset types.

**Version 7** added `register_material_libraries`.

**Version 6** added `register_transformers`.

**Version 5** replaced `register_step_widgets` with `step_settings_loaded` and `transformer_settings_loaded`.

**Version 4** removed `register_menu_items` and consolidated action registration into `register_actions`.

**Version 2** added `register_layout_strategies`.

**Version 1** was the initial release with core hooks for addon lifecycle, resource registration, and UI integration.
