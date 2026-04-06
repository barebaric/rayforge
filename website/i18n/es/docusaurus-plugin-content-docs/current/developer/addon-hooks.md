# Hooks de Addons

Los hooks son los puntos de conexión entre tu addon y Rayforge. Cuando algo sucede en la aplicación—se crea un paso, se abre un diálogo o se inicializa la ventana—Rayforge llama a cualquier hook registrado para que tu addon pueda responder.

## Cómo Funcionan los Hooks

Rayforge usa [pluggy](https://pluggy.readthedocs.io/) para su sistema de hooks. Para implementar un hook, decora una función con `@pluggy.HookimplMarker("rayforge")`:

```python
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context):
    # Your code runs when Rayforge finishes initializing
    pass
```

No tienes que implementar cada hook—solo los que necesites. Todos los hooks son opcionales.

## Hooks del Ciclo de Vida

Estos hooks manejan el ciclo de vida general de tu addon.

### `rayforge_init(context)`

Este es tu punto de entrada principal. Rayforge llama a este hook después de que el contexto de la aplicación está completamente inicializado, lo que significa que todos los gestores, configuraciones y hardware están listos. Úsalo para configuración general, registro o inyección de elementos de interfaz de usuario.

El parámetro `context` es una instancia de `RayforgeContext` que te da acceso a todo en Rayforge. Consulta [Accediendo a los Datos de Rayforge](./addon-overview.md#accessing-rayforges-data) para más detalles.

```python
@hookimpl
def rayforge_init(context):
    logger.info("My addon is starting up!")
    machine = context.machine
    if machine:
        logger.info(f"Running on machine: {machine.id}")
```

### `on_unload()`

Rayforge llama a esto cuando tu addon está siendo deshabilitado o descargado. Úsalo para limpiar recursos, cerrar conexiones o desregistrar manejadores.

```python
@hookimpl
def on_unload():
    logger.info("My addon is shutting down")
    # Clean up any resources here
```

### `main_window_ready(main_window)`

Este hook se dispara cuando la ventana principal está completamente inicializada. Es útil para registrar páginas de interfaz de usuario, comandos u otros componentes que necesitan que la ventana principal exista primero.

El parámetro `main_window` es la instancia de `MainWindow`.

```python
@hookimpl
def main_window_ready(main_window):
    # Add a custom page to the main window
    from .my_page import MyCustomPage
    main_window.add_page("my-page", MyCustomPage())
```

## Hooks de Registro

Estos hooks te permiten registrar componentes personalizados con los diversos registros de Rayforge.

### `register_machines(machine_manager)`

Úsalo para registrar nuevos controladores de máquinas. El `machine_manager` es una instancia de `MachineManager` que gestiona todas las configuraciones de máquinas.

```python
@hookimpl
def register_machines(machine_manager):
    from .my_driver import MyCustomMachine
    machine_manager.register("my_custom_machine", MyCustomMachine)
```

### `register_steps(step_registry)`

Registra tipos de pasos personalizados que aparecen en el panel de operaciones. El `step_registry` es una instancia de `StepRegistry`.

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)
```

### `register_producers(producer_registry)`

Registra productores de ops personalizados que generan trayectorias de herramientas. El `producer_registry` es una instancia de `ProducerRegistry`.

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyProducer
    producer_registry.register(MyProducer)
```

### `register_transformers(transformer_registry)`

Registra transformadores de ops personalizados para operaciones de post-procesamiento. Los transformadores modifican las operaciones después de que los productores las generan. El `transformer_registry` es una instancia de `TransformerRegistry`.

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyTransformer
    transformer_registry.register(MyTransformer)
```

### `register_commands(command_registry)`

Registra comandos del editor que extienden la funcionalidad del editor de documentos. El `command_registry` es una instancia de `CommandRegistry`.

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand)
```

### `register_actions(action_registry)`

Registra acciones de ventana con colocación opcional en menú y barra de herramientas. Las acciones son cómo agregas botones, elementos de menú y atajos de teclado. El `action_registry` es una instancia de `ActionRegistry`.

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

Registra estrategias de diseño personalizadas para organizar contenido en el documento. El `layout_registry` es una instancia de `LayoutStrategyRegistry`. Ten en cuenta que los metadatos de interfaz de usuario como etiquetas y atajos deben registrarse mediante `register_actions`, no aquí.

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(MyLayoutStrategy, name="my_layout")
```

### `register_asset_types(asset_type_registry)`

Registra tipos de assets personalizados que pueden almacenarse en documentos. Esto habilita la deserialización dinámica de assets proporcionados por addons. El `asset_type_registry` es una instancia de `AssetTypeRegistry`.

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(MyCustomAsset, type_name="my_asset")
```

### `register_renderers(renderer_registry)`

Registra renderizadores personalizados para mostrar tus tipos de assets en la interfaz de usuario. El `renderer_registry` es una instancia de `RendererRegistry`.

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer())
```

### `register_exporters(exporter_registry)`

Registra exportadores de archivos para formatos de exportación personalizados. El `exporter_registry` es una instancia de `ExporterRegistry`.

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter)
```

### `register_importers(importer_registry)`

Registra importadores de archivos para formatos de importación personalizados. El `importer_registry` es una instancia de `ImporterRegistry`.

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter)
```

### `register_material_libraries(library_manager)`

Registra bibliotecas de materiales adicionales. Llama a `library_manager.add_library_from_path(path)` para registrar directorios que contienen archivos YAML de materiales. Por defecto, las bibliotecas registradas son de solo lectura.

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

## Hooks de Extensión de Interfaz de Usuario

Estos hooks te permiten extender componentes de interfaz de usuario existentes.

### `step_settings_loaded(dialog, step, producer)`

Rayforge llama a esto cuando se está poblando un diálogo de configuración de pasos. Puedes agregar widgets personalizados al diálogo basándote en el tipo de productor del paso.

El `dialog` es una instancia de `GeneralStepSettingsView`. El `step` es el `Step` que se está configurando. El `producer` es la instancia de `OpsProducer`, o `None` si no está disponible.

```python
@hookimpl
def step_settings_loaded(dialog, step, producer):
    # Only add widgets for specific producer types
    if producer and producer.__class__.__name__ == "MyCustomProducer":
        from .my_widget import create_custom_widget
        dialog.add_widget(create_custom_widget(step))
```

### `transformer_settings_loaded(dialog, step, transformer)`

Se llama cuando se están poblando las configuraciones de post-procesamiento. Agrega widgets personalizados para tus transformadores aquí.

El `dialog` es una instancia de `PostProcessingSettingsView`. El `step` es el `Step` que se está configurando. El `transformer` es la instancia de `OpsTransformer`.

```python
@hookimpl
def transformer_settings_loaded(dialog, step, transformer):
    if transformer.__class__.__name__ == "MyCustomTransformer":
        from .my_widget import create_transformer_widget
        dialog.add_widget(create_transformer_widget(transformer))
```

## Historial de Versiones de API

Los hooks están versionados para mantener compatibilidad hacia atrás. Cuando se agregan nuevos hooks o los existentes cambian, la versión de API se incrementa. El campo `api_version` de tu addon debe ser al menos la versión mínima soportada.

La versión actual de API es 9. Esto es lo que cambió en versiones recientes:

**La versión 9** agregó `main_window_ready`, `register_exporters`, `register_importers` y `register_renderers`.

**La versión 8** agregó `register_asset_types` para tipos de assets personalizados.

**La versión 7** agregó `register_material_libraries`.

**La versión 6** agregó `register_transformers`.

**La versión 5** reemplazó `register_step_widgets` con `step_settings_loaded` y `transformer_settings_loaded`.

**La versión 4** eliminó `register_menu_items` y consolidó el registro de acciones en `register_actions`.

**La versión 2** agregó `register_layout_strategies`.

**La versión 1** fue el lanzamiento inicial con hooks principales para el ciclo de vida del addon, registro de recursos e integración de interfaz de usuario.
