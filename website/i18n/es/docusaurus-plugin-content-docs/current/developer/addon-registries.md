# Registros de Addons

Los registros son cómo Rayforge gestiona la extensibilidad. Cada registro contiene una colección de componentes relacionados—pasos, productores, acciones, etc. Cuando tu addon registra algo, se vuelve disponible en toda la aplicación.

## Cómo Funcionan los Registros

Todos los registros siguen un patrón similar. Proporcionan un método `register()` para agregar elementos, y varios métodos de búsqueda para recuperarlos. La mayoría de los registros también rastrean qué addon registró cada elemento, para que puedan limpiar cuando un addon se descarga.

Aquí está el patrón general:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

El parámetro `addon_name` es opcional pero recomendado. Asegura que tus componentes se eliminen correctamente si el usuario deshabilita tu addon.

## Registro de Pasos

El registro de pasos (`StepRegistry`) gestiona tipos de pasos que aparecen en el panel de operaciones. Cada paso representa un tipo de operación que los usuarios pueden agregar a su trabajo.

### Registrando un Paso

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

El nombre de clase del paso se usa como clave del registro. Tu clase de paso debe heredar de `Step` y definir atributos como `TYPELABEL`, `HIDDEN`, e implementar el método de clase `create()`.

### Recuperando Pasos

El registro proporciona varios métodos para buscar pasos:

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

## Registro de Productores

El registro de productores (`ProducerRegistry`) gestiona productores de ops. Los productores generan las operaciones de trayectoria de herramientas para un paso—esencialmente, convierten tu pieza de trabajo en instrucciones de máquina.

### Registrando un Productor

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyCustomProducer
    producer_registry.register(MyCustomProducer, addon_name="my_addon")
```

Por defecto, el nombre de la clase se convierte en la clave del registro. Puedes especificar un nombre personalizado:

```python
producer_registry.register(MyCustomProducer, name="custom_name", addon_name="my_addon")
```

### Recuperando Productores

```python
# Get a producer by name
producer_class = producer_registry.get("MyCustomProducer")

# Get all producers
all_producers = producer_registry.all_producers()
```

## Registro de Transformadores

El registro de transformadores (`TransformerRegistry`) gestiona transformadores de ops. Los transformadores post-procesan las operaciones después de que los productores las generan—piensa en tareas como optimización de rutas, suavizado o agregar pestañas de sujeción.

### Registrando un Transformador

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyCustomTransformer
    transformer_registry.register(MyCustomTransformer, addon_name="my_addon")
```

### Recuperando Transformadores

```python
# Get a transformer by name
transformer_class = transformer_registry.get("MyCustomTransformer")

# Get all transformers
all_transformers = transformer_registry.all_transformers()
```

## Registro de Acciones

El registro de acciones (`ActionRegistry`) gestiona acciones de ventana. Las acciones son cómo agregas elementos de menú, botones de barra de herramientas y atajos de teclado. Este es uno de los registros más ricos en funciones.

### Registrando una Acción

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

### Parámetros de Acción

Al registrar una acción, puedes proporcionar:

- `action_name`: El identificador de la acción (sin el prefijo "win.")
- `action`: La instancia de `Gio.SimpleAction`
- `addon_name`: El nombre de tu addon para limpieza
- `label`: Texto legible para humanos para menús y tooltips
- `icon_name`: Identificador de icono para barras de herramientas
- `shortcut`: Atajo de teclado usando sintaxis de acelerador GTK
- `menu`: Objeto `MenuPlacement` especificando qué menú y prioridad
- `toolbar`: Objeto `ToolbarPlacement` especificando grupo de barra de herramientas y prioridad

### Colocación en Menú

La clase `MenuPlacement` toma:

- `menu_id`: A qué menú agregar (ej., "tools", "arrange")
- `priority`: Números más bajos aparecen primero

### Colocación en Barra de Herramientas

La clase `ToolbarPlacement` toma:

- `group`: Identificador de grupo de barra de herramientas (ej., "main", "arrange")
- `priority`: Números más bajos aparecen primero

### Recuperando Acciones

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

## Registro de Comandos

El registro de comandos (`CommandRegistry`) gestiona comandos del editor. Los comandos extienden la funcionalidad del editor de documentos.

### Registrando un Comando

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand, addon_name="my_addon")
```

Las clases de comandos deben aceptar una instancia de `DocEditor` en su constructor.

### Recuperando Comandos

```python
# Get a command by name
command_class = command_registry.get("my_command")

# Get all commands
all_commands = command_registry.all_commands()
```

## Registro de Tipos de Assets

El registro de tipos de assets (`AssetTypeRegistry`) gestiona tipos de assets que pueden almacenarse en documentos. Esto habilita la deserialización dinámica—cuando Rayforge carga un documento que contiene tu asset personalizado, sabe cómo reconstruirlo.

### Registrando un Tipo de Asset

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

El `type_name` es la cadena usada en documentos serializados para identificar tu tipo de asset.

### Recuperando Tipos de Assets

```python
# Get an asset class by type name
asset_class = asset_type_registry.get("my_asset")

# Get all registered asset types
all_types = asset_type_registry.all_types()
```

## Registro de Estrategias de Diseño

El registro de estrategias de diseño (`LayoutStrategyRegistry`) gestiona estrategias de diseño para organizar contenido en el editor de documentos.

### Registrando una Estrategia de Diseño

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

Ten en cuenta que los metadatos de interfaz de usuario como etiquetas y atajos deben registrarse mediante el registro de acciones, no aquí.

### Recuperando Estrategias de Diseño

```python
# Get a strategy by name
strategy_class = layout_registry.get("my_layout")

# Get all strategy classes
all_strategies = layout_registry.list_all()

# Get all strategy names
strategy_names = layout_registry.list_names()
```

## Registro de Importadores

El registro de importadores (`ImporterRegistry`) gestiona importadores de archivos. Los importadores manejan la carga de archivos externos en Rayforge.

### Registrando un Importador

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter, addon_name="my_addon")
```

Tu clase de importador debe definir atributos de clase `extensions` y `mime_types` para que el registro sepa qué archivos maneja.

### Recuperando Importadores

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

## Registro de Exportadores

El registro de exportadores (`ExporterRegistry`) gestiona exportadores de archivos. Los exportadores manejan el guardado de documentos u operaciones de Rayforge a formatos externos.

### Registrando un Exportador

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter, addon_name="my_addon")
```

Tu clase de exportador debe definir atributos de clase `extensions` y `mime_types`.

### Recuperando Exportadores

```python
# Get exporter by file extension
exporter_class = exporter_registry.get_by_extension(".xyz")

# Get exporter by MIME type
exporter_class = exporter_registry.get_by_mime_type("application/x-xyz")

# Get all file filters for file dialogs
filters = exporter_registry.get_all_filters()
```

## Registro de Renderizadores

El registro de renderizadores (`RendererRegistry`) gestiona renderizadores de assets. Los renderizadores muestran assets en la interfaz de usuario.

### Registrando un Renderizador

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer(), addon_name="my_addon")
```

Ten en cuenta que registras una instancia de renderizador, no una clase. El nombre de clase del renderizador se usa como clave del registro.

### Recuperando Renderizadores

```python
# Get renderer by class name
renderer = renderer_registry.get("MyAssetRenderer")

# Get renderer by name (same as get)
renderer = renderer_registry.get_by_name("MyAssetRenderer")

# Get all renderers
all_renderers = renderer_registry.all()
```

## Gestor de Bibliotecas

El gestor de bibliotecas (`LibraryManager`) gestiona bibliotecas de materiales. Aunque no es técnicamente un registro, sigue patrones similares para registrar bibliotecas proporcionadas por addons.

### Registrando una Biblioteca de Materiales

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

Las bibliotecas registradas son de solo lectura por defecto. Los usuarios pueden ver y usar los materiales pero no pueden modificarlos a través de la interfaz de usuario.
