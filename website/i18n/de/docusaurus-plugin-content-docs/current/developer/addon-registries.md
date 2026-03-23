# Addon-Registries

Registries sind die Art, wie Rayforge Erweiterbarkeit verwaltet. Jede Registry hält eine Sammlung verwandter Komponenten – Schritte, Produzenten, Aktionen und so weiter. Wenn dein Addon etwas registriert, wird es in der gesamten Anwendung verfügbar.

## Wie Registries funktionieren

Alle Registries folgen einem ähnlichen Muster. Sie stellen eine `register()`-Methode zum Hinzufügen von Elementen und verschiedene Suchmethoden zum Abrufen bereit. Die meisten Registries verfolgen auch, welches Addon jedes Element registriert hat, damit sie bereinigen können, wenn ein Addon entladen wird.

Hier ist das allgemeine Muster:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

Der Parameter `addon_name` ist optional, aber empfohlen. Er stellt sicher, dass deine Komponenten ordnungsgemäß entfernt werden, wenn der Benutzer dein Addon deaktiviert.

## Schritt-Registry

Die Schritt-Registry (`StepRegistry`) verwaltet Schritttypen, die im Bedienfeld für Operationen erscheinen. Jeder Schritt repräsentiert einen Typ von Operation, den Benutzer zu ihrem Auftrag hinzufügen können.

### Einen Schritt registrieren

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

Der Klassenname des Schritts wird als Registrierungsschlüssel verwendet. Deine Schrittklasse sollte von `Step` erben und Attribute wie `TYPELABEL`, `HIDDEN` definieren und die Klassenmethode `create()` implementieren.

### Schritte abrufen

Die Registry bietet mehrere Methoden zum Nachschlagen von Schritten:

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

## Produzenten-Registry

Die Produzenten-Registry (`ProducerRegistry`) verwaltet Ops-Produzenten. Produzenten generieren die Werkzeugweg-Operationen für einen Schritt – im Wesentlichen konvertieren sie dein Werkstück in Maschinenanweisungen.

### Einen Produzenten registrieren

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyCustomProducer
    producer_registry.register(MyCustomProducer, addon_name="my_addon")
```

Standardmäßig wird der Klassenname zum Registrierungsschlüssel. Du kannst einen benutzerdefinierten Namen angeben:

```python
producer_registry.register(MyCustomProducer, name="custom_name", addon_name="my_addon")
```

### Produzenten abrufen

```python
# Get a producer by name
producer_class = producer_registry.get("MyCustomProducer")

# Get all producers
all_producers = producer_registry.all_producers()
```

## Transformatoren-Registry

Die Transformatoren-Registry (`TransformerRegistry`) verwaltet Ops-Transformatoren. Transformatoren bearbeiten Operationen nach, nachdem Produzenten sie generiert haben – denke an Aufgaben wie Pfadoptimierung, Glättung oder Haltelaschen hinzufügen.

### Einen Transformatoren registrieren

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyCustomTransformer
    transformer_registry.register(MyCustomTransformer, addon_name="my_addon")
```

### Transformatoren abrufen

```python
# Get a transformer by name
transformer_class = transformer_registry.get("MyCustomTransformer")

# Get all transformers
all_transformers = transformer_registry.all_transformers()
```

## Aktions-Registry

Die Aktions-Registry (`ActionRegistry`) verwaltet Fensteraktionen. Aktionen sind die Art, wie du Menüeinträge, Toolbar-Buttons und Tastaturkürzel hinzufügst. Dies ist eine der funktionsreicheren Registries.

### Eine Aktion registrieren

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

### Aktionsparameter

Beim Registrieren einer Aktion kannst du angeben:

- `action_name`: Die Kennung der Aktion (ohne das Präfix "win.")
- `action`: Die `Gio.SimpleAction`-Instanz
- `addon_name`: Der Name deines Addons für die Bereinigung
- `label`: Lesbarer Text für Menüs und Tooltips
- `icon_name`: Symbolkennung für Toolbars
- `shortcut`: Tastaturkürzel mit GTK-Akzelerator-Syntax
- `menu`: `MenuPlacement`-Objekt, das Menü und Priorität angibt
- `toolbar`: `ToolbarPlacement`-Objekt, das Toolbar-Gruppe und Priorität angibt

### Menü-Platzierung

Die Klasse `MenuPlacement` akzeptiert:

- `menu_id`: Zu welchem Menü hinzugefügt werden soll (z. B. "tools", "arrange")
- `priority`: Niedrigere Zahlen erscheinen zuerst

### Toolbar-Platzierung

Die Klasse `ToolbarPlacement` akzeptiert:

- `group`: Toolbar-Gruppenkennung (z. B. "main", "arrange")
- `priority`: Niedrigere Zahlen erscheinen zuerst

### Aktionen abrufen

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

## Befehls-Registry

Die Befehls-Registry (`CommandRegistry`) verwaltet Editor-Befehle. Befehle erweitern die Funktionalität des Dokumenteditors.

### Einen Befehl registrieren

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand, addon_name="my_addon")
```

Befehlsklassen sollten eine `DocEditor`-Instanz in ihrem Konstruktor akzeptieren.

### Befehle abrufen

```python
# Get a command by name
command_class = command_registry.get("my_command")

# Get all commands
all_commands = command_registry.all_commands()
```

## Asset-Typ-Registry

Die Asset-Typ-Registry (`AssetTypeRegistry`) verwaltet Asset-Typen, die in Dokumenten gespeichert werden können. Dies ermöglicht dynamische Deserialisierung – wenn Rayforge ein Dokument lädt, das dein benutzerdefiniertes Asset enthält, weiß es, wie es rekonstruiert wird.

### Einen Asset-Typ registrieren

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

Der `type_name` ist die Zeichenfolge, die in serialisierten Dokumenten verwendet wird, um deinen Asset-Typ zu identifizieren.

### Asset-Typen abrufen

```python
# Get an asset class by type name
asset_class = asset_type_registry.get("my_asset")

# Get all registered asset types
all_types = asset_type_registry.all_types()
```

## Layout-Strategie-Registry

Die Layout-Strategie-Registry (`LayoutStrategyRegistry`) verwaltet Layout-Strategien zum Anordnen von Inhalten im Dokumenteditor.

### Eine Layout-Strategie registrieren

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

Beachte, dass UI-Metadaten wie Beschriftungen und Kürzel über die Aktions-Registry registriert werden sollten, nicht hier.

### Layout-Strategien abrufen

```python
# Get a strategy by name
strategy_class = layout_registry.get("my_layout")

# Get all strategy classes
all_strategies = layout_registry.list_all()

# Get all strategy names
strategy_names = layout_registry.list_names()
```

## Importer-Registry

Die Importer-Registry (`ImporterRegistry`) verwaltet Datei-Importer. Importer behandeln das Laden externer Dateien in Rayforge.

### Einen Importer registrieren

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter, addon_name="my_addon")
```

Deine Importer-Klasse sollte Klassenattribute `extensions` und `mime_types` definieren, damit die Registry weiß, welche Dateien sie behandelt.

### Importer abrufen

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

## Exporter-Registry

Die Exporter-Registry (`ExporterRegistry`) verwaltet Datei-Exporter. Exporter behandeln das Speichern von Rayforge-Dokumenten oder Operationen in externe Formate.

### Einen Exporter registrieren

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter, addon_name="my_addon")
```

Deine Exporter-Klasse sollte Klassenattribute `extensions` und `mime_types` definieren.

### Exporter abrufen

```python
# Get exporter by file extension
exporter_class = exporter_registry.get_by_extension(".xyz")

# Get exporter by MIME type
exporter_class = exporter_registry.get_by_mime_type("application/x-xyz")

# Get all file filters for file dialogs
filters = exporter_registry.get_all_filters()
```

## Renderer-Registry

Die Renderer-Registry (`RendererRegistry`) verwaltet Asset-Renderer. Renderer zeigen Assets in der UI an.

### Einen Renderer registrieren

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer(), addon_name="my_addon")
```

Beachte, dass du eine Renderer-Instanz registrierst, nicht eine Klasse. Der Klassenname des Renderers wird als Registrierungsschlüssel verwendet.

### Renderer abrufen

```python
# Get renderer by class name
renderer = renderer_registry.get("MyAssetRenderer")

# Get renderer by name (same as get)
renderer = renderer_registry.get_by_name("MyAssetRenderer")

# Get all renderers
all_renderers = renderer_registry.all()
```

## Bibliotheks-Manager

Der Bibliotheks-Manager (`LibraryManager`) verwaltet Materialbibliotheken. Obwohl technisch keine Registry, folgt er ähnlichen Mustern zum Registrieren von Addon-bereitgestellten Bibliotheken.

### Eine Materialbibliothek registrieren

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

Registrierte Bibliotheken sind standardmäßig schreibgeschützt. Benutzer können die Materialien ansehen und verwenden, aber sie können sie nicht über die UI ändern.
