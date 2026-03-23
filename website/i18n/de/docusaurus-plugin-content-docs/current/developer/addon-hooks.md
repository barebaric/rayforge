# Addon-Hooks

Hooks sind die Verbindungspunkte zwischen deinem Addon und Rayforge. Wenn in der Anwendung etwas passiert – ein Schritt wird erstellt, ein Dialog öffnet sich oder das Fenster initialisiert sich – ruft Rayforge alle registrierten Hooks auf, damit dein Addon reagieren kann.

## Wie Hooks funktionieren

Rayforge verwendet [pluggy](https://pluggy.readthedocs.io/) für sein Hook-System. Um einen Hook zu implementieren, dekoriere eine Funktion mit `@pluggy.HookimplMarker("rayforge")`:

```python
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context):
    # Your code runs when Rayforge finishes initializing
    pass
```

Du musst nicht jeden Hook implementieren – nur die, die du benötigst. Alle Hooks sind optional.

## Lebenszyklus-Hooks

Diese Hooks behandeln den gesamten Lebenszyklus deines Addons.

### `rayforge_init(context)`

Dies ist dein Haupteinstiegspunkt. Rayforge ruft diesen Hook auf, nachdem der Anwendungskontext vollständig initialisiert wurde, was bedeutet, dass alle Manager, Konfigurationen und Hardware bereit sind. Verwende diesen für allgemeines Setup, Logging oder das Einfügen von UI-Elementen.

Der Parameter `context` ist eine `RayforgeContext`-Instanz, die dir Zugriff auf alles in Rayforge gibt. Siehe [Auf Rayforge-Daten zugreifen](./addon-overview.md#accessing-rayforges-data) für Details.

```python
@hookimpl
def rayforge_init(context):
    logger.info("My addon is starting up!")
    machine = context.machine
    if machine:
        logger.info(f"Running on machine: {machine.id}")
```

### `on_unload()`

Rayforge ruft dies auf, wenn dein Addon deaktiviert oder entladen wird. Verwende es, um Ressourcen freizugeben, Verbindungen zu schließen oder Handler abzumelden.

```python
@hookimpl
def on_unload():
    logger.info("My addon is shutting down")
    # Clean up any resources here
```

### `main_window_ready(main_window)`

Dieser Hook wird ausgelöst, wenn das Hauptfenster vollständig initialisiert ist. Er ist nützlich zum Registrieren von UI-Seiten, Befehlen oder anderen Komponenten, die erfordern, dass das Hauptfenster zuerst existiert.

Der Parameter `main_window` ist die `MainWindow`-Instanz.

```python
@hookimpl
def main_window_ready(main_window):
    # Add a custom page to the main window
    from .my_page import MyCustomPage
    main_window.add_page("my-page", MyCustomPage())
```

## Registrierungs-Hooks

Diese Hooks ermöglichen es dir, benutzerdefinierte Komponenten bei Rayforges verschiedenen Registrierungen zu registrieren.

### `register_machines(machine_manager)`

Verwende diesen, um neue Maschinentreiber zu registrieren. Der `machine_manager` ist eine `MachineManager`-Instanz, die alle Maschinenkonfigurationen verwaltet.

```python
@hookimpl
def register_machines(machine_manager):
    from .my_driver import MyCustomMachine
    machine_manager.register("my_custom_machine", MyCustomMachine)
```

### `register_steps(step_registry)`

Registriere benutzerdefinierte Schritttypen, die im Bedienfeld für Operationen erscheinen. Der `step_registry` ist eine `StepRegistry`-Instanz.

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)
```

### `register_producers(producer_registry)`

Registriere benutzerdefinierte Ops-Produzenten, die Werkzeugwege generieren. Der `producer_registry` ist eine `ProducerRegistry`-Instanz.

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyProducer
    producer_registry.register(MyProducer)
```

### `register_transformers(transformer_registry)`

Registriere benutzerdefinierte Ops-Transformatoren für Nachbearbeitungsvorgänge. Transformatoren modifizieren Operationen, nachdem Produzenten sie generiert haben. Der `transformer_registry` ist eine `TransformerRegistry`-Instanz.

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyTransformer
    transformer_registry.register(MyTransformer)
```

### `register_commands(command_registry)`

Registriere Editor-Befehle, die die Funktionalität des Dokumenteditors erweitern. Der `command_registry` ist eine `CommandRegistry`-Instanz.

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand)
```

### `register_actions(action_registry)`

Registriere Fensteraktionen mit optionaler Menü- und Toolbar-Platzierung. Aktionen sind die Art, wie du Buttons, Menüeinträge und Tastaturkürzel hinzufügst. Der `action_registry` ist eine `ActionRegistry`-Instanz.

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

Registriere benutzerdefinierte Layout-Strategien zum Anordnen von Inhalten im Dokument. Der `layout_registry` ist eine `LayoutStrategyRegistry`-Instanz. Beachte, dass UI-Metadaten wie Beschriftungen und Kürzel über `register_actions` registriert werden sollten, nicht hier.

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(MyLayoutStrategy, name="my_layout")
```

### `register_asset_types(asset_type_registry)`

Registriere benutzerdefinierte Asset-Typen, die in Dokumenten gespeichert werden können. Dies ermöglicht die dynamische Deserialisierung von Addon-bereitgestellten Assets. Der `asset_type_registry` ist eine `AssetTypeRegistry`-Instanz.

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(MyCustomAsset, type_name="my_asset")
```

### `register_renderers(renderer_registry)`

Registriere benutzerdefinierte Renderer zum Anzeigen deiner Asset-Typen in der UI. Der `renderer_registry` ist eine `RendererRegistry`-Instanz.

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer())
```

### `register_exporters(exporter_registry)`

Registriere Datei-Exporter für benutzerdefinierte Exportformate. Der `exporter_registry` ist eine `ExporterRegistry`-Instanz.

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter)
```

### `register_importers(importer_registry)`

Registriere Datei-Importer für benutzerdefinierte Importformate. Der `importer_registry` ist eine `ImporterRegistry`-Instanz.

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter)
```

### `register_material_libraries(library_manager)`

Registriere zusätzliche Materialbibliotheken. Rufe `library_manager.add_library_from_path(path)` auf, um Verzeichnisse mit Material-YAML-Dateien zu registrieren. Standardmäßig sind registrierte Bibliotheken schreibgeschützt.

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

## UI-Erweiterungs-Hooks

Diese Hooks ermöglichen es dir, bestehende UI-Komponenten zu erweitern.

### `step_settings_loaded(dialog, step, producer)`

Rayforge ruft dies auf, wenn ein Schritteinstellungsdialog gefüllt wird. Du kannst basierend auf dem Produzententyp des Schritts benutzerdefinierte Widgets zum Dialog hinzufügen.

Der `dialog` ist eine `GeneralStepSettingsView`-Instanz. Der `step` ist der konfigurierte `Step`. Der `producer` ist die `OpsProducer`-Instanz oder `None`, wenn nicht verfügbar.

```python
@hookimpl
def step_settings_loaded(dialog, step, producer):
    # Only add widgets for specific producer types
    if producer and producer.__class__.__name__ == "MyCustomProducer":
        from .my_widget import create_custom_widget
        dialog.add_widget(create_custom_widget(step))
```

### `transformer_settings_loaded(dialog, step, transformer)`

Wird aufgerufen, wenn Nachbearbeitungseinstellungen gefüllt werden. Füge hier benutzerdefinierte Widgets für deine Transformatoren hinzu.

Der `dialog` ist eine `PostProcessingSettingsView`-Instanz. Der `step` ist der konfigurierte `Step`. Der `transformer` ist die `OpsTransformer`-Instanz.

```python
@hookimpl
def transformer_settings_loaded(dialog, step, transformer):
    if transformer.__class__.__name__ == "MyCustomTransformer":
        from .my_widget import create_transformer_widget
        dialog.add_widget(create_transformer_widget(transformer))
```

## API-Versionshistorie

Hooks sind versioniert, um die Abwärtskompatibilität zu gewährleisten. Wenn neue Hooks hinzugefügt werden oder sich bestehende ändern, wird die API-Version erhöht. Das Feld `api_version` deines Addons muss mindestens die minimal unterstützte Version sein.

Die aktuelle API-Version ist 9. Hier ist, was sich in den letzten Versionen geändert hat:

**Version 9** fügte `main_window_ready`, `register_exporters`, `register_importers` und `register_renderers` hinzu.

**Version 8** fügte `register_asset_types` für benutzerdefinierte Asset-Typen hinzu.

**Version 7** fügte `register_material_libraries` hinzu.

**Version 6** fügte `register_transformers` hinzu.

**Version 5** ersetzte `register_step_widgets` durch `step_settings_loaded` und `transformer_settings_loaded`.

**Version 4** entfernte `register_menu_items` und konsolidierte die Aktionsregistrierung in `register_actions`.

**Version 2** fügte `register_layout_strategies` hinzu.

**Version 1** war die erste Veröffentlichung mit Kern-Hooks für Addon-Lebenszyklus, Ressourcenregistrierung und UI-Integration.
