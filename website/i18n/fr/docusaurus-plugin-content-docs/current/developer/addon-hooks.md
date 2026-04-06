# Hooks d'extension

Les hooks sont les points de connexion entre votre extension et Rayforge. Quand quelque chose se produit dans l'application — une étape est créée, une boîte de dialogue s'ouvre, ou la fenêtre s'initialise — Rayforge appelle tous les hooks enregistrés pour que votre extension puisse réagir.

## Fonctionnement des hooks

Rayforge utilise [pluggy](https://pluggy.readthedocs.io/) pour son système de hooks. Pour implémenter un hook, décorez une fonction avec `@pluggy.HookimplMarker("rayforge")` :

```python
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context):
    # Your code runs when Rayforge finishes initializing
    pass
```

Vous n'avez pas à implémenter tous les hooks — seulement ceux dont vous avez besoin. Tous les hooks sont optionnels.

## Hooks de cycle de vie

Ces hooks gèrent le cycle de vie global de votre extension.

### `rayforge_init(context)`

C'est votre point d'entrée principal. Rayforge appelle ce hook après que le contexte de l'application est entièrement initialisé, ce qui signifie que tous les gestionnaires, configurations et matériels sont prêts. Utilisez-le pour la configuration générale, la journalisation ou l'injection d'éléments UI.

Le paramètre `context` est une instance de `RayforgeContext` qui vous donne accès à tout dans Rayforge. Consultez [Accéder aux données de Rayforge](./addon-overview.md#accessing-rayforges-data) pour plus de détails.

```python
@hookimpl
def rayforge_init(context):
    logger.info("My addon is starting up!")
    machine = context.machine
    if machine:
        logger.info(f"Running on machine: {machine.id}")
```

### `on_unload()`

Rayforge appelle ceci quand votre extension est désactivée ou déchargée. Utilisez-le pour nettoyer les ressources, fermer les connexions ou désenregistrer les gestionnaires.

```python
@hookimpl
def on_unload():
    logger.info("My addon is shutting down")
    # Clean up any resources here
```

### `main_window_ready(main_window)`

Ce hook se déclenche quand la fenêtre principale est entièrement initialisée. Il est utile pour enregistrer des pages UI, des commandes ou d'autres composants qui ont besoin que la fenêtre principale existe d'abord.

Le paramètre `main_window` est l'instance de `MainWindow`.

```python
@hookimpl
def main_window_ready(main_window):
    # Add a custom page to the main window
    from .my_page import MyCustomPage
    main_window.add_page("my-page", MyCustomPage())
```

## Hooks d'enregistrement

Ces hooks vous permettent d'enregistrer des composants personnalisés auprès des différents registres de Rayforge.

### `register_machines(machine_manager)`

Utilisez ce hook pour enregistrer de nouveaux pilotes de machine. Le `machine_manager` est une instance de `MachineManager` qui gère toutes les configurations de machines.

```python
@hookimpl
def register_machines(machine_manager):
    from .my_driver import MyCustomMachine
    machine_manager.register("my_custom_machine", MyCustomMachine)
```

### `register_steps(step_registry)`

Enregistrez des types d'étapes personnalisés qui apparaissent dans le panneau des opérations. Le `step_registry` est une instance de `StepRegistry`.

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)
```

### `register_producers(producer_registry)`

Enregistrez des producteurs d'ops personnalisés qui génèrent des parcours d'outils. Le `producer_registry` est une instance de `ProducerRegistry`.

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyProducer
    producer_registry.register(MyProducer)
```

### `register_transformers(transformer_registry)`

Enregistrez des transformateurs d'ops personnalisés pour le post-traitement des opérations. Les transformateurs modifient les opérations après que les producteurs les ont générées. Le `transformer_registry` est une instance de `TransformerRegistry`.

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyTransformer
    transformer_registry.register(MyTransformer)
```

### `register_commands(command_registry)`

Enregistrez des commandes d'éditeur qui étendent les fonctionnalités de l'éditeur de documents. Le `command_registry` est une instance de `CommandRegistry`.

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand)
```

### `register_actions(action_registry)`

Enregistrez des actions de fenêtre avec placement optionnel dans les menus et barres d'outils. Les actions sont la façon d'ajouter des boutons, des éléments de menu et des raccourcis clavier. Le `action_registry` est une instance de `ActionRegistry`.

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

Enregistrez des stratégies de disposition personnalisées pour arranger le contenu dans le document. Le `layout_registry` est une instance de `LayoutStrategyRegistry`. Notez que les métadonnées UI comme les labels et raccourcis doivent être enregistrées via `register_actions`, pas ici.

```python
@hookimpl
def register_layout_strategies(layout_registry):
    from .my_layout import MyLayoutStrategy
    layout_registry.register(MyLayoutStrategy, name="my_layout")
```

### `register_asset_types(asset_type_registry)`

Enregistrez des types de ressources personnalisés qui peuvent être stockés dans les documents. Cela permet la désérialisation dynamique des ressources fournies par les extensions. Le `asset_type_registry` est une instance de `AssetTypeRegistry`.

```python
@hookimpl
def register_asset_types(asset_type_registry):
    from .my_asset import MyCustomAsset
    asset_type_registry.register(MyCustomAsset, type_name="my_asset")
```

### `register_renderers(renderer_registry)`

Enregistrez des rendereurs personnalisés pour afficher vos types de ressources dans l'interface. Le `renderer_registry` est une instance de `RendererRegistry`.

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer())
```

### `register_exporters(exporter_registry)`

Enregistrez des exportateurs de fichiers pour des formats d'export personnalisés. Le `exporter_registry` est une instance de `ExporterRegistry`.

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter)
```

### `register_importers(importer_registry)`

Enregistrez des importateurs de fichiers pour des formats d'import personnalisés. Le `importer_registry` est une instance de `ImporterRegistry`.

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter)
```

### `register_material_libraries(library_manager)`

Enregistrez des bibliothèques de matériaux supplémentaires. Appelez `library_manager.add_library_from_path(path)` pour enregistrer des répertoires contenant des fichiers YAML de matériaux. Par défaut, les bibliothèques enregistrées sont en lecture seule.

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

## Hooks d'extension UI

Ces hooks vous permettent d'étendre les composants UI existants.

### `step_settings_loaded(dialog, step, producer)`

Rayforge appelle ceci quand une boîte de dialogue de paramètres d'étape est en cours de remplissage. Vous pouvez ajouter des widgets personnalisés à la boîte de dialogue en fonction du type de producteur de l'étape.

Le paramètre `dialog` est une instance de `GeneralStepSettingsView`. Le paramètre `step` est l'instance `Step` en cours de configuration. Le paramètre `producer` est l'instance `OpsProducer`, ou `None` si non disponible.

```python
@hookimpl
def step_settings_loaded(dialog, step, producer):
    # Only add widgets for specific producer types
    if producer and producer.__class__.__name__ == "MyCustomProducer":
        from .my_widget import create_custom_widget
        dialog.add_widget(create_custom_widget(step))
```

### `transformer_settings_loaded(dialog, step, transformer)`

Appelé quand les paramètres de post-traitement sont en cours de remplissage. Ajoutez des widgets personnalisés pour vos transformateurs ici.

Le paramètre `dialog` est une instance de `PostProcessingSettingsView`. Le paramètre `step` est l'instance `Step` en cours de configuration. Le paramètre `transformer` est l'instance `OpsTransformer`.

```python
@hookimpl
def transformer_settings_loaded(dialog, step, transformer):
    if transformer.__class__.__name__ == "MyCustomTransformer":
        from .my_widget import create_transformer_widget
        dialog.add_widget(create_transformer_widget(transformer))
```

## Historique des versions de l'API

Les hooks sont versionnés pour maintenir la compatibilité ascendante. Quand de nouveaux hooks sont ajoutés ou que des hooks existants changent, la version de l'API est incrémentée. Le champ `api_version` de votre extension doit être au moins égal à la version minimale supportée.

La version actuelle de l'API est 9. Voici ce qui a changé dans les versions récentes :

**La version 9** a ajouté `main_window_ready`, `register_exporters`, `register_importers` et `register_renderers`.

**La version 8** a ajouté `register_asset_types` pour les types de ressources personnalisés.

**La version 7** a ajouté `register_material_libraries`.

**La version 6** a ajouté `register_transformers`.

**La version 5** a remplacé `register_step_widgets` par `step_settings_loaded` et `transformer_settings_loaded`.

**La version 4** a supprimé `register_menu_items` et consolidé l'enregistrement des actions dans `register_actions`.

**La version 2** a ajouté `register_layout_strategies`.

**La version 1** était la version initiale avec les hooks principaux pour le cycle de vie des extensions, l'enregistrement des ressources et l'intégration UI.
