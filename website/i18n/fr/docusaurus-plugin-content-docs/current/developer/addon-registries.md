# Registres d'extension

Les registres sont la façon dont Rayforge gère l'extensibilité. Chaque registre contient une collection de composants apparentés — étapes, producteurs, actions, etc. Quand votre extension enregistre quelque chose, il devient disponible dans toute l'application.

## Fonctionnement des registres

Tous les registres suivent un modèle similaire. Ils fournissent une méthode `register()` pour ajouter des éléments, et diverses méthodes de recherche pour les récupérer. La plupart des registres suivent également quelle extension a enregistré chaque élément, afin de pouvoir nettoyer quand une extension est déchargée.

Voici le modèle général :

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

Le paramètre `addon_name` est optionnel mais recommandé. Il garantit que vos composants sont correctement supprimés si l'utilisateur désactive votre extension.

## Registre des étapes

Le registre des étapes (`StepRegistry`) gère les types d'étapes qui apparaissent dans le panneau des opérations. Chaque étape représente un type d'opération que les utilisateurs peuvent ajouter à leur travail.

### Enregistrer une étape

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep, addon_name="my_addon")
```

Le nom de classe de l'étape est utilisé comme clé de registre. Votre classe d'étape doit hériter de `Step` et définir des attributs comme `TYPELABEL`, `HIDDEN`, et implémenter la méthode de classe `create()`.

### Récupérer des étapes

Le registre fournit plusieurs méthodes pour rechercher des étapes :

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

## Registre des producteurs

Le registre des producteurs (`ProducerRegistry`) gère les producteurs d'ops. Les producteurs génèrent les opérations de parcours d'outils pour une étape — essentiellement, ils convertissent votre pièce en instructions machine.

### Enregistrer un producteur

```python
@hookimpl
def register_producers(producer_registry):
    from .my_producer import MyCustomProducer
    producer_registry.register(MyCustomProducer, addon_name="my_addon")
```

Par défaut, le nom de classe devient la clé de registre. Vous pouvez spécifier un nom personnalisé :

```python
producer_registry.register(MyCustomProducer, name="custom_name", addon_name="my_addon")
```

### Récupérer des producteurs

```python
# Get a producer by name
producer_class = producer_registry.get("MyCustomProducer")

# Get all producers
all_producers = producer_registry.all_producers()
```

## Registre des transformateurs

Le registre des transformateurs (`TransformerRegistry`) gère les transformateurs d'ops. Les transformateurs post-traitent les opérations après que les producteurs les ont générées — pensez à des tâches comme l'optimisation de parcours, le lissage ou l'ajout de pattes de maintien.

### Enregistrer un transformateur

```python
@hookimpl
def register_transformers(transformer_registry):
    from .my_transformer import MyCustomTransformer
    transformer_registry.register(MyCustomTransformer, addon_name="my_addon")
```

### Récupérer des transformateurs

```python
# Get a transformer by name
transformer_class = transformer_registry.get("MyCustomTransformer")

# Get all transformers
all_transformers = transformer_registry.all_transformers()
```

## Registre des actions

Le registre des actions (`ActionRegistry`) gère les actions de fenêtre. Les actions sont la façon d'ajouter des éléments de menu, des boutons de barre d'outils et des raccourcis clavier. C'est l'un des registres les plus riches en fonctionnalités.

### Enregistrer une action

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

### Paramètres d'action

Lors de l'enregistrement d'une action, vous pouvez fournir :

- `action_name` : L'identifiant de l'action (sans le préfixe "win.")
- `action` : L'instance `Gio.SimpleAction`
- `addon_name` : Le nom de votre extension pour le nettoyage
- `label` : Texte lisible par l'humain pour les menus et infobulles
- `icon_name` : Identifiant d'icône pour les barres d'outils
- `shortcut` : Raccourci clavier utilisant la syntaxe d'accélérateur GTK
- `menu` : Objet `MenuPlacement` spécifiant quel menu et la priorité
- `toolbar` : Objet `ToolbarPlacement` spécifiant le groupe de barre d'outils et la priorité

### Placement dans le menu

La classe `MenuPlacement` prend :

- `menu_id` : Dans quel menu ajouter (ex: "tools", "arrange")
- `priority` : Les numéros les plus bas apparaissent en premier

### Placement dans la barre d'outils

La classe `ToolbarPlacement` prend :

- `group` : Identifiant du groupe de barre d'outils (ex: "main", "arrange")
- `priority` : Les numéros les plus bas apparaissent en premier

### Récupérer des actions

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

## Registre des commandes

Le registre des commandes (`CommandRegistry`) gère les commandes de l'éditeur. Les commandes étendent les fonctionnalités de l'éditeur de documents.

### Enregistrer une commande

```python
@hookimpl
def register_commands(command_registry):
    from .commands import MyCustomCommand
    command_registry.register("my_command", MyCustomCommand, addon_name="my_addon")
```

Les classes de commande doivent accepter une instance `DocEditor` dans leur constructeur.

### Récupérer des commandes

```python
# Get a command by name
command_class = command_registry.get("my_command")

# Get all commands
all_commands = command_registry.all_commands()
```

## Registre des types de ressources

Le registre des types de ressources (`AssetTypeRegistry`) gère les types de ressources qui peuvent être stockés dans les documents. Cela permet la désérialisation dynamique — quand Rayforge charge un document contenant votre ressource personnalisée, il sait comment la reconstruire.

### Enregistrer un type de ressource

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

Le `type_name` est la chaîne utilisée dans les documents sérialisés pour identifier votre type de ressource.

### Récupérer des types de ressources

```python
# Get an asset class by type name
asset_class = asset_type_registry.get("my_asset")

# Get all registered asset types
all_types = asset_type_registry.all_types()
```

## Registre des stratégies de disposition

Le registre des stratégies de disposition (`LayoutStrategyRegistry`) gère les stratégies de disposition pour arranger le contenu dans l'éditeur de documents.

### Enregistrer une stratégie de disposition

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

Notez que les métadonnées UI comme les labels et raccourcis doivent être enregistrées via le registre des actions, pas ici.

### Récupérer des stratégies de disposition

```python
# Get a strategy by name
strategy_class = layout_registry.get("my_layout")

# Get all strategy classes
all_strategies = layout_registry.list_all()

# Get all strategy names
strategy_names = layout_registry.list_names()
```

## Registre des importateurs

Le registre des importateurs (`ImporterRegistry`) gère les importateurs de fichiers. Les importateurs gèrent le chargement de fichiers externes dans Rayforge.

### Enregistrer un importateur

```python
@hookimpl
def register_importers(importer_registry):
    from .my_importer import MyCustomImporter
    importer_registry.register(MyCustomImporter, addon_name="my_addon")
```

Votre classe d'importateur doit définir les attributs de classe `extensions` et `mime_types` pour que le registre sache quels fichiers il gère.

### Récupérer des importateurs

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

## Registre des exportateurs

Le registre des exportateurs (`ExporterRegistry`) gère les exportateurs de fichiers. Les exportateurs gèrent l'enregistrement des documents ou opérations Rayforge vers des formats externes.

### Enregistrer un exportateur

```python
@hookimpl
def register_exporters(exporter_registry):
    from .my_exporter import MyCustomExporter
    exporter_registry.register(MyCustomExporter, addon_name="my_addon")
```

Votre classe d'exportateur doit définir les attributs de classe `extensions` et `mime_types`.

### Récupérer des exportateurs

```python
# Get exporter by file extension
exporter_class = exporter_registry.get_by_extension(".xyz")

# Get exporter by MIME type
exporter_class = exporter_registry.get_by_mime_type("application/x-xyz")

# Get all file filters for file dialogs
filters = exporter_registry.get_all_filters()
```

## Registre des rendereurs

Le registre des rendereurs (`RendererRegistry`) gère les rendereurs de ressources. Les rendereurs affichent les ressources dans l'interface.

### Enregistrer un rendereur

```python
@hookimpl
def register_renderers(renderer_registry):
    from .my_renderer import MyAssetRenderer
    renderer_registry.register(MyAssetRenderer(), addon_name="my_addon")
```

Notez que vous enregistrez une instance de rendereur, pas une classe. Le nom de classe du rendereur est utilisé comme clé de registre.

### Récupérer des rendereurs

```python
# Get renderer by class name
renderer = renderer_registry.get("MyAssetRenderer")

# Get renderer by name (same as get)
renderer = renderer_registry.get_by_name("MyAssetRenderer")

# Get all renderers
all_renderers = renderer_registry.all()
```

## Gestionnaire de bibliothèques

Le gestionnaire de bibliothèques (`LibraryManager`) gère les bibliothèques de matériaux. Bien que techniquement pas un registre, il suit des modèles similaires pour enregistrer les bibliothèques fournies par les extensions.

### Enregistrer une bibliothèque de matériaux

```python
@hookimpl
def register_material_libraries(library_manager):
    from pathlib import Path
    lib_path = Path(__file__).parent / "materials"
    library_manager.add_library_from_path(lib_path)
```

Les bibliothèques enregistrées sont en lecture seule par défaut. Les utilisateurs peuvent voir et utiliser les matériaux mais ne peuvent pas les modifier via l'interface.
