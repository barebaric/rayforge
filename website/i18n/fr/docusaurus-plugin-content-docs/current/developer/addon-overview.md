# Vue d'ensemble du développement d'extensions

Rayforge utilise un système d'extensions basé sur [pluggy](https://pluggy.readthedocs.io/) qui vous permet d'étendre les fonctionnalités, d'ajouter de nouveaux pilotes de machine ou d'intégrer une logique personnalisée sans modifier le code source principal.

## Démarrage rapide

La façon la plus rapide de commencer est d'utiliser le modèle officiel [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template). Forkez ou clonez-le, renommez le répertoire et mettez à jour les métadonnées pour correspondre à votre extension.

## Fonctionnement des extensions

L'`AddonManager` scanne le répertoire `addons` à la recherche d'extensions valides. Une extension est simplement un répertoire contenant un fichier manifeste `rayforge-addon.yaml` ainsi que votre code Python.

Voici à quoi ressemble une extension typique :

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Manifeste requis
├── my_addon/            <-- Votre package Python
│   ├── __init__.py
│   ├── backend.py       <-- Point d'entrée backend
│   └── frontend.py      <-- Point d'entrée frontend (optionnel)
├── assets/              <-- Ressources optionnelles
├── locales/             <-- Traductions optionnelles (fichiers .po)
└── README.md
```

## Votre première extension

Créons une extension simple qui enregistre un pilote de machine personnalisé. D'abord, créez le manifeste :

```yaml title="rayforge-addon.yaml"
name: my_laser_driver
display_name: "My Laser Driver"
description: "Adds support for the XYZ laser cutter."
api_version: 9

author:
  name: Jane Doe
  email: jane@example.com

provides:
  backend: my_addon.backend
```

Maintenant créez le module backend qui enregistre votre pilote :

```python title="my_addon/backend.py"
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def register_machines(machine_manager):
    """Register our custom machine driver."""
    from .my_driver import MyLaserMachine
    machine_manager.register("my_laser", MyLaserMachine)
```

C'est tout ! Votre extension sera maintenant chargée au démarrage de Rayforge, et votre pilote de machine sera disponible pour les utilisateurs.

La documentation du [Manifeste](./addon-manifest.md) couvre toutes les options de configuration disponibles.

## Comprendre les points d'entrée

Les extensions peuvent fournir deux points d'entrée, chacun chargé à des moments différents :

Le point d'entrée **backend** se charge à la fois dans le processus principal et dans les processus de travail. Utilisez-le pour les pilotes de machine, les types d'étapes, les producteurs et transformateurs d'ops, ou toute fonctionnalité principale qui n'a pas besoin de dépendances UI.

Le point d'entrée **frontend** ne se charge que dans le processus principal. C'est ici que vous mettez les composants UI, les widgets GTK, les éléments de menu et tout ce qui a besoin d'accéder à la fenêtre principale.

Les deux sont spécifiés comme des chemins de modules avec points comme `my_addon.backend`.

## Connexion à Rayforge avec les hooks

Rayforge utilise les hooks `pluggy` pour permettre aux extensions de s'intégrer à l'application. Décorez simplement vos fonctions avec `@pluggy.HookimplMarker("rayforge")` :

```python
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context: RayforgeContext):
    """Called when Rayforge is fully initialized."""
    # Your setup code here
    pass

@hookimpl
def on_unload():
    """Called when the addon is being disabled or unloaded."""
    # Clean up resources here
    pass
```

La documentation des [Hooks](./addon-hooks.md) décrit chaque hook disponible et quand il est appelé.

## Enregistrement de vos composants

La plupart des hooks reçoivent un objet registre que vous utilisez pour enregistrer vos composants personnalisés :

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)

@hookimpl
def register_actions(action_registry):
    from .actions import setup_actions
    setup_actions(action_registry)
```

La documentation des [Registres](./addon-registries.md) explique chaque registre et comment les utiliser.

## Accéder aux données de Rayforge

Le hook `rayforge_init` vous donne accès à un objet `RayforgeContext`. Via ce contexte, vous pouvez accéder à tout dans Rayforge :

Vous pouvez obtenir la machine actuellement active via `context.machine`, ou accéder à toutes les machines via `context.machine_mgr`. L'objet `context.config` contient les paramètres globaux, tandis que `context.camera_mgr` fournit l'accès aux flux vidéo. Pour les matériaux, utilisez `context.material_mgr`, et pour les recettes de traitement, utilisez `context.recipe_mgr`. Le gestionnaire de dialectes G-code est disponible via `context.dialect_mgr`, et les fonctionnalités IA passent par `context.ai_provider_mgr`. Pour la localisation, consultez `context.language` pour le code de langue actuel. Le gestionnaire d'extensions lui-même est disponible via `context.addon_mgr`, et si vous créez des extensions payantes, `context.license_validator` gère la validation des licences.

## Ajouter des traductions

Les extensions peuvent fournir des traductions en utilisant des fichiers `.po` standards. Organisez-les comme ceci :

```text
my-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── my_addon.po
│   └── es/
│       └── LC_MESSAGES/
│           └── my_addon.po
```

Rayforge compile automatiquement les fichiers `.po` en fichiers `.mo` lorsque votre extension est chargée.

## Tester pendant le développement

Pour tester votre extension localement, créez un lien symbolique depuis votre dossier de développement vers le répertoire d'extensions de Rayforge.

D'abord, trouvez votre répertoire de configuration. Sur Windows, c'est `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`. Sur macOS, cherchez dans `~/Library/Application Support/rayforge/addons`. Sur Linux, c'est `~/.config/rayforge/addons`.

Ensuite créez le lien symbolique :

```bash
ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
```

Redémarrez Rayforge et vérifiez la console pour un message comme `Loaded addon: my_laser_driver`.

## Partager votre extension

Quand vous êtes prêt à partager votre extension, poussez-la vers un dépôt Git public sur GitHub ou GitLab. Ensuite soumettez-la au [rayforge-registry](https://github.com/barebaric/rayforge-registry) en forkant le dépôt, en ajoutant les métadonnées de votre extension, et en ouvrant une pull request.

Une fois acceptée, les utilisateurs pourront installer votre extension directement via le gestionnaire d'extensions de Rayforge.
