# Développement d'addons

Rayforge utilise un système d'addons basé sur [pluggy](https://pluggy.readthedocs.io/)
pour permettre aux développeurs d'étendre les fonctionnalités, d'ajouter de nouveaux pilotes machine, ou
d'intégrer une logique personnalisée sans modifier le code source principal.

## 1. Démarrage rapide

La façon la plus rapide de commencer est d'utiliser le modèle officiel.

1. **Forkez ou clonez** le [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template).
2. **Renommez** le répertoire et mettez à jour les métadonnées.

## 2. Structure de l'addon

Le `AddonManager` scanne le répertoire `addons`. Un addon valide doit être un
répertoire contenant un fichier manifeste :

**Disposition du répertoire :**

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Manifeste requis
├── my_addon/            <-- Package Python
│   ├── __init__.py
│   ├── backend.py       <-- Point d'entrée backend
│   └── frontend.py      <-- Point d'entrée frontend (optionnel)
├── assets/              <-- Ressources optionnelles
├── locales/             <-- Traductions optionnelles (fichiers .po)
└── README.md
```

## 3. Le manifeste (`rayforge-addon.yaml`)

Ce fichier indique à Rayforge comment charger votre addon.

```yaml
# rayforge-addon.yaml

# Identifiant unique pour votre addon (nom du répertoire)
name: my_custom_addon

# Nom d'affichage lisible par l'homme
display_name: "Mon Addon Personnalisé"

# Description affichée dans l'interface
description: "Ajoute le support pour le découpeur laser XYZ."

# Version de l'API (doit correspondre au PLUGIN_API_VERSION de Rayforge)
api_version: 1

# Dépendances de version Rayforge
depends:
  - rayforge>=0.27.0,<2.0.0

# Optionnel : Dépendances d'autres addons
requires:
  - some-other-addon>=1.0.0

# Ce que l'addon fournit
provides:
  # Module backend (chargé dans les processus principal et worker)
  backend: my_addon.backend
  # Module frontend (chargé uniquement dans le processus principal, pour l'UI)
  frontend: my_addon.frontend
  # Fichiers assets optionnels
  assets:
    - path: assets/profiles.json
      type: profiles

# Métadonnées de l'auteur
author:
  name: Jean Dupont
  email: jean@example.com

url: https://github.com/username/my-custom-addon
```

### Champs requis

- `name` : Identifiant unique (doit correspondre au nom du répertoire)
- `display_name` : Nom lisible affiché dans l'interface
- `description` : Brève description de la fonctionnalité de l'addon
- `api_version` : Doit être `1` (correspond au `PLUGIN_API_VERSION` de Rayforge)
- `depends` : Liste des contraintes de version pour Rayforge
- `author` : Objet avec `name` (requis) et `email` (optionnel)

### Champs optionnels

- `requires` : Liste des dépendances d'autres addons
- `provides` : Points d'entrée et assets
- `url` : Page du projet ou dépôt

## 4. Points d'entrée

Les addons peuvent fournir deux types de points d'entrée :

### Backend (`provides.backend`)

Chargé dans le processus principal et les processus worker. Utilisez-le pour :
- Pilotes de machine
- Types d'étapes
- Producteurs d'ops
- Fonctionnalité principale sans dépendances UI

### Frontend (`provides.frontend`)

Chargé uniquement dans le processus principal. Utilisez-le pour :
- Composants UI
- Widgets GTK
- Éléments de menu
- Actions nécessitant la fenêtre principale

Les points d'entrée sont spécifiés comme des chemins de modules avec points (ex., `my_addon.backend`).

## 5. Écrire le code de l'addon

Rayforge utilise des hooks `pluggy`. Pour vous intégrer à Rayforge, définissez des fonctions décorées
avec `@pluggy.HookimplMarker("rayforge")`.

### Code de base (`backend.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Appelé lorsque Rayforge est entièrement initialisé.
    C'est votre point d'entrée principal pour accéder aux gestionnaires.
    """
    logger.info("Mon Addon Personnalisé a démarré !")

    machine = context.machine
    if machine:
        logger.info(f"Addon exécuté sur la machine : {machine.id}")

@hookimpl
def on_unload():
    """
    Appelé lorsque l'addon est désactivé ou déchargé.
    Nettoyer les ressources, fermer les connexions, désenregistrer les handlers.
    """
    logger.info("Mon Addon Personnalisé s'arrête")

@hookimpl
def register_machines(machine_manager):
    """
    Appelé au démarrage pour enregistrer de nouveaux pilotes machine.
    """
    from .my_driver import MyNewMachine
    machine_manager.register("my_new_machine", MyNewMachine)

@hookimpl
def register_steps(step_registry):
    """
    Appelé pour enregistrer des types d'étapes personnalisés.
    """
    from .my_step import MyCustomStep
    step_registry.register("my_custom_step", MyCustomStep)

@hookimpl
def register_producers(producer_registry):
    """
    Appelé pour enregistrer des producteurs d'ops personnalisés.
    """
    from .my_producer import MyProducer
    producer_registry.register("my_producer", MyProducer)

@hookimpl
def register_step_widgets(widget_registry):
    """
    Appelé pour enregistrer des widgets de paramètres d'étape personnalisés.
    """
    from .my_widget import MyStepWidget
    widget_registry.register("my_custom_step", MyStepWidget)

@hookimpl
def register_menu_items(menu_registry):
    """
    Appelé pour enregistrer des éléments de menu.
    """
    from .menu_items import register_menus
    register_menus(menu_registry)

@hookimpl
def register_commands(command_registry):
    """
    Appelé pour enregistrer des commandes d'éditeur.
    """
    from .commands import register_commands
    register_commands(command_registry)

@hookimpl
def register_actions(window):
    """
    Appelé pour enregistrer des actions de fenêtre.
    """
    from .actions import setup_actions
    setup_actions(window)
```

### Hooks disponibles

Définis dans `rayforge/core/hooks.py` :

**`rayforge_init`** (`context`)
: **Point d'entrée principal.** Appelé après le chargement de la configuration, de la caméra et du matériel. Utilisez-le pour la logique, les injections UI ou les écouteurs.

**`on_unload`** ()
: Appelé lorsqu'un addon est désactivé ou déchargé. Utilisez-le pour nettoyer
  les ressources, fermer les connexions, désenregistrer les handlers, etc.

**`register_machines`** (`machine_manager`)
: Appelé au démarrage pour enregistrer de nouveaux pilotes machine.

**`register_steps`** (`step_registry`)
: Appelé pour permettre aux plugins d'enregistrer des types d'étapes personnalisés.

**`register_producers`** (`producer_registry`)
: Appelé pour permettre aux plugins d'enregistrer des producteurs d'ops personnalisés.

**`register_step_widgets`** (`widget_registry`)
: Appelé pour permettre aux plugins d'enregistrer des widgets de paramètres d'étape personnalisés.

**`register_menu_items`** (`menu_registry`)
: Appelé pour permettre aux plugins d'enregistrer des éléments de menu.

**`register_commands`** (`command_registry`)
: Appelé pour permettre aux plugins d'enregistrer des commandes d'éditeur.

**`register_actions`** (`window`)
: Appelé pour permettre aux plugins d'enregistrer des actions de fenêtre.

## 6. Accéder aux données Rayforge

Le hook `rayforge_init` fournit le **`RayforgeContext`**. Via cet objet,
vous pouvez accéder à :

- **`context.machine`** : L'instance machine actuellement active.
- **`context.config`** : Paramètres de configuration globaux.
- **`context.config_mgr`** : Gestionnaire de configuration.
- **`context.machine_mgr`** : Gestionnaire de machines (toutes les machines).
- **`context.camera_mgr`** : Accès aux flux caméra et outils de vision par ordinateur.
- **`context.material_mgr`** : Accès à la bibliothèque de matériaux.
- **`context.recipe_mgr`** : Accès aux recettes de traitement.
- **`context.dialect_mgr`** : Gestionnaire de dialectes G-code.
- **`context.language`** : Code de langue actuel pour le contenu localisé.
- **`context.addon_mgr`** : Instance du gestionnaire d'addons.
- **`context.plugin_mgr`** : Instance du gestionnaire de plugins.
- **`context.debug_dump_manager`** : Gestionnaire de dumps de débogage.
- **`context.artifact_store`** : Stockage d'artefacts du pipeline.

## 7. Localisation

Les addons peuvent fournir des traductions en utilisant des fichiers `.po` :

```text
my-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── my_addon.po
│   └── fr/
│       └── LC_MESSAGES/
│           └── my_addon.po
```

Les fichiers `.po` sont automatiquement compilés en fichiers `.mo` lorsque l'addon
est installé ou chargé.

## 8. Développement et test

Pour tester votre addon localement sans le publier :

1.  **Localisez votre répertoire de configuration :**
    Rayforge utilise `platformdirs`.

    - **Windows :** `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`
    - **macOS :** `~/Library/Application Support/rayforge/addons`
    - **Linux :** `~/.config/rayforge/addons`
      _(Vérifiez les journaux au démarrage pour `Config dir is ...`)_

2.  **Créez un lien symbolique vers votre addon :**
    Au lieu de copier des fichiers aller-retour, créez un lien symbolique depuis votre dossier
    de développement vers le dossier des addons Rayforge.

    _Linux/macOS :_

    ```bash
    ln -s /chemin/vers/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
    ```

3.  **Redémarrez Rayforge :**
    L'application scanne le répertoire au démarrage. Vérifiez les journaux de la console pour :
    > `Loaded addon: my_custom_addon`

## 9. Publication

Pour partager votre addon avec la communauté :

1.  **Hébergez sur Git :** Poussez votre code vers un dépôt Git public (GitHub, GitLab,
    etc.).
2.  **Soumettez au registre :**
    - Allez sur [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Forkez le dépôt.
    - Ajoutez l'URL Git et les métadonnées de votre addon à la liste du registre.
    - Soumettez une Pull Request.

Une fois accepté, les utilisateurs peuvent installer votre addon directement via l'interface Rayforge ou en
utilisant l'URL Git.
