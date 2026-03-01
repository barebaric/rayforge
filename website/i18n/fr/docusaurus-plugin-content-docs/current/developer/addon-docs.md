# Guide du développeur d'addons Rayforge

Rayforge utilise un système d'addons basé sur [pluggy](https://pluggy.readthedocs.io/)
pour permettre aux développeurs d'étendre les fonctionnalités, d'ajouter de nouveaux pilotes machine, ou
d'intégrer une logique personnalisée sans modifier le code source principal.

## 1. Démarrage rapide

La façon la plus rapide de commencer est d'utiliser le modèle officiel.

1. **Forkez ou clonez** le [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template).
2. **Renommez** le répertoire et mettez à jour les métadonnées.

## 2. Structure de l'addon

Le `AddonManager` scanne le répertoire `addons`. Un addon valide doit être un
répertoire contenant au moins deux fichiers :

1. `rayforge_addon.yaml` (Métadonnées)
2. Un point d'entrée Python (par exemple, `addon.py`)

**Disposition du répertoire :**

```text
my-rayforge-addon/
├── rayforge_addon.yaml  <-- Manifeste requis
├── addon.py             <-- Point d'entrée (logique)
├── assets/              <-- Ressources optionnelles
└── README.md
```

## 3. Le manifeste (`rayforge_addon.yaml`)

Ce fichier indique à Rayforge comment charger votre addon.

```yaml
# rayforge_addon.yaml

# Identifiant unique pour votre addon
name: my_custom_addon

# Nom d'affichage lisible par l'homme
display_name: "Mon Addon Personnalisé"

# Description affichée dans l'interface
description: "Ajoute le support pour le découpeur laser XYZ."

# Dépendances (addon et contraintes de version)
depends:
  - rayforge>=0.27.0,~0.27

# Le fichier python à charger (relatif au dossier de l'addon)
entry_point: addon.py

# Métadonnées de l'auteur
author: Jean Dupont
url: https://github.com/username/my-custom-addon
```

## 4. Écrire le code de l'addon

Rayforge utilise des hooks `pluggy`. Pour vous intégrer à Rayforge, définissez des fonctions décorées
avec `@pluggy.HookimplMarker("rayforge")`.

### Code de base (`addon.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

# Définir le marqueur d'implémentation de hook
hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Appelé lorsque Rayforge est entièrement initialisé.
    C'est votre point d'entrée principal pour accéder aux gestionnaires.
    """
    logger.info("Mon Addon Personnalisé a démarré !")

    # Accéder aux systèmes principaux via le contexte
    machine = context.machine
    camera = context.camera_mgr

    if machine:
        logger.info(f"Addon exécuté sur la machine : {machine.id}")

@hookimpl
def register_machines(machine_manager):
    """
    Appelé au démarrage pour enregistrer de nouveaux pilotes machine.
    """
    # from .my_driver import MyNewMachine
    # machine_manager.register("my_new_machine", MyNewMachine)
    pass
```

### Hooks disponibles

Définis dans `rayforge/core/hooks.py` :

**`rayforge_init`** (`context`)
: **Point d'entrée principal.** Appelé après le chargement de la configuration, de la caméra et du matériel.
  Utilisez-le pour la logique, les injections UI ou les écouteurs.

**`register_machines`** (`machine_manager`)
: Appelé tôt dans le processus de démarrage. Utilisez-le pour enregistrer de nouvelles classes/pilotes
  matériels.

## 5. Accéder aux données Rayforge

Le hook `rayforge_init` fournit le **`RayforgeContext`**. Via cet objet,
vous pouvez accéder à :

- **`context.machine`** : L'instance machine actuellement active.
- **`context.config`** : Paramètres de configuration globaux.
- **`context.camera_mgr`** : Accès aux flux caméra et outils de vision par ordinateur.
- **`context.material_mgr`** : Accès à la bibliothèque de matériaux.
- **`context.recipe_mgr`** : Accès aux recettes de traitement.

## 6. Développement et test

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

## 7. Publication

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
