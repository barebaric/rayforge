# Configuration

Ce guide couvre la configuration de votre environnement de développement pour Rayforge.

## Linux

### Prérequis

Voir le [Guide d'installation](../../getting-started/installation#linux-pixi) pour les instructions d'installation de Pixi.

### Hooks pre-commit (Optionnel)

Pour formater et analyser automatiquement votre code avant chaque commit, vous pouvez installer les hooks pre-commit :

```bash
pixi run pre-commit-install
```

### Commandes utiles

Toutes les commandes sont exécutées via `pixi run` :

-   `pixi run rayforge` : Exécuter l'application.
    -   Ajoutez `--loglevel=DEBUG` pour une sortie plus verbeuse.
-   `pixi run test` : Exécuter la suite de tests complète avec `pytest`.
-   `pixi run format` : Formater tout le code avec `ruff`.
-   `pixi run lint` : Exécuter tous les linters (`flake8`, `pyflakes`, `pyright`).

## Windows

### Prérequis

-   [MSYS2](https://www.msys2.org/) (fournit l'environnement MinGW64).
-   [Git pour Windows](https://git-scm.com/download/win).

### Installation

Les tâches de développement sur Windows sont gérées via le script `run.bat`, qui est un wrapper pour le shell MSYS2.

Après avoir cloné le dépôt, exécutez la commande de configuration depuis une invite de commande Windows standard ou PowerShell :

```batch
.\run.bat setup
```

Cela exécute `scripts/win/win_setup.sh` pour installer tous les paquets système et Python nécessaires dans votre environnement MSYS2/MinGW64.

### Commandes utiles

Toutes les commandes sont exécutées via le script `run.bat` :

-   `run app` : Exécuter l'application depuis les sources.
    -   Ajoutez `--loglevel=DEBUG` pour une sortie plus verbeuse.
-   `run test` : Exécuter la suite de tests complète avec `pytest`.
-   `run build` : Construire l'exécutable Windows final (`.exe`).
