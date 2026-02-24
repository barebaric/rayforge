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

Voir le [Guide d'installation](../../getting-started/installation#windows-developer) pour les instructions détaillées de configuration du développement MSYS2.

### Démarrage rapide

Les tâches de développement sur Windows sont gérées via le script `run.bat`, qui est un wrapper pour le shell MSYS2.

Après avoir cloné le dépôt et terminé la configuration MSYS2, vous pouvez utiliser ces commandes depuis une invite de commande Windows standard ou PowerShell :

```batch
.\run.bat setup
```

Cela exécute `scripts/win/win_setup.sh` pour installer tous les paquets système et Python nécessaires dans votre environnement MSYS2/MinGW64.

### Hooks pre-commit (Optionnel)

Pour formater et analyser automatiquement votre code avant chaque commit, exécutez ceci depuis le shell MSYS2 MINGW64 :

```bash
bash scripts/win/win_setup_dev.sh
```

:::note

Les hooks pre-commit nécessitent d'exécuter les commandes git dans le shell MSYS2 MINGW64, et non depuis PowerShell ou l'invite de commande.

:::

### Commandes utiles

Toutes les commandes sont exécutées via le script `run.bat` :

-   `run app` : Exécuter l'application depuis les sources.
    -   Ajoutez `--loglevel=DEBUG` pour une sortie plus verbeuse.
-   `run test` : Exécuter la suite de tests complète avec `pytest`.
-   `run lint` : Exécuter tous les linters (`flake8`, `pyflakes`, `pyright`).
-   `run format` : Formater et corriger automatiquement le code avec `ruff`.
-   `run build` : Construire l'exécutable Windows final (`.exe`).

Alternativement, vous pouvez exécuter les scripts directement depuis le shell MSYS2 MINGW64 :

-   `bash scripts/win/win_run.sh` : Exécuter l'application.
-   `bash scripts/win/win_test.sh` : Exécuter la suite de tests.
-   `bash scripts/win/win_lint.sh` : Exécuter tous les linters.
-   `bash scripts/win/win_format.sh` : Formater et corriger automatiquement le code.
-   `bash scripts/win/win_build.sh` : Construire l'exécutable Windows.
