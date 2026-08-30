---
description: "Référence de la ligne de commande pour Rayforge."
---

# Ligne de Commande

Référence complète des options en ligne de commande.

```
rayforge [options] [fichiers...]
```

---

## Arguments positionnels

| Argument   | Description                         |
| ---------- | ----------------------------------- |
| `fichiers` | Fichiers SVG ou image au lancement. |

---

## Options

| Option              | Description                                |
| ------------------- | ------------------------------------------ |
| `--version`         | Afficher la version et quitter.            |
| `-h`, `--help`      | Afficher l'aide et quitter.                |
| `--loglevel NIVEAU` | Niveau de journalisation. Défaut : `INFO`. |
| `--config RÉP`      | Répertoire de configuration personnalisé.  |
| `--exit`            | Quitter après l'import.                    |
| `--vector`          | Forcer l'import en vecteurs directs.       |
| `--trace`           | Forcer l'import par traçage bitmap.        |
| `--script SCRIPT`   | Script de démarrage précoce.               |
| `--uiscript SCRIPT` | Script UI (post-chargement).               |

---

## Exemples

### Ouvrir un fichier

```bash
rayforge monprojet.ryp
```

### Ouvrir plusieurs fichiers

```bash
rayforge piece1.svg logo.png conception.ryp
```

### Importer avec traçage

```bash
rayforge --trace photo.png
```

### Exécuter un script précoce et quitter

```bash
rayforge --exit --script enregistrer.py \
    monprojet.ryp
```

### Script UI (automatisation)

```bash
rayforge --exit --uiscript screenshot.py \
    monprojet.ryp
```

### Traitement par lots

```bash
rayforge --exit --vector entree.svg
```

---

## Scripts précoces (`--script`)

L'option `--script` exécute un script Python **de manière
synchrone au démarrage**, avant le chargement des modules
et avant la création de la fenêtre principale. Utile pour :

- Enregistrer des plugins avec le gestionnaire `pluggy`
- Configurer le contexte de l'application
- Enregistrer des fonctions de modèle pour les zones de texte
- Définir des variables d'environnement avant le démarrage

Le script a accès au contexte via `get_context()` :

```python
from rayforge.context import get_context

ctx = get_context()
```

### Exemple : Enregistrer une fonction de modèle

```python
"""Enregistrer une fonction pour les modèles de texte.

Exécuter avec : rayforge --script enregistrer_fn.py
"""
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function("mon_id", lambda: "PIECE-001")
```

Maintenant `{mon_id()}` fonctionne dans toute zone de texte.

Voir
[Fonctions de modèle personnalisées](../features/sketcher/expressions.md#custom-template-functions)
dans la documentation du sketcher pour un tutoriel complet.

---

## Scripts UI (`--uiscript`)

L'option `--uiscript` exécute un script Python **après le
chargement complet de la fenêtre principale**, dans un thread
en arrière-plan. Utile pour :

- Tests UI automatisés
- Captures d'écran de l'application
- Flux de travail de bout en bout

Le script peut importer l'application et la fenêtre
directement :

```python
from rayforge.uiscript import app, win
```

Le script s'exécute dans un **thread en arrière-plan** —
sois attentif à la sécurité des threads lors de l'accès
aux widgets GTK
(utilise `GLib.idle_add` pour les opérations GTK).

### Exemple : Capturer une capture d'écran

```python
"""Capturer la fenêtre principale."""
from rayforge.uiscript import app, win

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib

def capture():
    surface = win.get_surface()
    if surface:
        surface.write_to_png("/tmp/rayforge_screenshot.png")
    return GLib.SOURCE_REMOVE

GLib.idle_add(capture)
```

---

## Utiliser les deux options

`--script` et `--uiscript` peuvent être utilisés ensemble.
Le `--script` s'exécute en premier (de manière synchrone),
puis la fenêtre se charge, et ensuite `--uiscript`
s'exécute :

```bash
rayforge --script setup_precoce.py \
    --uiscript automatisation.py \
    monprojet.ryp
```

C'est utile quand tu dois enregistrer des plugins en premier
et ensuite contrôler l'interface plus tard.
