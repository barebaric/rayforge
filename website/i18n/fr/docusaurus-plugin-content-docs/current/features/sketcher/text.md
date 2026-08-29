---
description: "Modèles de zones de texte et fonctions de modèle personnalisées dans l'esquisseur paramétrique 2D de Rayforge."
---

# Modèles de texte

Les zones de texte prennent en charge les expressions de modèle entre accolades.
Celles-ci sont résolues lors de la résolution en utilisant les valeurs de
paramètres actuelles, le texte se met donc à jour automatiquement lorsque tu
modifies une dimension ou une variable d'entrée.

## Substitution de variables

Référence n'importe quel paramètre d'esquisse ou variable d'entrée par nom :

- `{width}` — la valeur actuelle du paramètre « width »
- `{name}` — la valeur d'un paramètre d'entrée de type chaîne
- `{count:.0f}` — formaté avec un spécificateur de format Python (sans décimales)

## Expressions mathématiques

Tu peux utiliser des fonctions mathématiques dans les modèles :

- `{sqrt(area):.2f}` — racine carrée de « area », formatée à 2 décimales
- `{width * 2}` — expressions arithmétiques

Les fonctions mathématiques standard (`sqrt`, `sin`, `cos`, `tan`, `pi`, etc.)
sont disponibles.

## Fonctions intégrées

| Fonction        | Type retour | Description                                           |
| --------------- | ----------- | ----------------------------------------------------- |
| `{today()}`     | `date`      | Date UTC actuelle (ex : `2026-08-26`)                 |
| `{date()}`      | `date`      | Alias de `today()`                                    |
| `{now()}`       | `datetime`  | Date et heure UTC actuelles                           |
| `{time()}`      | `time`      | Heure UTC actuelle (ex : `15:30:00.123456+00:00`)     |
| `{timestamp()}` | `float`     | Horodatage Unix (secondes depuis l'époque)            |
| `{uuid4()}`     | `str`       | Chaîne hexadécimale de 8 caractères (ex : `a1b2c3d4`) |
| `{uuid8()}`     | `str`       | Alias de `uuid4()`                                    |
| `{uuid()}`      | `str`       | Chaîne UUID v4 complète (36 caractères)               |

## Spécifications de format

Les spécifications de format Python fonctionnent avec
tout résultat d'expression :

- `{width:.1f}` — un décimal
- `{timestamp():.0f}` — sans décimal sur l'horodatage
- `{today()}` — représentation chaîne par défaut

## Exemples d'utilisation

- `Pièce #{uuid4()}` — numéro de série unique à chaque résolution
- `L={width:.1f} H={height:.1f}` — étiquettes de dimensions dynamiques
- `Date : {today()}` — dater chaque pièce
- `{name} - {count:.0f}pcs` — combiner paramètres chaîne et numérique
- `{timestamp():.0f}` — horodatage Unix pour journalisation production

## Fonctions de modèle personnalisées

Tu peux enregistrer tes propres fonctions pour les utiliser
dans les modèles de texte. C'est utile pour récupérer des
numéros de série depuis une base de données, lire des
données externes ou générer des étiquettes personnalisées.

### Écrire le script d'enregistrement

Crée un fichier Python (par ex.
`~/.config/rayforge/mes_fonctions.py`) :

```python
"""Enregistrer des fonctions pour modèles de texte."""
import sqlite3

from sketcher.core.template_functions import (
    register_template_function,
)

CHEMIN_DB = "/home/toi/production.db"


def prochain_serial() -> str:
    """Récupérer le prochain numéro de série."""
    conn = sqlite3.connect(CHEMIN_DB)
    try:
        cur = conn.execute(
            "UPDATE compteurs SET valeur = valeur + 1 "
            "WHERE nom = 'serial' RETURNING valeur"
        )
        row = cur.fetchone()
        conn.commit()
        return f"SN-{row[0]:06d}"
    finally:
        conn.close()


register_template_function("prochain_serial", prochain_serial)
```

Points clés :

- Appelle `register_template_function(nom, callable)` pour
  chaque fonction.
- Ta fonction peut faire tout ce que Python peut : ouvrir des
  fichiers, se connecter à des bases de données, appeler des
  APIs, etc.
- La fonction est appelée à **chaque rendu**, elle doit donc
  être rapide.
- Les fonctions sont thread-safe si ton callable l'est.

### Exécuter Rayforge avec le script

Utilise l'option `--script` pour charger tes fonctions
avant l'ouverture de la fenêtre :

```bash
rayforge --script ~/.config/rayforge/mes_fonctions.py \
    mon_document.ryp
```

Cela exécute ton script tôt au démarrage — avant le
chargement des modules et avant la création de la fenêtre
principale — pour que la fonction soit disponible lorsque
l'esquisse est résolue pour la première fois.

### Utiliser la fonction dans une zone de texte

Crée une zone de texte avec :

```
{prochain_serial()}
```

Les spécifications de format fonctionnent aussi :

```
{prochain_serial():>20}
```

### Enregistrer des fonctions programmatiquement

Si tu écris un module ou une bibliothèque réutilisable,
tu peux appeler `register_template_function` depuis n'importe
quel code Python qui s'exécute avant la résolution de
l'esquisse :

```python
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function(
    "numero_piece",
    lambda: f"P-{hash('x') % 10000:04d}"
)
```

### Les fonctions intégrées ne peuvent pas être supprimées

Les fonctions intégrées (`today`, `now`, `uuid`, etc.)
ne peuvent pas être désenregistrées. Si tu dois modifier
leur comportement, enregistre une fonction avec un nom
différent.
