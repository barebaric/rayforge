---
description:
  "Paramètres d'esquisse, expressions de contraintes et modèles de zones de texte dans l'esquisseur
  Rayforge : piloter la géométrie et les étiquettes avec des valeurs nommées et des formules."
---

# Expressions et paramètres

Une esquisse devient véritablement paramétrique lorsque ses dimensions sont pilotées par des valeurs
nommées plutôt que par des nombres codés en dur. L'esquisseur prend cela en charge à deux endroits :
les contraintes dimensionnelles acceptent des **expressions**, et les zones de texte acceptent des
**expressions de modèle**. Les deux sont évaluées par le solveur, si bien que l'esquisse se met à
jour automatiquement dès qu'une valeur change.

## Paramètres d'esquisse

Chaque esquisse possède sa propre liste de paramètres, affichée dans le panneau **Paramètres
d'esquisse** à gauche de l'éditeur d'esquisse. **Ajouter un paramètre** en crée un, avec le choix
entre un entier, un nombre à virgule flottante, un curseur ou une seule ligne de texte. Chaque
paramètre a un nom — la colonne `key` — et c'est ce nom que les expressions référencent.

Une configuration typique pour une boîte d'épaisseur de paroi variable est deux paramètres, `width`
et `thickness`. Rien ne contraint encore la géométrie ; les paramètres ne sont que des noms pour des
nombres tant qu'une expression ne les utilise pas.

## Expressions dans les contraintes

Double-cliquez sur une contrainte dimensionnelle (voir [Contraintes](constraints.md)) et saisissez
une expression au lieu d'un nombre simple :

```
width / 2
```

La valeur de la contrainte devient le résultat de cette expression, réévalué à chaque résolution de
l'esquisse. Modifiez le paramètre `width` et la géométrie contrainte suit — une seule modification
met désormais à jour chaque dimension qui y fait référence. Les contraintes pilotées par une
expression dessinent leur marqueur en orange, et l'étiquette affiche la valeur calculée.

Les expressions peuvent combiner paramètres, opérations arithmétiques et fonctions mathématiques
standard de Python :

```
width - 2 * thickness
sqrt(area) / 2
2 * pi * radius
```

Des fonctions comme `sqrt`, `sin`, `cos` et `tan`, et des constantes comme `pi`, proviennent du
module `math` de Python — ce module, plus les paramètres, est exactement ce qu'une expression de
contrainte peut référencer. Les paramètres de type chaîne peuvent aussi être référencés, ce qui est
surtout utile dans les zones de texte.

## Expressions de modèle dans les zones de texte {#template-expressions-in-text-boxes}

Les zones de texte résolvent les expressions entre accolades au moment de la résolution, si bien que
les étiquettes et le texte gravé affichent des valeurs à jour :

```
W = {width}, H = {height}
```

N'importe quel paramètre peut être substitué par son nom, et le résultat peut être formaté avec un
spécificateur de format Python après deux points :

- `{width}` — la valeur actuelle du paramètre « width »
- `{name}` — la valeur d'un paramètre de type chaîne
- `{width:.1f}` — une décimale
- `{timestamp():.0f}` — sans décimales sur un résultat de fonction

Les mathématiques fonctionnent ici aussi, que ce soit sous forme d'expression comme `{width * 2}` ou
via une fonction comme `{sqrt(area):.2f}`. Par rapport aux expressions de contraintes, les modèles
de texte disposent d'une boîte à outils plus riche : outre le module mathématique, ils exposent les
fonctions intégrées ci-dessous, et des fonctions personnalisées peuvent leur être enregistrées (voir
[plus bas](#custom-template-functions)).

### Fonctions de modèle intégrées

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

Les utilisations typiques comprennent les numéros de série uniques à chaque résolution
(`Pièce #{uuid4()}`), les étiquettes de dimensions dynamiques (`W={width:.1f} H={height:.1f}`), les
dates (`Date : {today()}`), les compteurs de production (`{name} - {count:.0f}pcs`) ou les
horodatages Unix pour la journalisation de production (`{timestamp():.0f}`).

## Fonctions de modèle personnalisées {#custom-template-functions}

Vous pouvez enregistrer vos propres fonctions à utiliser dans les modèles de zones de texte. C'est
utile pour récupérer des numéros de série depuis une base de données, lire des données externes ou
générer des étiquettes personnalisées.

### Écrire le script d'enregistrement

Créez un fichier Python (par ex. `~/.config/rayforge/mes_fonctions.py`) :

```python
"""Enregistrer des fonctions pour modèles de texte."""
import sqlite3

from sketcher.core.template_functions import (
    register_template_function,
)

CHEMIN_DB = "/home/toi/production.db"


def prochain_serial() -> str:
    """Récupérer et réserver le prochain numéro de série."""
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

Appelez `register_template_function(nom, callable)` pour chaque fonction. La fonction peut faire
tout ce que Python permet — ouvrir des fichiers, se connecter à des bases de données, appeler des
APIs — et elle est appelée à **chaque rendu**, elle doit donc être rapide (utilisez un cache si les
données sous-jacentes ne changent pas entre les rendus). Les fonctions sont thread-safe si votre
callable l'est.

### Exécuter Rayforge avec le script

Utilisez l'option `--script` pour charger vos fonctions avant l'ouverture de la fenêtre :

```bash
rayforge --script ~/.config/rayforge/mes_fonctions.py \
    mon_document.ryp
```

Cela exécute votre script tôt au démarrage — avant le chargement des extensions et avant la création
de la fenêtre principale — pour que la fonction soit disponible lorsque l'esquisse est résolue pour
la première fois.

### Utiliser la fonction dans une zone de texte

Dans l'esquisseur, créez une zone de texte avec :

```
{prochain_serial()}
```

Les spécifications de format fonctionnent aussi :

```
{prochain_serial():>20}
```

### Enregistrer des fonctions programmatiquement

Si vous écrivez une extension ou une bibliothèque réutilisable, vous pouvez appeler
`register_template_function` depuis n'importe quel code Python qui s'exécute avant la résolution de
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

Les fonctions intégrées (`today`, `now`, `uuid`, etc.) ne peuvent pas être désenregistrées. Si vous
devez modifier leur comportement, enregistrez une fonction avec un nom différent.
