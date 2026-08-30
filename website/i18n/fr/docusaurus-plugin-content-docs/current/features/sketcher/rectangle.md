---
description: "Dessinez des rectangles et des rectangles arrondis dans l'esquisseur Rayforge, avec points centraux, touches modificatrices et saisie de dimensions."
---

# Rectangle et rectangle arrondi

L'esquisseur propose deux outils de rectangle qui partagent les mêmes
gestes et touches modificatrices : l'outil **rectangle** (`G+R`) et
l'outil **rectangle arrondi** (`G+O`).

![Un rectangle et un rectangle arrondi](/screenshots/addons-sketcher-tool-rectangle.webp)

## Dessiner des rectangles

Dessinez un rectangle en spécifiant deux coins opposés, ou appuyez au
premier coin, glissez et relâchez au coin opposé. Les touches
modificatrices fonctionnent de la même manière pour les deux outils :

- Maintenez `Shift` pour placer le rectangle symétriquement autour du
  point de départ.
- Maintenez `Ctrl` pour le contraindre à un carré.

Chaque rectangle crée automatiquement un **point central** contraint au
centre géométrique, pour que vous puissiez coter ou vous accrocher au
milieu de la forme.

Pendant qu'un aperçu est actif, vous pouvez saisir la taille exacte : la
barre d'état affiche les champs `W` et `H` (plus `R` pour le rayon des
coins des rectangles arrondis). Saisissez une valeur, appuyez sur `Tab`
pour passer d'un champ à l'autre et sur `Entrée` pour l'appliquer. Les
deux outils acceptent indifféremment le geste en deux clics et le geste
clic-glisser ; `Esc` annule l'aperçu.

Le rayon des coins du rectangle arrondi peut aussi être modifié plus tard
en éditant ses contraintes — les coins sont entièrement contraints, le
rayon reste donc ajustable.
