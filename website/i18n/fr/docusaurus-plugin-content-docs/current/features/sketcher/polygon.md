---
description:
  "Dessinez des polygones réguliers et des étoiles dans l'esquisseur Rayforge. Ils sont faits de
  points et de lignes ordinaires, maintenus réguliers par des contraintes."
---

# Polygone et étoile

L'outil **polygone** (`G+X`) dessine un polygone régulier, par exemple un hexagone pour un
sous-verre, ou une étoile.

## Dessiner un polygone

Cliquez à l'endroit du centre, déplacez le pointeur et cliquez à nouveau, ou appuyez au centre,
faites glisser et relâchez. Le pointeur fixe le premier sommet ; il détermine donc à la fois la
taille et la rotation. Un nouveau polygone a 6 sommets.

- Maintenez `Ctrl` pour dessiner une étoile à la place : un sommet intérieur est ajouté entre chaque
  paire de sommets extérieurs, à la moitié du rayon extérieur.

Pendant que l'aperçu est actif, vous pouvez saisir des valeurs exactes : la barre d'état affiche les
champs `R`, `N` et `Star`. `R` est le rayon du centre aux sommets, `N` le nombre de sommets (de 3
à 100) et `Star` le rayon intérieur en fraction du rayon extérieur (entre 0 et 1). Saisir une valeur
`Star` transforme le polygone en étoile. Tapez une valeur, appuyez sur `Tab` pour passer au champ
suivant et sur `Enter` pour créer la forme ; `Esc` annule l'aperçu. L'outil retient le nombre de
sommets pour le polygone suivant.

## Ce que vous obtenez

Un polygone n'est pas un objet spécial : il se compose de points et de lignes ordinaires, de sorte
que le congé, le chanfrein, le décalage et les opérations booléennes s'y appliquent comme à
n'importe quelle autre forme.

Des contraintes le maintiennent régulier. Tous les côtés ont la même longueur et les sommets se
trouvent sur un cercle de construction autour du point central ; une étoile possède un second cercle
de construction pour ses sommets intérieurs et conserve le rapport entre les deux rayons. Si vous
avez saisi un rayon, une contrainte de rayon le fixe. Vous pouvez ensuite faire glisser le centre
pour déplacer le polygone, ou faire glisser un sommet pour le redimensionner et le faire pivoter :
il reste régulier. Pour modifier la taille plus tard, modifiez la contrainte de rayon ou ajoutez-en
une au cercle de construction.
