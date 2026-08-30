---
description:
  "Biseautez les coins vifs avec l'outil chanfrein ou arrondissez-les avec l'outil congé dans
  l'esquisseur Rayforge."
---

# Chanfrein et congé

L'esquisseur fournit deux outils pour modifier les coins où deux lignes se rencontrent :

- **Chanfrein** (`C+H`) : remplace un coin vif par un bord biseauté.
- **Congé** (`C+F`) : remplace un coin vif par un bord arrondi.

![Un rectangle chanfreiné à côté d'un rectangle muni d'un congé](/screenshots/addons-sketcher-tool-chamfer-fillet.webp)

Pour appliquer l'un d'eux :

1. Sélectionnez un point de jonction où exactement deux lignes se rencontrent.
2. Appuyez sur `C+H` pour le chanfrein ou `C+F` pour le congé, ou choisissez l'outil dans le menu
   radial.

Le coin est remplacé en une seule étape. Les deux lignes sont raccourcies et le nouveau bord est
inséré entre elles, avec des contraintes qui gardent les segments raccourcis collinéaires avec les
originaux et le coin symétrique. Sur un chanfrein, la longueur du biseau vaut par défaut une
fraction de la ligne adjacente la plus courte ; sur un congé, le rayon de l'arc est choisi pour
s'ajuster. Glisser ensuite les extrémités du bord inséré en ajuste la taille, les contraintes
gardant le coin intact.
