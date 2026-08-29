---
description: "Découvrez les contraintes géométriques et dimensionnelles dans l'esquisseur paramétrique 2D de Rayforge."
---

# Système de contraintes paramétriques

Le système de contraintes est le cœur de l'esquisseur paramétrique, vous
permettant de définir des relations géométriques précises :

## Contraintes géométriques

- **Coïncidence** : Force deux points à occuper le même emplacement
- **Verticale** : Contraint une ligne à être parfaitement verticale
- **Horizontale** : Contraint une ligne à être parfaitement horizontale
- **Tangente** : Rend une ligne tangente à un cercle ou un arc
- **Perpendiculaire** : Force deux lignes, une ligne et un arc/cercle, ou deux
  arcs/cercles à se rencontrer à 90 degrés
- **Point sur ligne/forme** : Contraint un point à se trouver sur une ligne, un
  arc ou un cercle
- **Collinéaire** : Force deux lignes ou plus à se trouver sur la même ligne
  infinie
- **Symétrie** : Crée des relations symétriques entre les éléments. Prend en
  charge deux modes :
  - **Symétrie de point** : Sélectionnez 3 points (le premier est le centre)
  - **Symétrie de ligne** : Sélectionnez 2 points et 1 ligne (la ligne est
    l'axe)

## Contraintes dimensionnelles

- **Distance** : Définit la distance exacte entre deux points ou le long d'une
  ligne
- **Diamètre** : Définit le diamètre d'un cercle
- **Rayon** : Définit le rayon d'un cercle ou d'un arc
- **Angle** : Impose un angle spécifique entre deux lignes
- **Rapport d'aspect** : Force le rapport entre deux distances à être égal à une
  valeur spécifiée
- **Longueur/Rayon égal** : Force plusieurs éléments (lignes, arcs, ellipses ou
  cercles) à avoir la même longueur ou le même rayon
- **Distance égale** : Rend deux segments de ligne de même longueur (différent
  de Longueur/Rayon égal, qui peut aussi s'appliquer aux arcs et cercles)
