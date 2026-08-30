---
description: "Outils de l'esquisseur, raccourcis clavier, menu radial, mode construction, grille, accrochage, chanfrein et congé dans Rayforge."
---

# Outils de l'esquisseur

## Interface du menu radial

L'esquisseur dispose d'un menu radial contextuel qui fournit un accès rapide à
tous les outils de dessin et de contrainte. Ce menu circulaire apparaît lorsque
vous faites un clic droit dans l'espace de travail de l'esquisse et s'adapte
selon votre contexte et votre sélection actuels.

Les éléments du menu radial affichent dynamiquement les options disponibles en
fonction de votre sélection. Par exemple, en cliquant sur un espace vide, vous
verrez les outils de dessin. En cliquant sur de la géométrie sélectionnée, vous
verrez les contraintes applicables.

![Menu radial de l'esquisseur](/screenshots/sketcher-pie-menu.webp)

## Raccourcis clavier

L'esquisseur fournit des raccourcis clavier pour un flux de travail efficace :

### Raccourcis d'outils

- `Space` : Outil de sélection
- `G+P` : Outil de tracé (lignes et courbes de Bézier)
- `G+A` : Outil d'arc
- `G+C` : Outil d'ellipse
- `G+R` : Outil de rectangle
- `G+O` : Outil de rectangle arrondi
- `G+F` : Outil de remplissage de zone
- `G+T` : Outil de zone de texte
- `G+Y` : Outil de tableau circulaire
- `G+W` : Outil de tableau le long d'une courbe
- `G+G` : Outil de grille (basculer la visibilité de la grille)
- `G+N` : Basculer le mode construction sur la sélection

### Raccourcis d'actions

- `C+H` : Ajouter un chanfrein
- `C+F` : Ajouter un congé
- `C+S` : Aplatir les courbes de Bézier sélectionnées en lignes
- `M+V` : Symétrie verticale de la sélection
- `M+H` : Symétrie horizontale de la sélection
- `Ctrl+D` : Dupliquer la sélection sur place

### Raccourcis de contraintes

- `H` : Appliquer la contrainte Horizontale
- `V` : Appliquer la contrainte Verticale
- `N` : Appliquer la contrainte Perpendiculaire
- `T` : Appliquer la contrainte Tangente
- `E` : Appliquer la contrainte Égal
- `O` ou `C` : Appliquer la contrainte d'Alignement (Coïncidence)
- `S` : Appliquer la contrainte de Symétrie
- `K+D` : Appliquer la contrainte de Distance
- `K+R` : Appliquer la contrainte de Rayon
- `K+O` : Appliquer la contrainte de Diamètre
- `K+A` : Appliquer la contrainte d'Angle
- `K+X` : Appliquer la contrainte de Rapport d'aspect

### Raccourcis généraux

- `Ctrl+Z` : Annuler
- `Ctrl+Y` ou `Ctrl+Shift+Z` : Rétablir
- `Ctrl+D` : Dupliquer les éléments sélectionnés
- `Delete` : Supprimer les éléments sélectionnés
- `Touches fléchées` : Déplacer les entités sélectionnées (maintenez `Shift` pour un pas plus large)
- `Escape` : Annuler l'opération en cours ou désélectionner
- `F` : Ajuster la vue au contenu

## Miroir, duplication et décalage

Plusieurs outils de transformation agissent sur la sélection actuelle :

- **Symétrie verticale / horizontale** (`M+V` / `M+H`) : retourne la
  sélection sur place par rapport au centre de sa boîte d'englobage. Les
  contraintes traversant la limite de sélection sont supprimées ; les
  contraintes internes sont préservées.
- **Dupliquer** (`Ctrl+D`) : copie la sélection sur place. Les copies
  reçoivent de nouveaux identifiants et des contraintes internes
  remappées ; seules les copies restent sélectionnées ensuite. L'annulation
  les supprime.
- **Décalage** : avec des entités sélectionnées, les **touches fléchées**
  déplacent la sélection. Maintenez `Shift` pour un pas de décalage plus
  large.

Ces outils sont accessibles depuis la barre d'outils et le menu **Esquisse**.

## Mode construction

Le mode construction vous permet de marquer des entités comme « géométrie de
construction », des éléments auxiliaires utilisés pour guider votre conception
mais qui ne font pas partie du résultat final. Les entités de construction sont
affichées différemment (généralement sous forme de lignes tiretées) et ne sont
pas incluses lorsque l'esquisse est utilisée pour la découpe ou la gravure
laser.

Pour basculer le mode construction :

- Sélectionnez une ou plusieurs entités
- Appuyez sur `N` ou `G+N`, ou utilisez l'option Construction dans le menu
  radial

Les entités de construction sont utiles pour :

- Créer des lignes et des cercles de référence
- Définir une géométrie temporaire pour l'alignement
- Construire des formes complexes à partir d'un cadre de guides

## Grille, accrochage et commandes de visibilité

### Outil de grille

L'outil de grille fournit un repère visuel pour l'alignement et le
dimensionnement :

- Activez/désactivez la grille avec le bouton de l'outil ou `G+G`
- La grille s'adapte à votre niveau de zoom pour un espacement cohérent

### Accrochage magnétique

Lors de la création ou du déplacement de géométrie, Rayforge attire
automatiquement votre curseur vers les éléments proches : extrémités, milieux de
lignes, intersections et autres points de référence. Cela facilite la connexion
précise des formes sans avoir à placer manuellement chaque point. L'indicateur
d'accrochage se met en surbrillance lorsque votre curseur est proche d'une
cible d'accrochage.

### Auto-contrainte pendant la création

De nombreux outils de dessin appliquent automatiquement des contraintes lors de
la création de géométrie. Par exemple, lors du tracé d'une ligne proche de
l'horizontale ou de la verticale, l'esquisseur proposera de la verrouiller en
place. L'outil de tracé crée également des contraintes horizontales et
verticales automatiquement lorsque les guides d'accrochage montrent un
alignement pendant le dessin. Cela aide à garder votre esquisse ordonnée dès le départ, plutôt que de
corriger les choses par la suite.

### Commandes afficher/masquer

La barre d'outils de l'esquisseur inclut des boutons de bascule pour contrôler
la visibilité :

- **Afficher/masquer la géométrie de construction** : Bascule la visibilité des
  entités de construction
- **Afficher/masquer les contraintes** : Bascule la visibilité des marqueurs de
  contraintes

Ces commandes aident à réduire l'encombrement visuel lors du travail sur des
esquisses complexes.

### Déplacement contraint aux axes

Lors du glissement de points ou de géométrie, maintenez `Shift` pour contraindre
le déplacement à l'axe le plus proche (horizontal ou vertical). Ceci est utile
pour maintenir l'alignement lors des ajustements.

## Chanfrein et congé

L'esquisseur fournit des outils pour modifier les coins de votre géométrie :

- **Chanfrein** : Remplace un coin anguleux par un bord biseauté. Sélectionnez
  un point de jonction (où deux lignes se rencontrent) et appliquez l'action de
  chanfrein.
- **Congé** : Remplace un coin anguleux par un bord arrondi. Sélectionnez un
  point de jonction (où deux lignes se rencontrent) et appliquez l'action de
  congé.

Pour utiliser le chanfrein ou le congé :

1. Sélectionnez un point de jonction où deux lignes se rencontrent
2. Appuyez sur `C+H` pour le chanfrein ou `C+F` pour le congé
3. Utilisez le menu radial ou les raccourcis clavier pour appliquer la
   modification
