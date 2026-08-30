---
description: "Outils de l'esquisseur, raccourcis clavier, menu radial, mode construction, grille, accrochage, décalage, chanfrein et congé dans Rayforge."
---

# Outils de l'esquisseur

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
- `G+G` : Outil de grille (créer une grille de copies à partir de la
  sélection)
- `G+N` : Basculer le mode construction sur la sélection

### Raccourcis d'actions

- `O+F` : Décaler le contour sélectionné
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

## Commandes de visibilité

La grille s'adapte au niveau de zoom et sert toujours de référence de
dimensionnement ; le fonctionnement de l'accrochage est décrit dans
[l'aperçu de l'esquisseur](index.md#grid-and-snapping).

La barre d'outils de l'esquisseur inclut des boutons de bascule pour
contrôler la visibilité :

- **Afficher/masquer la géométrie de construction** : Bascule la visibilité
  des entités de construction
- **Afficher/masquer les contraintes** : Bascule la visibilité des
  marqueurs de contraintes

Ces commandes aident à réduire l'encombrement visuel lors du travail sur
des esquisses complexes.

### Auto-contrainte pendant la création

De nombreux outils de dessin appliquent automatiquement des contraintes
lors de la création de géométrie. L'outil de tracé crée des contraintes
horizontales et verticales lorsque les guides d'accrochage montrent un
alignement pendant le dessin, ce qui aide à garder votre esquisse ordonnée
dès le départ, plutôt que de corriger les choses par la suite.

### Déplacement contraint aux axes

Lors du glissement de points ou de géométrie, maintenez `Shift` pour
contraindre le déplacement à l'axe le plus proche (horizontal ou vertical).
Ceci est utile pour maintenir l'alignement lors des ajustements.

## Décalage de contour

L'outil de décalage agrandit ou réduit le contour sélectionné d'une distance
donnée, ou transforme un tracé ouvert en une lumière (slot). Sélectionnez les
entités formant un contour (ou utilisez le double-clic pour sélectionner la
géométrie connectée), puis appuyez sur `O+F` ou utilisez l'entrée
**Décalage** du menu radial.

![Boîte de dialogue de décalage de contour](/screenshots/addons-sketcher-offset-dialog.webp)

La boîte de dialogue demande la distance de décalage et affiche un aperçu en
direct du résultat sur le canevas pendant la saisie :

- Les **contours fermés** s'agrandissent avec une distance positive et se
  réduisent avec une distance négative. Un décalage qui ferait s'effondrer le
  contour est refusé.
- Les **tracés ouverts** deviennent un contour fermé en forme de lumière de la
  largeur indiquée, avec des extrémités arrondies.

![Contour de Bézier](/screenshots/addons-sketcher-offset-before.webp)
![Bézier décalé en une lumière](/screenshots/addons-sketcher-offset-after.webp)

Le décalage remplace le contour sélectionné par le résultat :

- Les cercles, arcs et ellipses isolés conservent leur type d'entité et sont
  mis à jour sur place : ils restent modifiables et contraints comme avant.
- Les chaînes de segments connectés (y compris les Béziers) sont remplacées
  par une entité polygone. Le polygone s'édite comme un tout : faites glisser
  son point central pour le déplacer et son point de poignée pour le faire
  pivoter ou le redimensionner uniformément.

Si la sélection contient plusieurs contours déconnectés, chacun est décalé
indépendamment en une seule étape.

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
