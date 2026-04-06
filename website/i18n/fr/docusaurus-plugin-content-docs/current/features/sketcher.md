# Esquisseur paramétrique 2D

L'Esquisseur paramétrique 2D est une fonctionnalité puissante de Rayforge qui
vous permet de créer et modifier des conceptions 2D précises basées sur des
contraintes directement dans l'application. Cette fonctionnalité vous permet de
concevoir des pièces personnalisées à partir de zéro sans avoir besoin d'un
logiciel de CAO externe.

## Aperçu

L'esquisseur fournit un ensemble complet d'outils pour créer des formes
géométriques et appliquer des contraintes paramétriques afin de définir des
relations précises entre les éléments. Cette approche garantit que vos
conceptions conservent leur géométrie prévue même lorsque les dimensions sont
modifiées.

## Création et modification d'esquisses

### Créer une nouvelle esquisse

1. Cliquez sur le bouton « Nouvelle esquisse » dans la barre d'outils ou
   utilisez le menu principal
2. Un nouvel espace de travail vide s'ouvrira avec l'interface de l'éditeur
   d'esquisse
3. Commencez à créer de la géométrie avec les outils de dessin du menu radial
   ou les raccourcis clavier
4. Appliquez des contraintes pour définir les relations entre les éléments
5. Cliquez sur « Terminer l'esquisse » pour enregistrer votre travail et
   revenir à l'espace de travail principal

### Modifier une esquisse existante

1. Double-cliquez sur une pièce basée sur une esquisse dans l'espace de travail
   principal
2. Alternativement, sélectionnez une esquisse et choisissez « Modifier
   l'esquisse » dans le menu contextuel
3. Effectuez vos modifications avec les mêmes outils et contraintes
4. Cliquez sur « Terminer l'esquisse » pour enregistrer les modifications ou
   sur « Annuler l'esquisse » pour les annuler

## Création de géométrie 2D

L'esquisseur prend en charge la création des éléments géométriques de base
suivants :

- **Tracés (lignes et courbes de Bézier)** : Dessinez des lignes droites et des
  courbes de Bézier lisses avec l'outil de tracé unifié. Cliquez pour placer
  des points, glissez pour créer des poignées de Bézier.
- **Arcs** : Dessinez des arcs en spécifiant un point central, un point de
  départ et un point d'arrivée
- **Ellipses** : Créez des ellipses (et des cercles) en définissant un point
  central et en glissant pour définir la taille et le rapport d'aspect.
  Maintenez `Ctrl` en glissant pour contraindre à un cercle parfait.
- **Rectangles** : Dessinez des rectangles en spécifiant deux coins opposés
- **Rectangles arrondis** : Dessinez des rectangles avec des coins arrondis
- **Zones de texte** : Ajoutez des éléments textuels à votre esquisse
- **Remplissages** : Remplissez des régions fermées pour créer des zones
  solides

Ces éléments constituent la base de vos conceptions 2D et peuvent être combinés
pour créer des formes complexes. Les remplissages sont particulièrement utiles
pour créer des régions solides qui seront gravées ou découpées en une seule
pièce.

## Travailler avec les courbes de Bézier

L'outil de tracé prend en charge les courbes de Bézier pour créer des formes
lisses et organiques :

### Dessiner des courbes de Bézier

1. Sélectionnez l'outil de tracé dans le menu radial ou utilisez le raccourci
   clavier
2. Cliquez pour placer des points ; chaque clic crée un nouveau point
3. Glissez après avoir cliqué pour créer des poignées de Bézier pour des
   courbes lisses
4. Continuez à ajouter des points pour construire votre tracé
5. Appuyez sur Échap ou double-cliquez pour terminer le tracé

### Modifier les courbes de Bézier

- **Déplacer des points** : Cliquez et glissez n'importe quel point pour le
  repositionner
- **Ajuster les poignées** : Glissez les extrémités des poignées pour modifier
  la forme de la courbe
- **Se connecter aux points existants** : Lors de la modification d'un tracé,
  vous pouvez vous accrocher aux points existants de votre esquisse
- **Rendre lisse/symétrique** : Les points connectés par une contrainte de
  coïncidence peuvent être rendus lisses (tangente continue) ou symétriques
  (poignées en miroir)

### Convertir des courbes en lignes

Utilisez l'**outil d'aplatissement** pour convertir les courbes de Bézier en
lignes droites. Ceci est utile lorsque vous avez besoin d'une géométrie propre
et simple. Sélectionnez les segments de Bézier que vous souhaitez convertir et
appliquez l'action d'aplatissement.

## Système de contraintes paramétriques

Le système de contraintes est le cœur de l'esquisseur paramétrique, vous
permettant de définir des relations géométriques précises :

### Contraintes géométriques

- **Coïncidence** : Force deux points à occuper le même emplacement
- **Verticale** : Contraint une ligne à être parfaitement verticale
- **Horizontale** : Contraint une ligne à être parfaitement horizontale
- **Tangente** : Rend une ligne tangente à un cercle ou un arc
- **Perpendiculaire** : Force deux lignes, une ligne et un arc/cercle, ou deux
  arcs/cercles à se rencontrer à 90 degrés
- **Point sur ligne/forme** : Contraint un point à se trouver sur une ligne, un
  arc ou un cercle
- **Colinéaire** : Force deux lignes ou plus à se trouver sur la même ligne
  infinie
- **Symétrie** : Crée des relations symétriques entre les éléments. Prend en
  charge deux modes :
  - **Symétrie de point** : Sélectionnez 3 points (le premier est le centre)
  - **Symétrie de ligne** : Sélectionnez 2 points et 1 ligne (la ligne est
    l'axe)

### Contraintes dimensionnelles

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

## Interface du menu radial

L'esquisseur dispose d'un menu radial contextuel qui fournit un accès rapide à
tous les outils de dessin et de contrainte. Ce menu circulaire apparaît lorsque
vous faites un clic droit dans l'espace de travail de l'esquisse et s'adapte
selon votre contexte et votre sélection actuels.

Les éléments du menu radial affichent dynamiquement les options disponibles en
fonction de votre sélection. Par exemple, en cliquant sur un espace vide, vous
verrez les outils de dessin. En cliquant sur de la géométrie sélectionnée, vous
verrez les contraintes applicables.

![Menu radial de l'esquisseur](/screenshots/sketcher-pie-menu.png)

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
- `G+G` : Outil de grille (basculer la visibilité de la grille)
- `G+N` : Basculer le mode construction sur la sélection

### Raccourcis d'actions

- `C+H` : Ajouter un chanfrein
- `C+F` : Ajouter un congé
- `C+S` : Aplatir les courbes de Bézier sélectionnées en lignes

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
- `Delete` : Supprimer les éléments sélectionnés
- `Escape` : Annuler l'opération en cours ou désélectionner
- `F` : Ajuster la vue au contenu

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
place. Cela aide à garder votre esquisse ordonnée dès le départ, plutôt que de
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

## Importation et exportation

### Exporter des objets

Vous pouvez exporter toute pièce sélectionnée vers divers formats vectoriels :

1. Sélectionnez une pièce sur le canevas
2. Choisissez **Objet → Exporter l'objet...** (ou faites un clic droit et
   sélectionnez dans le menu contextuel)
3. Choisissez le format d'exportation :
   - **RFS (.rfs)** : Format d'esquisse paramétrique natif de Rayforge ;
     préserve toutes les contraintes et peut être réimporté pour modification
   - **SVG (.svg)** : Format vectoriel standard ; largement compatible avec les
     logiciels de conception
   - **DXF (.dxf)** : Format d'échange CAO ; compatible avec la plupart des
     applications de CAO

### Enregistrer des esquisses

Vous pouvez enregistrer vos esquisses 2D dans des fichiers pour les réutiliser
dans d'autres projets. Toutes les contraintes paramétriques sont préservées lors
de l'enregistrement, garantissant que vos conceptions conservent leurs relations
géométriques.

### Importer des esquisses

Les esquisses enregistrées peuvent être importées dans n'importe quel espace de
travail, vous permettant de créer une bibliothèque d'éléments de conception
couramment utilisés. Le processus d'importation maintient toutes les contraintes
et relations dimensionnelles.

## Conseils de flux de travail

1. **Commencez par une géométrie approximative** : Créez d'abord des formes de
   base, puis affinez avec des contraintes
2. **Utilisez les contraintes tôt** : Appliquez des contraintes au fur et à
   mesure pour maintenir l'intention de conception
3. **Vérifiez l'état des contraintes** : Le système indique quand les esquisses
   sont entièrement contraintes
4. **Surveillez les conflits** : Les contraintes en conflit sont mises en
   évidence en rouge et affichées dans le panneau des contraintes pour une
   identification facile
5. **Utilisez la symétrie** : Les contraintes de symétrie peuvent accélérer
   considérablement les conceptions complexes
6. **Utilisez la grille** : Activez la grille pour un alignement précis et
   utilisez Ctrl pour l'accrochage à la grille
7. **Itérez et affine** : N'hésitez pas à modifier les contraintes pour obtenir
   le résultat souhaité

## Fonctionnalités d'édition

- **Prise en charge complète d'annuler/rétablir** : L'état complet de
  l'esquisse est enregistré à chaque opération
- **Curseur dynamique** : Le curseur change pour refléter l'outil de dessin
  actif
- **Visualisation des contraintes** : Les contraintes appliquées sont clairement
  indiquées dans l'interface
- **Mises à jour en temps réel** : Les modifications des contraintes mettent à
  jour immédiatement la géométrie
- **Édition par double-clic** : Double-cliquer sur des contraintes
  dimensionnelles (Distance, Rayon, Diamètre, Angle, Rapport d'aspect) ouvre une
  boîte de dialogue pour modifier leurs valeurs
- **Expressions paramétriques** : Les contraintes dimensionnelles prennent en
  charge les expressions, permettant de calculer des valeurs à partir d'autres
  paramètres (par ex., `width/2` pour un rayon égal à la moitié de la largeur)
