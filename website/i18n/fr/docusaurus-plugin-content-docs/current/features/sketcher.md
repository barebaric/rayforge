# Dessinateur 2D Paramétrique

Le Dessinateur 2D Paramétrique est une fonctionnalité puissante de Rayforge qui vous permet de créer et d'éditer des designs 2D précis basés sur des contraintes directement dans l'application. Cette fonctionnalité vous permet de concevoir des pièces personnalisées from scratch sans avoir besoin de logiciel CAO externe.

## Aperçu

Le dessinateur fournit un ensemble complet d'outils pour créer des formes géométriques et appliquer des contraintes paramétriques pour définir des relations précises entre les éléments. Cette approche assure que vos designs maintiennent leur géométrie prévue même lorsque les dimensions sont modifiées.

## Créer et Éditer des Esquisses

### Créer une Nouvelle Esquisse

1. Cliquez sur le bouton "Nouvelle Esquisse" dans la barre d'outils ou utilisez le menu principal
2. Un nouvel espace de travail d'esquisse vide s'ouvrira avec l'interface de l'éditeur d'esquisse
3. Commencez à créer de la géométrie en utilisant les outils de dessin depuis le menu circulaire ou les raccourcis clavier
4. Appliquez des contraintes pour définir les relations entre les éléments
5. Cliquez sur "Terminer l'Esquisse" pour sauvegarder votre travail et retourner à l'espace de travail principal

### Éditer des Esquisses Existantes

1. Double-cliquez sur une pièce basée sur esquisse dans l'espace de travail principal
2. Alternativement, sélectionnez une esquisse et choisissez "Éditer l'Esquisse" depuis le menu contextuel
3. Effectuez vos modifications en utilisant les mêmes outils et contraintes
4. Cliquez sur "Terminer l'Esquisse" pour sauvegarder les changements ou "Annuler l'Esquisse" pour les ignorer

## Créer de la Géométrie 2D

Le dessinateur supporte la création des éléments géométriques de base suivants :

- **Tracés (Lignes et Courbes de Bézier)** : Dessinez des lignes droites et des courbes
  de bézier fluides avec l'outil de tracé unifié. Cliquez pour placer des points,
  glissez pour créer des poignées de bézier.
- **Arcs** : Dessinez des arcs en spécifiant un point central, un point de départ et un point de fin
- **Cercles** : Créez des cercles en définissant un point central et un rayon
- **Rectangles** : Dessinez des rectangles en spécifiant deux coins opposés
- **Rectangles Arrondis** : Dessinez des rectangles avec des coins arrondis
- **Zones de Texte** : Ajoutez des éléments de texte à votre esquisse
- **Remplissages** : Remplissez des régions fermées pour créer des zones solides

Ces éléments forment la base de vos designs 2D et peuvent être combinés pour créer des formes complexes. Les remplissages sont particulièrement utiles pour créer des régions solides qui seront gravées ou coupées comme une seule pièce.

## Travailler avec les Courbes de Bézier

L'outil de tracé supporte les courbes de bézier pour créer des formes fluides et organiques :

### Dessiner des Courbes de Bézier

1. Sélectionnez l'outil de tracé depuis le menu circulaire ou utilisez le raccourci clavier
2. Cliquez pour placer des points - chaque clic crée un nouveau point
3. Glissez après avoir cliqué pour créer des poignées de bézier pour des courbes fluides
4. Continuez à ajouter des points pour construire votre tracé
5. Pressez Échap ou double-cliquez pour terminer le tracé

### Éditer des Courbes de Bézier

- **Déplacer des points** : Cliquez et glissez n'importe quel point pour le repositionner
- **Ajuster les poignées** : Glissez les extrémités des poignées pour modifier la forme de la courbe
- **Connecter à des points existants** : Lors de l'édition d'un tracé, vous pouvez vous aligner sur des points existants dans votre esquisse
- **Rendre fluide/symétrique** : Les points connectés par une contrainte coïncidente peuvent être rendus fluides (tangente continue) ou symétriques (poignées en miroir)

### Convertir des Courbes en Lignes

Utilisez l'**outil de redressement** pour convertir les courbes de bézier en lignes droites.
Ceci est utile lorsque vous avez besoin d'une géométrie simple et propre. Sélectionnez les segments
de bézier que vous souhaitez convertir et appliquez l'action de redressement.

## Système de Contraintes Paramétriques

Le système de contraintes est le cœur du dessinateur paramétrique, vous permettant de définir des relations géométriques précises :

### Contraintes Géométriques

- **Coïncident** : Force deux points à occuper le même emplacement
- **Vertical** : Contraint une ligne à être parfaitement verticale
- **Horizontal** : Contraint une ligne à être parfaitement horizontale
- **Tangent** : Rend une ligne tangente à un cercle ou arc
- **Perpendiculaire** : Force deux lignes, une ligne et un arc/cercle, ou deux arcs/cercles à se rencontrer à 90 degrés
- **Point sur Ligne/Forme** : Contraint un point à se trouver sur une ligne, un arc ou un cercle
- **Symétrie** : Crée des relations symétriques entre éléments. Supporte deux modes :
  - **Symétrie de Point** : Sélectionnez 3 points (le premier est le centre)
  - **Symétrie de Ligne** : Sélectionnez 2 points et 1 ligne (la ligne est l'axe)

### Contraintes Dimensionnelles

- **Distance** : Définit la distance exacte entre deux points ou le long d'une ligne
- **Diamètre** : Définit le diamètre d'un cercle
- **Rayon** : Définit le rayon d'un cercle ou d'un arc
- **Angle** : Impose un angle spécifique entre deux lignes
- **Ratio d'Aspect** : Force le ratio entre deux distances à être égal à une valeur spécifiée
- **Longueur/Rayon Égal** : Force plusieurs éléments (lignes, arcs ou cercles) à avoir la même longueur ou le même rayon
- **Distance Égale** : Force la distance entre deux paires de points à être égale

## Interface du Menu Circulaire

Le dessinateur propose un menu contextuel adaptatif qui fournit un accès rapide à tous les outils de dessin et de contrainte. Ce menu radial apparaît lorsque vous faites un clic droit dans l'espace de travail d'esquisse et s'adapte selon votre contexte et sélection actuels.

Les éléments du menu circulaire affichent dynamiquement les options disponibles selon ce que vous avez sélectionné. Par exemple, en cliquant sur un espace vide, vous verrez les outils de dessin. En cliquant sur de la géométrie sélectionnée, vous verrez les contraintes applicables.

![Menu Circulaire du Dessinateur](/screenshots/sketcher-pie-menu.png)

## Raccourcis Clavier

Le dessinateur fournit des raccourcis clavier pour un flux de travail efficace :

### Raccourcis d'Outils
- `Espace` : Outil de sélection
- `G+P` : Outil tracé (lignes et courbes de bézier)
- `G+A` : Outil arc
- `G+C` : Outil cercle
- `G+R` : Outil rectangle
- `G+O` : Outil rectangle arrondi
- `G+F` : Outil zone de remplissage
- `G+T` : Outil zone de texte
- `G+G` : Outil grille (basculer la visibilité de la grille)
- `G+N` : Basculer le mode construction sur la sélection

### Raccourcis d'Action
- `C+H` : Ajouter un coin de chanfrein
- `C+F` : Ajouter un coin de congé
- `C+S` : Redresser les courbes de bézier sélectionnées en lignes

### Raccourcis de Contrainte
- `H` : Appliquer la contrainte Horizontale
- `V` : Appliquer la contrainte Verticale
- `N` : Appliquer la contrainte Perpendiculaire
- `T` : Appliquer la contrainte Tangente
- `E` : Appliquer la contrainte Égal
- `O` ou `C` : Appliquer la contrainte d'Alignement (Coïncident)
- `S` : Appliquer la contrainte de Symétrie
- `K+D` : Appliquer la contrainte de Distance
- `K+R` : Appliquer la contrainte de Rayon
- `K+O` : Appliquer la contrainte de Diamètre
- `K+A` : Appliquer la contrainte d'Angle
- `K+X` : Appliquer la contrainte de Ratio d'Aspect

### Raccourcis Généraux
- `Ctrl+Z` : Annuler
- `Ctrl+Y` ou `Ctrl+Shift+Z` : Rétablir
- `Supprimer` : Supprimer les éléments sélectionnés
- `Échap` : Annuler l'opération actuelle ou désélectionner
- `F` : Ajuster la vue au contenu

## Mode Construction

Le mode construction vous permet de marquer des entités comme "géométrie de construction" - des éléments d'aide utilisés pour guider votre design mais ne faisant pas partie de la sortie finale. Les entités de construction sont affichées différemment (typiquement comme des lignes pointillées) et ne sont pas incluses lorsque l'esquisse est utilisée pour la découpe ou gravure laser.

Pour basculer le mode construction :
- Sélectionnez une ou plusieurs entités
- Appuyez sur `N` ou `G+N`, ou utilisez l'option Construction dans le menu circulaire

Les entités de construction sont utiles pour :
- Créer des lignes et cercles de référence
- Définir une géométrie temporaire pour l'alignement
- Construire des formes complexes à partir d'un cadre de guides

## Grille et Contrôles de Visibilité

### Outil Grille

L'outil grille fournit une référence visuelle pour l'alignement et le dimensionnement :

- Basculer la grille on/off en utilisant le bouton de l'outil grille ou `G+G`
- La grille s'adapte à votre niveau de zoom pour un espacement constant
- Maintenez `Ctrl` en plaçant ou déplaçant des points pour s'aligner sur la grille

### Contrôles Afficher/Masquer

La barre d'outils du dessinateur inclut des boutons de basculement pour contrôler la visibilité :

- **Afficher/masquer la géométrie de construction** : Basculer la visibilité des entités de construction
- **Afficher/masquer les contraintes** : Basculer la visibilité des marqueurs de contraintes

Ces contrôles aident à réduire l'encombrement visuel lors du travail sur des esquisses complexes.

### Mouvement Contraint à l'Axe

Lors du déplacement de points ou de géométrie, maintenez `Maj` pour contraindre le mouvement
à l'axe le plus proche (horizontal ou vertical). Ceci est utile pour maintenir l'alignement
lors des ajustements.

## Chanfrein et Congé

Le dessinateur fournit des outils pour modifier les coins de votre géométrie :

- **Chanfrein** : Remplace un coin vif par un bord biseauté. Sélectionnez un point de jonction (où deux lignes se rencontrent) et appliquez l'action de chanfrein.
- **Congé** : Remplace un coin vif par un bord arrondi. Sélectionnez un point de jonction (où deux lignes se rencontrent) et appliquez l'action de congé.

Pour utiliser chanfrein ou congé :
1. Sélectionnez un point de jonction où deux lignes se rencontrent
2. Appuyez sur `C+H` pour chanfrein ou `C+F` pour congé
3. Utilisez le menu circulaire ou les raccourcis clavier pour appliquer la modification

## Import et Export

### Exporter des Objets

Vous pouvez exporter toute pièce sélectionnée vers différents formats vectoriels :

1. Sélectionnez une pièce sur le canevas
2. Choisissez **Objet → Exporter l'Objet...** (ou clic droit et sélectionnez depuis le menu contextuel)
3. Choisissez le format d'exportation :
   - **RFS (.rfs)** : Format d'esquisse paramétrique natif de Rayforge - préserve toutes les contraintes et peut être ré-importé pour édition
   - **SVG (.svg)** : Format vectoriel standard - largement compatible avec les logiciels de design
   - **DXF (.dxf)** : Format d'échange CAO - compatible avec la plupart des applications CAO

### Sauvegarder des Esquisses

Vous pouvez sauvegarder vos esquisses 2D dans des fichiers pour réutilisation dans d'autres projets. Toutes les contraintes paramétriques sont préservées lors de la sauvegarde, assurant que vos designs maintiennent leurs relations géométriques.

### Importer des Esquisses

Les esquisses sauvegardées peuvent être importées dans tout espace de travail, vous permettant de créer une bibliothèque d'éléments de design couramment utilisés. Le processus d'importation maintient toutes les contraintes et relations dimensionnelles.

## Conseils de Flux de Travail

1. **Commencez avec une Géométrie Approximative** : Créez d'abord des formes de base, puis affinez avec des contraintes
2. **Utilisez les Contraintes Tôt** : Appliquez des contraintes pendant que vous construisez pour maintenir l'intention du design
3. **Vérifiez le Statut des Contraintes** : Le système indique quand les esquisses sont totalement contraintes
4. **Surveillez les Conflits** : Les contraintes qui entrent en conflit les unes avec les autres sont surlignées en rouge et affichées dans le panneau des contraintes pour une identification facile
5. **Utilisez la Symétrie** : Les contraintes de symétrie peuvent accélérer significativement les designs complexes
6. **Utilisez la Grille** : Activez la grille pour un alignement précis, et utilisez Ctrl pour s'aligner sur la grille
7. **Itérez et Affinez** : N'hésitez pas à modifier les contraintes pour obtenir le résultat souhaité

## Fonctionnalités d'Édition

- **Support Complet Annuler/Rétablir** : Tout l'état de l'esquisse est sauvegardé avec chaque opération
- **Curseur Dynamique** : Le curseur change pour refléter l'outil de dessin actif
- **Visualisation des Contraintes** : Les contraintes appliquées sont clairement indiquées dans l'interface
- **Mises à Jour en Temps Réel** : Les changements de contraintes mettent à jour immédiatement la géométrie
- **Édition par Double-Clic** : Double-cliquer sur les contraintes dimensionnelles (Distance, Rayon, Diamètre, Angle, Ratio d'Aspect) ouvre une boîte de dialogue pour éditer leurs valeurs
- **Expressions Paramétriques** : Les contraintes dimensionnelles supportent les expressions, permettant aux valeurs d'être calculées à partir d'autres paramètres (ex : `largeur/2` pour un rayon qui est la moitié de la largeur)
