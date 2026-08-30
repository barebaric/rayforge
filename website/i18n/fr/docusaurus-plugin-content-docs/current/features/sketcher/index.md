---
description: "L'esquisseur paramétrique 2D intégré de Rayforge permet de dessiner des conceptions basées sur des contraintes et pilotées par les dimensions, qui restent modifiables et précises."
---

# Esquisseur paramétrique 2D

Rayforge inclut un esquisseur paramétrique 2D pour dessiner des pièces
directement dans l'application. Au lieu d'importer des illustrations finies
depuis un autre programme, vous esquissez des lignes, des courbes et des
formes sur un canevas infini et vous les reliez entre elles avec des
contraintes. Le résultat est une conception qui reste précise, quelle que
soit la fréquence à laquelle vous changez d'avis sur ses dimensions.

![L'éditeur d'esquisse](/screenshots/addons-sketcher-editor.webp)

## Ce que « paramétrique » signifie ici

Une esquisse est plus qu'un dessin : c'est un petit modèle avec des règles.
Ces règles sont des **contraintes** : des affirmations comme « ces deux
lignes sont parallèles », « cet angle est un angle droit » ou « ce bord
mesure exactement 100 mm ». Après chaque modification, un solveur
réorganise la géométrie pour que toutes les règles soient à nouveau
respectées.

Cela a une conséquence pratique : vous pouvez capturer votre intention de
conception une seule fois, puis continuer à modifier. Passez la contrainte
de distance de 100 mm à 130 mm et la pièce entière suit. Les contraintes
dimensionnelles acceptent aussi les expressions : un rayon de `width/2`
reste la moitié de la largeur, quelle que soit celle-ci.

Quand chaque degré de liberté restant est verrouillé par une contrainte,
l'esquisse est *entièrement contrainte*. L'éditeur vous indique où vous en
êtes grâce aux couleurs : la géométrie tenue par des contraintes est
dessinée en vert, les points non contraints en noir, et une fois l'esquisse
entièrement contrainte le vert devient plus foncé. Les contraintes qui se
contredisent sont marquées en rouge et listées dans le panneau des conflits
de la barre latérale, où vous pouvez les inspecter ou les supprimer.

![Une esquisse cotée](/screenshots/addons-sketcher-constraints.webp)

Une esquisse sous-contrainte n'est pas une erreur — c'est souvent
exactement ce que vous voulez pendant l'expérimentation. La page
[Contraintes](constraints.md) détaille chaque type de contrainte
disponible.

## L'éditeur d'esquisse

Les esquisses vivent dans le document comme toute autre pièce. Créez-en une
avec le bouton **Nouvelle esquisse** du panneau inférieur (ou faites un
clic droit sur le canevas et choisissez la même entrée dans le menu
contextuel), et l'éditeur d'esquisse prend le contrôle de la fenêtre : le
canevas au centre, un panneau de propriétés avec le nom de l'esquisse et
ses paramètres à gauche, et une barre d'outils en haut.

La barre d'outils regroupe les outils de session — annuler et rétablir, les
bascules de visibilité des contraintes et de la géométrie de construction,
les couleurs de remplissage et de ligne, la mise en miroir — ainsi que les
boutons **Terminer** et **Annuler**. **Terminer** enregistre l'esquisse
dans le document ; **Annuler** abandonne les modifications faites pendant
cette session. Pour modifier à nouveau une esquisse existante plus tard,
double-cliquez dessus dans l'espace de travail principal, ou
sélectionnez-la et choisissez **Modifier l'esquisse** dans le menu
contextuel.

L'éditeur privilégie le clavier. La barre d'état en bas affiche toujours
les raccourcis applicables à l'outil et à la sélection courants, de sorte
que les touches pertinentes sont à l'écran exactement au moment où vous en
avez besoin. L'annulation et le rétablissement complets sont disponibles
pour chaque opération.

## Le menu radial

Un clic droit n'importe où dans l'éditeur d'esquisse ouvre le menu radial —
un menu circulaire qui place chaque outil de dessin et de modification à un
clic. Le menu tient compte du contexte : un clic droit dans l'espace vide
propose les outils de dessin, tandis qu'un clic droit sur une ligne
sélectionnée propose les contraintes et les modifications qui ont du sens
pour une ligne. Les outils apparentés sont regroupés ; survolez un groupe
pour déployer ses sous-entrées. Cliquez à nouveau avec le bouton droit pour
fermer le menu ou le rouvrir ailleurs.

![Le menu radial ouvert sur une ligne sélectionnée](/screenshots/addons-sketcher-pie-menu.webp)

## Grille et accrochage

Le canevas affiche une grille adaptative dont l'espacement s'ajuste au
niveau de zoom et est gradué le long des axes dans vos unités préférées ;
elle sert donc aussi de règle : vous pouvez lire tailles et positions
directement sur le canevas.

Pendant que vous dessinez ou glissez, *l'accrochage magnétique* attire le
curseur vers les points de référence proches. Le canevas indique ce vers
quoi le curseur est attiré :

- un **cercle bleu** marque un point existant (extrémité),
- des **flèches vertes** marquent un milieu,
- un **surlignage rose** signifie que le curseur survole un bord,
- des **lignes tiretées** en travers du canevas sont des guides
  d'alignement, affichés lorsque le curseur s'aligne horizontalement ou
  verticalement avec un autre point,
- d'autres indicateurs couvrent les cas particuliers comme les espacements
  équidistants (orange), la tangence (violet) et les centres (rouge).

L'accrochage n'est pas qu'une aide visuelle : poser la géométrie sur une
cible d'accrochage crée automatiquement la contrainte correspondante.
Terminer une ligne sur une extrémité existante rend les deux coïncidentes ;
l'accrochage à un milieu crée une contrainte de symétrie ; les guides
d'alignement deviennent des contraintes horizontales ou verticales. Si vous
préférez un placement libre, `Tab` désactive l'accrochage magnétique.
Maintenir `Shift` pendant un glissement contraint le déplacement à l'axe le
plus proche.

![Guides d'alignement et indicateur d'accrochage équidistant pendant le dessin](/screenshots/addons-sketcher-snap.webp)

## Géométrie de construction

N'importe quelle entité peut être marquée comme géométrie de construction.
Les entités de construction sont dessinées en tiretés, servent de guides de
mise en page pour le solveur comme toute autre géométrie, et sont exclues
des trajectoires d'outils lors de la fabrication de l'esquisse. Elles sont
pratiques pour les lignes de centre, les cercles de construction et
l'échafaudage derrière les conceptions symétriques. La bascule de
construction dans la barre d'outils les masque quand elles gênent.

## Pour aller plus loin

[Création de géométrie 2D](geometry.md) présente les outils de dessin et
leurs modificateurs, [Outils de l'esquisseur](tools.md) est la référence
des raccourcis clavier et des modifications comme le décalage, le chanfrein
et le congé, [Tableaux](arrays.md) couvre les tableaux circulaires et le
long d'une courbe, et [Expressions](expressions.md) explique les
paramètres, les expressions et les zones de texte paramétriques. Les
esquisses peuvent être enregistrées puis réimportées avec toutes leurs
contraintes — voir [Importation et exportation](import-export.md).
