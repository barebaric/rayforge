---
description: "Fonctionnement des contraintes dans l'esquisseur Rayforge : ajout, modification, sélection, suppression et résolution des conflits."
---

# Contraintes

Les contraintes sont les règles qui maintiennent une esquisse cohérente.
Chacune est une petite affirmation sur la géométrie — « ces deux points ne
font qu'un », « cette ligne mesure exactement 80 mm » — et après chaque
modification le solveur réorganise l'esquisse pour que toutes les
affirmations restent vraies simultanément. Une géométrie sans contraintes
peut dériver librement ; chaque contrainte ajoutée verrouille un degré de
liberté.

Il existe deux familles. Les **contraintes géométriques** capturent des
relations qui ne portent aucune mesure : coïncidence, horizontalité,
tangence, symétrie. Les **contraintes dimensionnelles** attachent un
nombre à la géométrie : une distance, un rayon, un angle. Les valeurs
dimensionnelles acceptent des expressions (voir
[plus bas](#editing-dimensional-values)) : c'est là que le côté
« paramétrique » de l'esquisse paramétrique prend tout son sens.

Le solveur signale son état par les couleurs. La géométrie tenue par des
contraintes est dessinée en vert, les points non contraints en noir, et une
esquisse entièrement contrainte rend le vert plus foncé. Les marqueurs de
contraintes valides sont verts, les marqueurs basés sur des expressions
orange, et les marqueurs des contraintes que le solveur ne peut pas
satisfaire deviennent rouges (voir
[conflits](#when-constraints-conflict)).

## Ajouter une contrainte

Sélectionnez la géométrie à laquelle la contrainte doit s'appliquer, puis
appuyez sur le raccourci clavier ou choisissez la contrainte dans le menu
radial — les contraintes géométriques se trouvent dans le groupe
**Contraindre**, les contraintes dimensionnelles dans le groupe
**Dimension**. Chaque contrainte exige une sélection particulière :

| Contrainte                     | Sélection                        | Raccourci  |
| ------------------------------ | -------------------------------- | ---------- |
| Horizontale / Verticale        | 2 points, ou des lignes          | `H` / `V`  |
| Coïncidence / Point sur forme  | 2 points, ou un point + une forme | `O` ou `C` |
| Perpendiculaire                | 2 formes                         | `N`        |
| Tangente                       | 1 ligne + 1 arc ou cercle        | `T`        |
| Symétrie                       | 3 points, ou 2 points + 1 ligne  | `S`        |
| Longueur égale                 | 2 formes ou plus                 | `E`        |
| Distance                       | 2 points, ou 1 ligne             | `K+D`      |
| Diamètre                       | 1 cercle                         | `K+O`      |
| Rayon                          | 1 arc ou cercle                  | `K+R`      |
| Angle                          | 2 lignes                         | `K+A`      |
| Rapport d'aspect               | 2 lignes                         | `K+X`      |

L'ordre de la sélection n'a jamais d'importance, à une exception près :
avec trois points sélectionnés, la Symétrie utilise le **dernier** point
comme centre de miroir. Un raccourci ne se déclenche que si la sélection
courante correspond à la contrainte — tout le reste est également filtré du
menu radial.

Des contraintes apparaissent aussi d'elles-mêmes pendant que vous dessinez :
l'accrochage à une extrémité crée une contrainte de coïncidence, et les
guides d'alignement deviennent des contraintes horizontales ou verticales
(voir [l'aperçu de l'esquisseur](index.md#grid-and-snapping)).

## Contraintes géométriques

Une contrainte de **coïncidence** fusionne deux points distincts en un seul
emplacement. Sélectionnez les deux points et ils sont rapprochés ; le
marqueur est un anneau autour du point joint. Tracer une ligne qui se
termine exactement sur une extrémité existante crée cette contrainte
automatiquement.

![Deux lignes jointes par une contrainte de coïncidence](/screenshots/addons-sketcher-constraint-coincident.webp)

**Horizontale** et **Verticale** font pivoter la ligne sélectionnée, ou la
paire de points sélectionnés, sur un axe. Les marqueurs sont de petites
barres — horizontale et verticale respectivement — dessinées à côté de la
géométrie.

![Une contrainte horizontale](/screenshots/addons-sketcher-constraint-horizontal.webp)

![Une contrainte verticale](/screenshots/addons-sketcher-constraint-vertical.webp)

**Perpendiculaire** force deux formes à se rencontrer à angle droit. Cela
fonctionne pour deux lignes, une ligne et un arc ou cercle, ou deux arcs et
cercles. Le marqueur est un arc d'angle droit à l'intersection.

![Deux lignes se rencontrant à angle droit](/screenshots/addons-sketcher-constraint-perpendicular.webp)

**Tangente** adoucit la transition là où une ligne rencontre un arc ou un
cercle : la ligne est pivotée pour toucher la courbe sans la traverser. Son
marqueur est un petit « T » au point de contact.

![Une ligne tangente à un cercle](/screenshots/addons-sketcher-constraint-tangent.webp)

**Point sur forme** attache un point sur une ligne, un arc ou un cercle —
sans le fusionner avec un point particulier comme le fait la coïncidence.
Sélectionnez un point et une forme ; le marqueur est un anneau autour du
point contraint. Quand la forme est une courbe (Bézier), le point est
contraint à glisser le long d'elle.

![Une extrémité de ligne reposant sur une autre ligne](/screenshots/addons-sketcher-constraint-point-on-line.webp)

**Symétrie** met deux points en miroir par rapport à un centre ou un axe,
avec les deux modes déjà mentionnés : sélectionnez trois points et le
dernier devient le centre autour duquel les deux premiers se miroitent, ou
sélectionnez deux points et une ligne pour vous mirroiter par rapport à
cette ligne. Le marqueur est une paire de pointes de flèches opposées au
milieu entre les points mirrorés.

![Deux points mis en miroir par rapport à une ligne](/screenshots/addons-sketcher-constraint-symmetry.webp)

Une septième contrainte géométrique, **collinéaire**, force des points sur
une même ligne infinie. Elle n'a pas de marqueur sur le canevas et ne peut
pas être appliquée à la main — les outils chanfrein et congé la créent pour
garder le coin modifié aligné.

## Contraintes dimensionnelles

La contrainte de **distance** fixe l'écart entre deux points, ou la
longueur d'une ligne. Son étiquette affiche la valeur actuelle au milieu de
la portée mesurée ; lorsque les deux points ne sont pas déjà reliés par une
ligne, une ligne de rappel tiretée indique clairement ce qui est mesuré.

![Une contrainte de distance de 80 mm](/screenshots/addons-sketcher-constraint-distance.webp)

Les cercles et les arcs ont leurs propres dimensions. **Diamètre** étiquette
la largeur complète d'un cercle avec un préfixe `Ø`, **rayon** étiquette la
distance depuis le centre d'un arc ou d'un cercle avec un préfixe `R`, et
tous deux placent l'étiquette juste à l'extérieur de la forme avec une
courte ligne de rappel.

![Une contrainte de diamètre](/screenshots/addons-sketcher-constraint-diameter.webp)

![Une contrainte de rayon](/screenshots/addons-sketcher-constraint-radius.webp)

La contrainte d'**angle** définit l'angle entre deux lignes sélectionnées.
Elle dessine un arc entre les deux directions à leur intersection, étiqueté
avec la valeur en degrés.

![Une contrainte d'angle de 45 degrés](/screenshots/addons-sketcher-constraint-angle.webp)

Le **rapport d'aspect** lie les longueurs de deux lignes : la longueur de
la première divisée par la longueur de la seconde doit être égale à la
valeur donnée. Son marqueur, une paire de crochets d'angle opposés, se
trouve à la jonction où les lignes se rencontrent.

![Une contrainte de rapport d'aspect entre deux lignes](/screenshots/addons-sketcher-constraint-aspect-ratio.webp)

Enfin, la contrainte de **longueur égale** appliquée à deux lignes, arcs,
cercles ou ellipses ou plus leur fait partager une même longueur ou un même
rayon, en marquant chaque forme d'un signe `=`. Le solveur utilise aussi en
interne une variante à distance égale de cette contrainte — par exemple
pour garder un cercle rond ou les deux côtés d'un chanfrein symétriques —
qui porte le même marqueur `=` mais ne peut pas être appliquée à la main.

![Deux lignes de longueur égale](/screenshots/addons-sketcher-constraint-equal-length.webp)

## Modifier les valeurs dimensionnelles {#editing-dimensional-values}

Double-cliquez sur l'étiquette d'une contrainte dimensionnelle pour la
modifier. La boîte de dialogue accepte un nombre simple ou une expression :
les paramètres d'esquisse et les variables d'entrée peuvent être référencés
par leur nom, et les fonctions mathématiques sont disponibles — un rayon de
`width/2` suit le paramètre de largeur où qu'il aille. Une fois qu'une
contrainte est pilotée par une expression, son marqueur devient orange pour
rappeler que le nombre est calculé, pas saisi. La syntaxe complète, ainsi
que les paramètres d'esquisse qu'elle peut référencer, est décrite dans
[Expressions](expressions.md).

Double-cliquer sur une ligne, un arc ou un cercle pas encore coté propose de
créer directement la dimension correspondante (distance, rayon ou
diamètre).

## Sélection et suppression

Les marqueurs de contraintes participent à la sélection comme tout le
reste : le survol affiche un surlignage jaune et une infobulle avec le nom
de la contrainte, et un clic la sélectionne en la dessinant en bleu.
Appuyer sur `Delete` supprime la contrainte sélectionnée et libère la
géométrie qu'elle tenait. Supprimer une géométrie emporte ses contraintes
avec elle. Pour les contraintes dimensionnelles, la boîte de dialogue
d'édition décrite ci-dessus n'a pas de bouton de suppression — retirer une
dimension se fait par une suppression normale du marqueur sélectionné.

## En cas de conflit entre contraintes {#when-constraints-conflict}

Des contraintes qui se contredisent — un triangle dont les côtés ne
peuvent pas être tous vrais à la fois, par exemple — ne peuvent pas casser
l'esquisse : le solveur fait de son mieux et signale ce qu'il n'a pas pu
satisfaire. Les contraintes en conflit deviennent rouges, leurs marqueurs
comme la géométrie qu'elles tiennent, de sorte que la zone endommagée est
visible d'un coup d'œil.

![Contraintes de distance en conflit, signalées dans la barre latérale](/screenshots/addons-sketcher-conflicts.webp)

La barre latérale liste chaque conflit sous **Contraintes en conflit**,
chaque ligne nommant la contrainte et les points qu'elle touche. Les lignes
sont interactives : en survoler une met en évidence la contrainte sur le
canevas, cliquer sur une la sélectionne, et le bouton de suppression à
droite la retire. En général, le moyen le plus rapide de sortir d'un
conflit est de supprimer ou de revaloriser la contrainte qui exprime
l'intention périmée — la liste existe précisément parce que le solveur ne
peut pas deviner laquelle des règles contradictoires est la mauvaise.

## Pour aller plus loin

Chaque outil de dessin est documenté sur sa propre page — voir
[Tracé](path.md), [Arc et ellipse](arc-ellipse.md) et
[Rectangle](rectangle.md) pour savoir comment dessiner les formes
auxquelles ces contraintes s'attachent.
