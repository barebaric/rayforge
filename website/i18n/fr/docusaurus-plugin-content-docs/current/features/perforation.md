---
description:
  "Découpez des lignes en pointillés dans Rayforge avec le post-traitement Perforation : réglez une
  longueur de coupe et une longueur de saut pour plis, lignes prédécoupées et trous de couture."
---

# Perforation

Tous les traits ne servent pas à détacher une pièce du matériau. Certains existent pour que le
matériau se plie le long d'eux, se déchire proprement, ou pour marquer le passage de l'aiguille.
C'est le rôle du post-traitement **Perforation** : il transforme une coupe continue en coupe en
pointillés. Au lieu de laisser le laser allumé du début à la fin, il tire sur la **Longueur de
coupe**, parcourt la **Longueur de saut** laser éteint, et répète ce rythme le long de chaque
contour de la pièce. Il reste une ligne de petites coupes qui maintient le matériau — jusqu'à ce que
vous décidiez de le plier ou de le déchirer.

![Les réglages de la Perforation sur la page de post-traitement d'une étape de Contour](/screenshots/step-settings-contour-post-perforation.webp)

## Quelques cas d'usage

Chaque fois qu'un trait doit affaiblir le matériau sans le couper, une coupe perforée a du sens. Les
grands classiques :

- **Les lignes de pliage** dans le carton et le carton ondulé se plient net le long d'une coupe en
  pointillés — boîtes, cartes et créations pop-up, par exemple.
- **Les lignes prédécoupées** ouvrent billets, bons et emballages sans ciseaux.
- **Les trous de couture** dans le cuir marquent le passage de l'aiguille avant la couture.
- **Les contours en pointillés** sont en eux-mêmes un effet décoratif, sur le bois, le papier ou
  l'acrylique.

## Activer la perforation

La perforation se trouve sur la page **Post-traitement** des réglages d'une étape de
[Contour](operations/contour), aux côtés des autres post-traitements. Elle est désactivée par défaut
; dès que vous basculez l'interrupteur, deux valeurs apparaissent :

- La **Longueur de coupe** est la distance pendant laquelle le laser reste allumé avant chaque
  interruption.
- La **Longueur de saut** est la distance parcourue laser éteint entre deux coupes.

Le motif recommence au début de chaque contour, toujours par une coupe complète, pour que toutes les
lignes en pointillés de la pièce démarrent de la même façon. Un contour trop court pour ne serait-ce
qu'une coupe et un saut est simplement découpé d'un seul tenant.

## Le résultat

Sur le canevas, la ligne en pointillés remplace le contour de coupe plein : chaque interruption du
tracé sera aussi une interruption dans la pièce finie.

![Un rectangle et un cercle découpés en pointillés dans une planche de contreplaqué](/screenshots/operations-perforation.webp)

## Conseils

Pour une ligne de pliage qui ne doit pas traverser le matériau, commencez avec une longueur de saut
proche de la longueur de coupe et baissez la puissance : une rainure peu profonde plie aussi bien
qu'une coupe pointillée profonde, et l'envers reste propre.

Les longueurs très courtes (quelques dixièmes de millimètre) font allumer et éteindre le laser à
grande cadence, ce qui réduit la puissance effective et peut laisser des traces irrégulières. Si le
motif paraît inégal, essayez des valeurs plus longues.

La perforation se combine bien avec le reste du post-traitement : les
[Ponts de Maintien](holding-tabs), l'[Approche / Sortie](lead-in-out) et la
[Passe Multiple](multi-pass) continuent de fonctionner comme d'habitude.

## Pages associées

- [Coupe de Contour](operations/contour) — l'opération de coupe à laquelle appartient ce
  post-traitement
- [Ponts de Maintien](holding-tabs) — interruptions placées à la main pour garder les pièces
  attachées
