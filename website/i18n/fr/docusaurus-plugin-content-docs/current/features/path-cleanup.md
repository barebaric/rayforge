# Nettoyage des chemins

Les logos et dessins provenant d'autres logiciels ne sont pas toujours propres. Un contour peut
s'arrêter une fraction de millimètre avant son point de départ, une forme peut être composée de
plusieurs morceaux ouverts, ou le même chemin peut figurer deux fois dans le fichier. Rayforge
découpe ces chemins tels quels : une forme qui semble seulement fermée est traitée comme ouverte, et
un chemin en double est découpé deux fois.

Les outils de nettoyage des chemins corrigent cela sur les pièces sélectionnées. Vous les trouverez
dans le menu **Objet** et sous **Nettoyer les chemins** dans le menu contextuel du canevas. Chaque
outil correspond à une seule étape d'annulation.

## Fermer les chemins

**Fermer les chemins…** ferme chaque chemin ouvert dont le point de départ et le point d'arrivée
sont plus proches que la tolérance saisie. L'écart est comblé par une ligne droite entre les deux
points existants : la forme ne se déplace pas et ne change pas de taille.

## Joindre les chemins ouverts

**Joindre les chemins ouverts…** relie en chemins plus longs les chemins ouverts dont les extrémités
sont plus proches que la tolérance. Les morceaux sont inversés si nécessaire, l'ordre et le sens
dans lesquels ils ont été dessinés n'ont donc pas d'importance. Si les extrémités d'un chemin joint
se rejoignent ensuite, le chemin est également fermé.

Les deux outils retiennent la dernière tolérance utilisée pendant la session. Une tolérance comprise
entre 0,05 mm et 0,2 mm convient à la plupart des fichiers importés.

## Supprimer les doublons

**Supprimer les doublons** retire les chemins qui se trouvent exactement sur un autre chemin de la
même pièce, quels que soient leur sens ou leur point de départ. Si vous sélectionnez plusieurs
pièces, une pièce qui est la copie exacte d'une autre pièce sélectionnée à la même position est
également retirée.

Les chemins qui ne se chevauchent que sur une partie de leur longueur sont conservés. Le
post-processeur [Fusionner les lignes](merge-lines) s'en charge lors de la génération de la tâche.

## Décomposer

**Décomposer** transforme chaque chemin d'une pièce en une pièce distincte, trous et chemins ouverts
compris. C'est différent de **Scinder**, qui garde un îlot avec ses trous.

Le nettoyage des chemins ne fonctionne que sur les pièces importées et vectorisées. Les pièces
issues d'une esquisse se modifient dans le [Sketcher](sketcher/index).

## Pages connexes

- [Fusionner les lignes](merge-lines) - Ne découper qu'une fois les segments qui se chevauchent
- [Outils du canevas](../ui/canvas-tools) - Sélectionner et supprimer des segments individuels
- [Importer des fichiers](../files/importing) - Importer des dessins dans Rayforge
