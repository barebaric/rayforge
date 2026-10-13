# Édition d'images

L'extension Édition d'images change l'apparence de l'image d'une pièce sans toucher au fichier
importé. Chaque modification est enregistrée avec la pièce, ce qui permet de l'annuler, de la
supprimer plus tard ou de placer plusieurs fois le même fichier avec des modifications différentes.

Le canevas, l'aperçu 3D et chaque opération utilisent l'image modifiée. Ce que vous voyez sur le
canevas est ce que le laser grave.

## Inverser l'image

Sélectionnez une ou plusieurs pièces et choisissez **Tools - Invert Image**. Les zones claires
deviennent sombres et les zones sombres deviennent claires ; les zones transparentes restent
transparentes, l'arrière-plan n'est donc toujours pas gravé.

![Une photo en niveaux de gris de montagnes et d'un lac, telle qu'importée](/images/image-edit-invert-before.webp)

![La même photo après Invert Image](/images/image-edit-invert-after.webp)

Choisissez-le à nouveau pour supprimer l'inversion. Avec une sélection mixte, les pièces qui ne sont
pas encore inversées le sont, pour que toute la sélection finisse identique.

Utilisez-le pour les lithophanes et le gaufrage dans une opération de gravure, lorsque les zones
claires doivent être gravées en profondeur, ou pour graver un motif clair sur un fond sombre.

## Projets plus anciens

Avant cette extension, Inverser était un réglage de l'opération de gravure. Lorsque vous ouvrez un
ancien projet qui l'utilisait, Rayforge déplace le réglage vers les pièces de cette couche, afin que
le résultat reste le même.

## Lorsque l'extension est désactivée

Un projet conserve ses modifications d'image lorsque l'extension est désactivée ou non installée.
Rayforge vous signale que le projet utilise des fonctionnalités non disponibles et affiche les
images sans ces modifications jusqu'à ce que l'extension soit réactivée.
