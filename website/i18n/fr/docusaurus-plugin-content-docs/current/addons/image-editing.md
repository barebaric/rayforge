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

## Ajuster l'image

Sélectionnez une ou plusieurs pièces et choisissez **Tools - Adjust Image...** pour corriger une
photo avant de la graver ou de la vectoriser :

- **Brightness** (luminosité) éclaircit ou assombrit l'image.
- **Contrast** (contraste) écarte les tons du gris moyen ou les en rapproche.
- **Gamma** au-dessus de 1 éclaircit les tons moyens, en dessous de 1 les assombrit ; le noir et le
  blanc ne changent pas.
- **Sharpen** (netteté) renforce les contours et les détails fins (un masque flou). **Sharpen
  Radius** fixe la taille de ces détails en millimètres, si bien que le résultat est le même quel
  que soit l'intervalle de ligne ou le zoom.

![La boîte de dialogue Ajuster l'image avec les curseurs de tonalité et de netteté](/screenshots/addon-image-editing-adjust.webp)

Le canevas affiche les modifications pendant l'édition. **Apply** les conserve en une seule étape
d'annulation ; fermer la boîte de dialogue sans appliquer les abandonne, et **Reset** remet toutes
les valeurs au neutre. Les zones transparentes restent transparentes.

La boîte de dialogue s'ouvre avec les valeurs de la première pièce sélectionnée, et Apply donne les
mêmes valeurs à toutes les pièces sélectionnées. La netteté est appliquée avant les corrections de
tons, et les autres modifications comme Invert Image gardent leur place.

## Projets plus anciens

Avant cette extension, Inverser était un réglage de l'opération de gravure. Lorsque vous ouvrez un
ancien projet qui l'utilisait, Rayforge déplace le réglage vers les pièces de cette couche, afin que
le résultat reste le même.

## Lorsque l'extension est désactivée

Un projet conserve ses modifications d'image lorsque l'extension est désactivée ou non installée.
Rayforge vous signale que le projet utilise des fonctionnalités non disponibles et affiche les
images sans ces modifications jusqu'à ce que l'extension soit réactivée.
