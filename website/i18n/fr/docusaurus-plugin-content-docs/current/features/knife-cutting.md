---
description:
  "Découpez le vinyle, le carton, le cuir et autres matériaux en feuille avec une lame traînante ou
  une lame tangentielle motorisée au lieu d'un laser."
---

# Découpe à la lame

Rayforge peut aussi piloter des porte-lames en plus des lasers. Deux étapes intégrées découpent les
contours vectoriels avec une lame : **Découpe à la lame traînante** pour un porte-lame à lame
traînante et **Découpe à la lame tangentielle** pour une lame rotative motorisée. Les étapes de
découpe à la lame n'apparaissent que sur les machines équipées d'une tête de lame compatible, ce qui
permet de mélanger des opérations laser et de découpe à la lame dans un même projet.

## Prérequis

- Une machine avec une tête **Lame traînante** ou **Lame tangentielle**. Ajoutez la tête dans
  [Réglages de la machine → Têtes](../machine/general.md) avec **Ajouter une tête**.
- Un contrôleur G-code capable de déplacer la machine sans laser. Les étapes de lame découpent sur
  la ligne médiane et n'utilisent pas la puissance laser.
- L'extension Drag Knife intégrée. Elle est fournie avec Rayforge ; si elle a été désactivée,
  activez-la dans **Réglages → Extensions**.

## Configurer une tête de lame

1. Ouvrez **Machine → Réglages actuels de la machine → Têtes**.
2. Cliquez sur **Ajouter une tête** et choisissez **Lame traînante** ou **Lame tangentielle**.
3. Donnez un nom à la tête. Pour une lame traînante, saisissez le **Décalage de la lame** : la
   distance entre la pointe de la lame et le point de pivot de la machine.
4. Fermez les réglages. Les étapes de lame correspondantes apparaissent désormais lors de l'ajout
   d'une opération.

La tête de lame est aussi une capacité de la machine : les opérations qui nécessitent une lame ne
sont disponibles que tant que la machine active en possède une.

## Découpe à la lame traînante

Une lame traînante possède une lame qui suit derrière le point de pivot. Rayforge compense
automatiquement le décalage, afin que la lame suive le contour plutôt que le point de pivot.

1. Sélectionnez les contours à découper.
2. Ajoutez une opération **Découpe à la lame traînante**.
3. Choisissez la tête **Lame** et réglez le **Décalage de la lame**, la **Hauteur Z de sécurité**,
   l'**Angle de pivotement** et la vitesse de découpe.

![Réglages de la découpe à la lame traînante](/screenshots/step-settings-drag-knife-general.webp)

L'étape réécrit chaque contour en la trajectoire que la machine doit suivre : un arc d'entrée aligne
la lame sur la première direction de coupe, les segments droits sont décalés vers l'avant du
décalage de la lame, les angles dans l'angle de pivotement sont parcourus lame baissée, les angles
plus vifs relèvent la lame pour pivoter dans l'air, et un arc de sortie dépasse la fin de la
trajectoire.

Réglez le **Décalage de la lame** d'après la documentation du porte-lame ou mesurez-le. Une valeur
trop faible laisse des angles arrondis ; une valeur trop grande les dépasse. L'**Angle de
pivotement** limite l'angle de rotation lame baissée ; les angles plus vifs relèvent d'abord la lame
pour qu'elle ne traîne pas sur le côté.

## Découpe à la lame tangentielle

Une lame tangentielle possède un moteur qui fait tourner la lame pour la maintenir tangente à la
trajectoire. L'étape ajoute la direction de la trajectoire comme valeur de l'axe **A** à chaque
mouvement de coupe, afin que le contrôleur puisse tourner la lame.

1. Sélectionnez les contours à découper.
2. Ajoutez une opération **Découpe à la lame tangentielle**.
3. Choisissez la tête **Lame** et réglez la **Tolérance d'angle**, la **Tolérance de rayon**, la
   **Vitesse de broche**, la **Hauteur Z de sécurité** et la vitesse de découpe.

![Réglages de la découpe à la lame tangentielle](/screenshots/step-settings-tangential-knife-general.webp)

Les changements de direction supérieurs à la **Tolérance d'angle** ou les arcs plus serrés que la
**Tolérance de rayon** relèvent la lame à la Z de sécurité, la font pivoter et la replongent. Les
changements plus faibles font tourner la lame sur place, lame baissée. La **Vitesse de broche** est
émise comme commande M3 pour le moteur de la lame ; Rayforge n'utilise jamais M4, car le moteur de
la lame ne tourne que dans le sens horaire.

## Astuces

- Lancez d'abord la tâche à vitesse réduite, surtout sur un matériau épais ou fibreux.
- Utilisez un tapis de découpe sacrificiel ou une faible profondeur pour que la lame ne marque pas
  le plateau.
- Les étapes de lame sont des opérations G-code ordinaires : elles se combinent avec les étapes
  laser, se réordonnent, se dupliquent et s'enregistrent dans des recettes comme les autres.
