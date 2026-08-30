---
description:
  "Dessinez des arcs et des ellipses (y compris des cercles) dans l'esquisseur Rayforge, avec
  touches modificatrices et saisie de dimensions."
---

# Arc et ellipse

L'esquisseur fournit deux outils de formes courbes : l'**outil d'arc** pour les arcs circulaires et
l'**outil d'ellipse** pour les ellipses et les cercles.

![Un arc et une ellipse tels que créés par leurs outils](/screenshots/addons-sketcher-tool-arc-ellipse.webp)

## Outil d'arc

L'outil d'arc (`G+A`) crée un arc en trois clics :

1. Cliquez sur le point **central**.
2. Cliquez sur le point de **départ** — sa distance au centre définit le rayon.
3. Déplacez le curseur pour prévisualiser l'arc balayant entre les deux points et cliquez sur la
   position d'**arrivée**.

Pendant que l'aperçu est actif, vous pouvez saisir un nombre pour fixer exactement le rayon ;
appuyez sur `Tab` ou `Entrée` pour l'appliquer. `Tab` avant la saisie bascule l'accrochage
magnétique.

## Outil d'ellipse

L'outil d'ellipse (`G+C`) crée des ellipses et des cercles en deux clics : le premier définit le
centre, le second le point du bord. Vous pouvez également appuyer au centre, glisser et relâcher au
bord — les deux gestes fonctionnent de façon interchangeable.

- Maintenez `Ctrl` pour contraindre la forme à un cercle parfait.
- Maintenez `Shift` pour utiliser le point de départ comme centre de l'ellipse.

## Deux clics ou glisser

Comme les outils [rectangle](rectangle.md), l'outil d'ellipse accepte deux gestes de façon
interchangeable : cliquez au premier point, déplacez, puis cliquez au second, ou appuyez au premier
point, glissez et relâchez au second. Un clic rapide sans mouvement arme simplement l'outil et
attend le second point ; les clics accidentels ne laissent donc jamais de géométrie dégénérée.
Pendant qu'un aperçu est actif, la barre d'état affiche les touches modificatrices disponibles, et
`Esc` annule l'aperçu.
