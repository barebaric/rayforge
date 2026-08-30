---
description:
  "Tracez des lignes droites et des courbes de Bézier lisses avec l'outil de tracé dans l'esquisseur
  Rayforge."
---

# Outil de tracé

L'outil de tracé (`G+P` ou `G+L`) dessine des chaînes connectées de lignes droites et de courbes de
Bézier lisses en un flux de travail unifié. C'est l'outil de dessin le plus polyvalent de
l'esquisseur : cliquez pour placer des points, glissez pour cintrer le segment en courbe.

![Un tracé de deux lignes jointes par un segment de Bézier, avec ses points de passage et ses poignées](/screenshots/addons-sketcher-tool-path.webp)

## Dessiner des tracés

1. Sélectionnez l'outil de tracé dans le menu radial, le menu **Esquisse**, ou avec `G+P`.
2. Cliquez pour placer le premier point. Un aperçu en direct suit le curseur.
3. Cliquez à nouveau sans glisser pour terminer un segment droit — le segment suivant part
   immédiatement de ce point.
4. Appuyez sur un point et glissez avant de relâcher pour transformer le segment en courbe de
   Bézier. Le glissement contrôle la « flèche » de la courbe.
5. Continuez à ajouter des points pour construire votre tracé.
6. Appuyez sur `Escape` ou double-cliquez pour terminer le tracé.

Pendant qu'un aperçu est actif, la barre d'état liste les touches modificatrices applicables, et
`Esc` l'annule.

## Travailler avec les courbes de Bézier

Les courbes de Bézier créent des formes lisses et organiques :

- **Ajuster les poignées** : sélectionnez une Bézier et glissez les extrémités des poignées rondes
  pour modifier la forme de la courbe. Chaque poignée cintre la courbe de son côté du point de
  passage.
- **Se connecter aux points existants** : pendant le dessin, l'accrochage magnétique attache les
  nouveaux segments aux points existants de votre esquisse, et la contrainte correspondante est
  créée automatiquement.

### Types de points de passage

Le point où deux segments d'un tracé se rencontrent est un _point de passage_. Le type de point de
passage contrôle la façon dont la courbe le traverse :

- **Vif** : les poignées des deux côtés sont indépendantes, ce qui produit un angle.
- **Lisse** : les poignées partagent une tangente, ce qui produit une transition continue et
  arrondie.
- **Symétrique** : comme Lisse, mais les poignées sont aussi en miroir, de sorte que les deux côtés
  cintrent également.

Pour changer le type d'un point de passage, faites un clic droit dessus (ou sur le segment de Bézier
adjacent) et choisissez le type dans le menu radial. Les points de passage de Bézier nouvellement
dessinés sont symétriques.

![Le menu radial sur un point de passage de Bézier sélectionné, avec les outils Aplatissement, Vif, Lisse et Symétrique](/screenshots/addons-sketcher-tool-path-pie-menu.webp)

### Convertir des courbes en lignes

L'outil d'**aplatissement** du même menu radial reconvertit les courbes de Bézier en lignes droites,
ce qui est utile lorsque vous avez besoin d'une géométrie propre et simple. Sélectionnez les
segments de Bézier que vous souhaitez convertir et appliquez l'action d'aplatissement. Les segments
s'effondrent vers la liaison droite entre leurs extrémités.

## Contraintes automatiques

L'outil de tracé participe à l'accrochage magnétique comme tout autre outil de dessin. Quand les
guides d'accrochage montrent un alignement pendant le dessin, les contraintes horizontales et
verticales correspondantes sont créées automatiquement, ce qui garde votre esquisse ordonnée dès le
départ plutôt que de corriger les choses par la suite. Maintenez `Shift` pour contraindre le nouveau
segment à l'axe le plus proche. Voir [Grille et accrochage](index.md#grid-and-snapping) pour la
liste complète des indicateurs d'accrochage.
