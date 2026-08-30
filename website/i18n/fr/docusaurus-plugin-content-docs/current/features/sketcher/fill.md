---
description:
  "Remplissez des régions fermées d'esquisse avec des aplats de couleur ou des remplissages dégradés
  dans l'esquisseur Rayforge."
---

# Remplissage de zones

L'outil de remplissage (`G+F`) remplit les régions fermées d'une esquisse avec une zone solide. Les
remplissages sont utiles pour les régions qui seront gravées en une seule pièce.

![Un rectangle rempli](/screenshots/addons-sketcher-tool-fill.webp)

## Créer et supprimer des remplissages

1. Dessinez un ou plusieurs contours fermés (par exemple avec les outils [rectangle](rectangle.md)
   ou [tracé](path.md)).
2. Choisissez l'outil de remplissage dans le menu radial, le menu **Esquisse**, ou appuyez sur
   `G+F`.
3. Cliquez n'importe où dans une région fermée pour la remplir.
4. Cliquez à nouveau sur une région remplie pour supprimer son remplissage.

Un clic dans une zone de texte bascule le remplissage des glyphes du texte au lieu de créer un
remplissage de région.

## Couleur de remplissage

La couleur de remplissage des nouveaux remplissages se choisit avec le bouton **Couleur de
remplissage** de la barre d'outils de l'esquisseur. Les remplissages existants conservent leur
couleur jusqu'à ce qu'ils soient supprimés puis recréés.

Comme tout dans l'esquisseur, un remplissage est lié à son contour : redimensionnez la géométrie
environnante et le remplissage suit.
