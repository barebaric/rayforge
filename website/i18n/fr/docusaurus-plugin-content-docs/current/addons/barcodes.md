---
description:
  "Gravez des QR codes, des codes-barres Code 128 et EAN-13 dans Rayforge. Le contenu peut être un
  modèle, pour que chaque pièce porte son propre code."
---

# Codes-barres

L'extension Codes-barres ajoute des QR codes et des codes-barres en tant que pièces. Ils sont
dessinés sous forme de contours vectoriels avec un remplissage : une opération de Gravure les grave
proprement et une opération de Contour peut quand même les suivre.

## Ajouter un code-barres

Choisissez **Objet - Ajouter un code-barres**. Un QR code apparaît au centre de la zone de travail,
déjà sélectionné. Modifiez-le dans le groupe **Code-barres** du panneau des propriétés :

- **Contenu** — le texte à encoder. Appuyez sur `Entrée` ou sur le bouton d'application pour mettre
  à jour le code.
- **Type** — QR code, Code 128 ou EAN-13.
- **Correction d'erreurs** (QR) — la part du code qui peut être endommagée tout en restant lisible :
  Faible, Moyenne, Quartile ou Élevée. Les niveaux plus élevés agrandissent le code.
- **Taille de module** — la largeur de la barre la plus fine, ou d'un carré du QR code. La taille de
  module et la hauteur des barres s'affichent dans l'unité de longueur que vous avez choisie. La
  pièce prend la taille du code : c'est donc ce réglage qui la dimensionne.
- **Hauteur des barres** (Code 128, EAN-13) — la hauteur des barres.
- **Afficher le texte** (Code 128, EAN-13) — imprime le contenu sous les barres. L'EAN-13 utilise sa
  disposition habituelle, avec les barres de garde qui descendent entre les chiffres.
- **Inverser** — grave plutôt les parties claires, pour les matériaux sombres : le code devient
  clair sur une bordure sombre. **Zone de silence** règle cette bordure en modules et ne peut être
  modifiée que lorsque **Inverser** est activé ; par défaut, elle vaut le minimum requis par le
  type.

Laissez une marge libre autour d'un code non inversé : 4 modules pour les QR codes, 10 pour le Code
128 et 9 pour l'EAN-13.

## Types

- **QR code** accepte n'importe quel texte, y compris des liens.
- **Code 128** accepte l'ASCII imprimable. Les suites de chiffres sont automatiquement regroupées
  par paires.
- **EAN-13** prend 12 chiffres et ajoute le chiffre de contrôle. Avec 13 chiffres, le chiffre de
  contrôle est vérifié. Le champ de contenu devient rouge lorsque le contenu ne convient pas au type
  ; survolez-le pour en voir la raison.

## Modèles

Le contenu peut contenir des expressions `{...}` comme une zone de texte, par exemple `{2 * 21}`.
Avec [Texte variable](../features/variable-text.md), `SN-{serial:05d}` donne à chaque pièce son
propre numéro de série, et les copies d'un réseau continuent de compter.
