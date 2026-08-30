---
description:
  "Placez du texte gravé, des étiquettes et des numéros de série sur une esquisse avec l'outil de
  zone de texte de Rayforge."
---

# Zone de texte

L'outil de zone de texte (`G+T`) place du texte sur l'esquisse comme géométrie modifiable — texte
gravé, étiquettes et numéros de série. Les zones de texte sont entièrement paramétriques : les
glyphes vivent dans un cadre contraint, si bien qu'ils se résolvent à nouveau dès que le cadre est
déplacé ou dimensionné.

![Un logotype et une étiquette de pièce](/screenshots/addons-sketcher-tool-text-box.webp)

## Créer et modifier du texte

1. Sélectionnez l'outil de zone de texte dans le menu radial, le menu **Esquisse**, ou avec `G+T`.
2. Cliquez à l'endroit où le texte doit commencer : une zone de texte apparaît au point du clic et
   l'outil passe directement en mode modification.
3. Saisissez le texte — la zone se redimensionne pour l'adapter au fur et à mesure.
4. Appuyez sur `Entrée` ou `Échap` pour terminer la modification.

Pour modifier une zone de texte existante, cliquez à l'intérieur. Un double-clic sélectionne un mot,
un triple-clic toute la ligne, et le texte peut être sélectionné et remplacé comme dans n'importe
quel éditeur de texte, y compris `Ctrl+C`/`Ctrl+V`, annuler/rétablir et coller en cours de
modification.

## Propriétés de la police

![Le panneau des propriétés de la police](/screenshots/addons-sketcher-tool-text-box-font-properties.webp)

Le panneau **Propriétés de la police** de la barre latérale contrôle l'apparence de la zone de texte
sélectionnée sur le canevas :

- **Famille de police** — choisissez parmi les polices système installées.
- **Taille de la police** — en points.
- Interrupteurs **Gras** et **Italique**.

## Un cadre paramétrique

Une zone de texte n'est pas une image matricielle : ses glyphes sont une véritable géométrie
d'esquisse, disposée dans un cadre défini par une origine et des points de largeur et de hauteur. Le
cadre est dessiné en tiretés comme géométrie de construction, il sert donc de référence de mise en
page et ne se retrouve jamais dans les trajectoires d'outils lors de la fabrication de l'esquisse.
Comme tout le reste dans l'esquisseur, le cadre est contraint, si bien qu'il peut être dimensionné
comme n'importe quelle autre géométrie — modifiez la contrainte de largeur et le texte se résout à
nouveau pour remplir la zone.

Un clic dans une zone de texte avec l'[outil de remplissage](fill.md) bascule le remplissage des
glyphes du texte au lieu de créer un remplissage de région.

## Expressions de modèle

Les zones de texte acceptent les **expressions de modèle** : tout ce qui est entre accolades est
évalué quand l'esquisse se résout, si bien que les étiquettes peuvent afficher des valeurs en direct
comme des dimensions, des dates ou des numéros de série uniques. Voir
[Expressions de modèle dans les zones de texte](expressions.md#template-expressions-in-text-boxes)
pour les détails et les fonctions intégrées.
