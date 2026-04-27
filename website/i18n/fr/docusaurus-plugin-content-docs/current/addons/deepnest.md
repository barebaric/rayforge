# Deepnest

Deepnest organise automatiquement vos workpieces dans une disposition compacte
sur votre matériau de stock ou la zone de travail de la machine. Il utilise un
algorithme génétique pour trouver un rangement efficace des formes, minimisant
les pertes et plaçant davantage de pièces sur chaque feuille.

![Dialogue des paramètres Deepnest](/screenshots/addon-deepnest.png)

## Prérequis

Sélectionnez un ou plusieurs workpieces sur le canevas avant de lancer le
nesting. Vous pouvez également sélectionner des éléments de stock pour définir
les limites de la feuille. Si aucun stock n'est sélectionné, le module
utilise le stock du document ou se rabat sur la zone de travail de la machine.

## Exécuter la mise en page par nesting

Déclenchez la mise en page par nesting depuis le menu **Disposition**, le
bouton de la barre d'outils ou le raccourci clavier **Ctrl+Alt+N**. Un
dialogue de paramètres s'ouvre avant que l'algorithme ne s'exécute.

## Paramètres de nesting

Le dialogue de paramètres offre les options suivantes avant le début de
l'algorithme de nesting.

**Espacement** définit la distance entre les formes imbriquées, en
millimètres. La valeur par défaut est issue de la taille du spot laser de
votre machine. Augmentez cette valeur pour ajouter une marge de sécurité
entre les pièces.

**Contraindre la rotation** conserve toutes les pièces dans leur orientation
d'origine. Lorsque cette option est désactivée, l'algorithme fait pivoter les
pièces par incréments de 10 degrés pour trouver un ajustement plus serré.
Laisser la rotation libre offre une meilleure utilisation du matériau mais
prend plus de temps à calculer.

**Autoriser le retournement horizontal** met en miroir les pièces
horizontalement pendant le nesting. Cela peut aider à ajuster les pièces plus
étroitement, mais les découpes résultantes seront en miroir.

**Autoriser le retournement vertical** met en miroir les pièces verticalement
pendant le nesting. La même considération concernant la sortie en miroir
s'applique.

Cliquez sur **Démarrer le nesting** pour commencer. Le dialogue se ferme et
l'algorithme s'exécute en arrière-plan. Un indicateur de progression apparaît
dans le panneau inférieur pendant le déroulement du nesting.

## Après le nesting

Lorsque l'algorithme est terminé, tous les workpieces sur le canevas sont
repositionnés à leurs emplacements imbriqués. Les positions sont appliquées
comme une seule action annulable, vous pouvez donc annuler la mise en page en
une seule étape si le résultat ne vous convient pas.

Si l'algorithme n'a pas pu placer tous les workpieces sur le stock disponible,
les éléments non placés sont déplacés à droite de la zone de stock afin de
rester visibles et faciles à identifier.

Si le résultat du nesting est moins bon que la disposition d'origine — par
exemple, si les pièces étaient déjà bien ajustées — les workpieces restent
dans leurs positions d'origine.

## Sujets associés

- [Gestion du stock](../features/stock-handling) - Définir le matériau de stock pour le nesting
- [Positionnement des workpieces](../features/workpiece-positioning) - Positionner les workpieces manuellement
