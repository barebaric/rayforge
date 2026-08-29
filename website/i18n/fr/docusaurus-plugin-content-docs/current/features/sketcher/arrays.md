---
description: "Créez des tableaux circulaires et des tableaux le long d'une courbe dans l'esquisseur paramétrique de Rayforge."
---

# Tableaux

L'esquisseur propose deux outils de tableau pour créer des tableaux
paramétriques : **Tableau circulaire** et **Tableau le long d'une courbe**.

## Tableaux circulaires

L'outil **Tableau circulaire** (`G+Y`) crée un motif polaire paramétrique
à partir de la sélection actuelle :

1. Sélectionnez les entités que vous souhaitez mettre en motif.
2. Activez l'outil depuis la barre d'outils, le menu **Esquisse → Tableaux**,
   ou `G+Y`.
3. Un cercle guide apparaît sur le canevas et une boîte de dialogue non
   modale s'ouvre avec un aperçu en direct.
4. Définissez le **nombre** et l'**angle total**. Les copies sont générées
   paramétriquement autour du centre du cercle guide.
5. Glissez le centre du cercle guide pour repositionner le tableau, ou
   glissez l'entité d'origine pour modifier le rayon — les champs de la
   boîte de dialogue se mettent à jour en direct.
6. Le **dimension du rayon** du cercle guide redimensionne tout le tableau.
   **Double-cliquez** sur le cercle guide pour rouvrir la boîte de dialogue
   d'édition et régénérer les membres manquants ou redistributionner.

Les copies sont une géométrie statique cuite sans contraintes de
solveur : elles sont régénérées à partir du modèle lorsque le tableau
est modifié. La suppression d'un membre ne supprime que la géométrie de
ce membre et ne redistribue jamais les survivants.

## Tableau le long d'une courbe

L'outil **Tableau le long d'une courbe** distribue des copies d'une ou
plusieurs entités le long d'un chemin guide (une ligne, un arc ou une courbe
de Bézier). Les copies sont placées directement sur le chemin et suivent
sa tangente à chaque position.

### Créer un tableau le long d'une courbe

1. Dessinez la forme que vous souhaitez distribuer (le modèle) et le chemin
   guide que vous souhaitez suivre.
2. Sélectionnez les deux : cliquez d'abord sur le **chemin guide**, puis
   faites Maj-clic sur les **entités du modèle**.
3. Activez l'outil depuis la barre d'outils, le menu **Esquisse → Tableaux**,
   ou `G+W`.
4. Une boîte de dialogue non modale s'ouvre avec un aperçu en direct des
   copies distribuées le long du chemin.
5. Ajustez le **nombre** (total des membres incluant le modèle au début du
   chemin) ou définissez une valeur d'**espacement** pour dériver le nombre
   automatiquement de la longueur du chemin.
6. Activez optionnellement **Aligner sur la tangente** pour que chaque copie
   s'oriente selon la direction du chemin à sa position.
7. Utilisez **Décalage depuis le début** pour ignorer une section initiale
   du chemin avant de placer la première copie.

### Modifier un tableau le long d'une courbe

- **Double-cliquez** sur le chemin guide (ou cliquez sur **Modifier** dans la
  barre d'outils) pour rouvrir la boîte de dialogue et modifier le nombre,
  l'espacement, le décalage ou les paramètres d'alignement.
- **Glissez** une extrémité du chemin guide pour le remodeler. Lorsque vous
  relâchez, toutes les copies sont automatiquement redistribuées le long de
  la nouvelle géométrie du chemin — y compris les mises à jour de rotation
  lorsque *Aligner sur la tangente* est activé.
- La forme du modèle peut être modifiée comme n'importe quelle autre
  géométrie d'esquisse ; les changements se propagent à toutes les copies
  lors de la prochaine mise à jour.

### Fonctionnement

Les copies sont une géométrie statique cuite — elles ne sont pas liées au
modèle par des contraintes de solveur. Lorsque le chemin guide est modifié,
`sync_arrays` détecte la modification et régénère toutes les copies
à partir de la géométrie actuelle du chemin. Cela maintient les mises à
jour rapides et évite la surcharge du solveur.

Le modèle (emplacement 0) est placé au début du chemin. Sa position et
son orientation se mettent à jour automatiquement lorsque le chemin est
modifié. Les entités du modèle d'origine sont supprimées lors de la
création du tableau ; l'annulation les restaure.
