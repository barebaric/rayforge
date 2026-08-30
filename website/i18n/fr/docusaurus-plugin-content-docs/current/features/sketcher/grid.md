---
description: "Créez une grille de construction de rangées et de colonnes comme échafaudage de dessin dans l'esquisseur Rayforge."
---

# Grille

L'outil de grille (`G+G`) crée une grille homogène de lignes de
construction — des rangées et colonnes de guides régulièrement espacés
qui servent d'échafaudage de dessin, par exemple pour disposer un motif de
perforations ou aligner des éléments répétés.

![Une grille de construction 4x6](/screenshots/addons-sketcher-tool-grid.webp)

1. Sélectionnez l'outil de grille dans le menu radial, le menu
   **Esquisse**, ou avec `G+G`.
2. Une boîte de dialogue demande le nombre de **rangées** et de
   **colonnes**.
3. Confirmez pour créer la grille à l'origine de l'esquisse avec des
   cellules de 10 mm.

La grille se compose de géométrie de construction : elle est dessinée en
tiretés, sert de référence d'accrochage et d'alignement comme toute autre
géométrie, et est exclue des trajectoires d'outils lors de la fabrication
de l'esquisse (voir [Géométrie de construction](index.md#construction-geometry)).
Chaque ligne peut être déplacée ou supprimée comme n'importe quelle autre
géométrie, et les sélectionner puis basculer le mode construction avec
`G+N` transforme l'échafaudage en vraie géométrie.
