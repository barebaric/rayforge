---
description:
  "Agrandissez, réduisez ou transformez en lumière des contours avec l'outil de décalage dans
  l'esquisseur Rayforge."
---

# Décalage de contour

L'outil de décalage (`O+F`) agrandit ou réduit le contour sélectionné d'une distance donnée, ou
transforme un tracé ouvert en une lumière (slot). Sélectionnez les entités formant un contour (ou
utilisez le double-clic pour sélectionner la géométrie connectée), puis appuyez sur `O+F` ou
utilisez l'entrée **Décalage** du menu radial.

![Boîte de dialogue de décalage de contour](/screenshots/addons-sketcher-offset-dialog.webp)

La boîte de dialogue demande la distance de décalage et affiche un aperçu en direct du résultat sur
le canevas pendant la saisie :

- Les **contours fermés** s'agrandissent avec une distance positive et se réduisent avec une
  distance négative. Un décalage qui ferait s'effondrer le contour est refusé.
- Les **tracés ouverts** deviennent un contour fermé en forme de lumière de la largeur indiquée,
  avec des extrémités arrondies.

![Contour de Bézier](/screenshots/addons-sketcher-offset-before.webp)
![Bézier décalé en une lumière](/screenshots/addons-sketcher-offset-after.webp)

Le décalage remplace le contour sélectionné par le résultat :

- Les cercles, arcs et ellipses isolés conservent leur type d'entité et sont mis à jour sur place :
  ils restent modifiables et contraints comme avant.
- Les chaînes de segments connectés (y compris les Béziers) sont remplacées par une entité polygone.
  Le polygone s'édite comme un tout : faites glisser son point central pour le déplacer et son point
  de poignée pour le faire pivoter ou le redimensionner uniformément.

Si la sélection contient plusieurs contours déconnectés, chacun est décalé indépendamment en une
seule étape.
