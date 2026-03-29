# Fusionner les lignes

Lorsque vous importez un dessin contenant des chemins qui se chevauchent, le
laser peut couper plusieurs fois la même ligne. Cela fait perdre du temps,
peut provoquer une carbonisation excessive et élargir le trait de coupe au-delà
de ce qui était prévu.

Le post-processeur **Fusionner les lignes** détecte les segments de chemin
qui se chevauchent ou coïncident, puis les fusionne en un seul passage. Le
laser ne suit chaque ligne unique qu'une seule fois.

## Quand l'utiliser

Ceci se produit le plus souvent lorsque :

- Vous importez un SVG ou DXF dont les formes partagent des arêtes (par exemple
  un motif en grille ou une tessellation)
- Vous combinez plusieurs pièces dont les contours se chevauchent
- Votre logiciel de conception exporte des chemins en double

## Quand ne pas l'utiliser

Si les coupes qui se chevauchent sont intentionnelles — par exemple, pour
effectuer plusieurs passages sur la même ligne afin de couper un matériau plus
épais — laissez Fusionner les lignes désactivé. Dans ce cas, vous pouvez
utiliser la fonction [Passes multiples](multi-pass) à la place, qui vous donne
un contrôle explicite sur le nombre de passages.

## Pages associées

- [Optimisation des chemins](path-optimization) - Réduction des déplacements
  inutiles
- [Passes multiples](multi-pass) - Passages intentionnels multiples sur le même
  chemin
- [Coupe de contour](operations/contour) - L'opération de coupe principale
