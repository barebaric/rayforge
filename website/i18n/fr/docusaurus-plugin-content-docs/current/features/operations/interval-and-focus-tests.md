---
description:
  "Trouvez le meilleur intervalle de ligne pour la gravure et le point focal de votre laser avec le
  test d'intervalle et le test de mise au point."
---

# Tests d'intervalle et de mise au point

À côté de la [grille de test de matériau](material-test-grid.md), le menu Outils propose deux autres
tâches de calibrage. Les deux sont créées comme un calque ordinaire avec des pièces et des
opérations : vous pouvez le déplacer, le cadrer et le prévisualiser comme tout autre contenu, puis
modifier les réglages de chaque cellule ou ligne dans son opération.

## Test d'intervalle

**Outils → Créer un test d'intervalle** grave une rangée de carrés remplis. Chaque carré a sa propre
opération **Graver** avec son propre intervalle de ligne, réparti régulièrement du plus petit au plus
grand intervalle saisi, tandis que la puissance et la vitesse restent les mêmes pour tous. Les
étiquettes sous chaque carré indiquent l'intervalle en millimètres et la densité de lignes
correspondante en lignes par pouce (LPI).

Choisissez le carré rempli de façon homogène, sans lignes visibles et sans brûlure trop profonde, et
utilisez son intervalle pour les gravures sur ce matériau. Les étiquettes sont découpées avant les
carrés avec une opération séparée à faible puissance.

## Test de mise au point

**Outils → Créer un test de mise au point** trouve la hauteur à laquelle le faisceau est le plus
net. Un décalage positif signifie plus de distance entre la tête et le matériau. La ligne la plus
fine indique la meilleure mise au point.

| Méthode                                | Comment la hauteur change                                                                       |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Pas de l'axe Z**                     | La tête rejoint chaque décalage par des mouvements Z relatifs et revient à la hauteur de départ |
| **Manuel (pause entre les lignes)**    | La tâche se met en pause (`M0`) avant chaque ligne ; vous déplacez la tête et appuyez Reprendre |
| **Rampe (matériau incliné)**           | Une longue ligne graduée ; vous surélevez une extrémité d'une bande plate                       |

Les pas de l'axe Z ne sont proposés que sur les machines dotées d'un axe Z, et les pas de l'axe Z
comme la méthode manuelle nécessitent un contrôleur G-code, car ils utilisent des opérations
[Commande](command.md) entre les lignes. Les décalages sont limités à ±10 mm de la hauteur de départ.

Avec la méthode manuelle, faites d'abord la mise au point du laser comme d'habitude. Les étiquettes
sont gravées à cette hauteur. À la première pause, placez la tête au premier décalage ; à chaque
pause suivante, déplacez-la d'un pas. Vérifiez que votre contrôleur s'arrête sur `M0` et que le
bouton Reprendre poursuit la tâche avant de vous y fier, par exemple avec un essai à vide à 0 % de
puissance.

Pour la rampe, la hauteur sous chaque point de la ligne découle de la pente de la bande : à une
distance _d_ le long d'une ligne de longueur _L_ sur une bande qui monte de _h_, le matériau est
_h_ × _d_ / _L_ plus haut qu'au départ.
