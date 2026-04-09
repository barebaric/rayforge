# Approche / Sortie

Les mouvements d'approche et de sortie étendent chaque parcours de contour avec de courts segments sans puissance avant le début de la coupe et après sa fin. Cela donne au laser le temps d'atteindre une vitesse constante avant que la coupe réelle ne commence et de ralentir progressivement après la fin de la coupe, ce qui produit des résultats plus propres aux points de début et de fin de chaque coupe.

## Comment ça Fonctionne

Lorsque l'approche/sortie est activée, Rayforge examine la direction tangente de chaque parcours de contour à ses points de début et de fin. Il insère ensuite un court mouvement rectiligne sans puissance du laser le long de cette tangente avant le premier point de coupe et un autre après le dernier point de coupe. Le laser est éteint pendant ces segments supplémentaires, aucun matériau n'est donc retiré en dehors du parcours prévu.

## Paramètres

### Activer l'Approche/Sortie

Active ou désactive la fonction pour l'opération. Lorsqu'elle est désactivée, la coupe commence et se termine exactement aux points d'extrémité du parcours sans mouvements d'approche ou de sortie supplémentaires.

### Distance Automatique

Lorsque cette option est activée, Rayforge calcule automatiquement la distance d'approche et de sortie en fonction de la vitesse de coupe et du réglage d'accélération de la machine. La formule utilise un facteur de sécurité de deux pour s'assurer que la tête laser a suffisamment d'espace pour atteindre la vitesse maximale. Chaque fois que vous modifiez la vitesse de coupe ou que l'accélération de la machine est mise à jour, la distance est recalculée.

### Distance d'Approche

La longueur du mouvement d'approche sans puissance avant le début de la coupe, en millimètres. La valeur par défaut est 2 mm. Ce champ n'est modifiable que lorsque la distance automatique est désactivée.

### Distance de Sortie

La longueur du mouvement de sortie sans puissance après la fin de la coupe, en millimètres. La valeur par défaut est 2 mm. Ce champ n'est modifiable que lorsque la distance automatique est désactivée.

## Quand Utiliser l'Approche/Sortie

L'approche/sortie est surtout utile lorsque vous remarquez des marques de brûlure, une surbrûlure ou une qualité de coupe incohérente aux points de début et de fin de vos contours. L'approche sans puissance donne à la machine le temps d'accélérer jusqu'à la vitesse de coupe pour que le laser atteigne le matériau à pleine vitesse, et la sortie sans puissance permet un ralentissement en douceur au lieu de rester à pleine puissance sur le dernier point.

Elle est disponible en option de post-traitement sur les opérations de contour, de contour de cadre et de shrink wrap.

---

## Pages Connexes

- [Coupe de Contour](operations/contour) - Opération de coupe principale
- [Contour de Cadre](operations/frame-outline) - Coupe de limite rectangulaire
- [Shrink Wrap](operations/shrink-wrap) - Coupe de limite efficace
- [Ponts de Maintien](holding-tabs) - Maintenir les pièces sécurisées pendant la coupe
