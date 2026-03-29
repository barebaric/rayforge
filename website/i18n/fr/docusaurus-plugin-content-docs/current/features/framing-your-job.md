# Cadrage de votre travail

Apprenez à utiliser la fonction de cadrage pour prévisualiser les limites de
votre travail et assurer un alignement correct avant la découpe.

## Aperçu

Le cadrage vous permet de prévisualiser les limites exactes de votre travail
laser en traçant un contour avec le laser à faible puissance ou laser éteint.
Cela permet de vérifier le positionnement et d'éviter les erreurs coûteuses.

## Quand utiliser le cadrage

- **Premiers réglages** : Vérifier le placement du matériau
- **Positionnement précis** : S'assurer que le design tient dans les limites
  du matériau
- **Travaux multiples** : Confirmer l'alignement avant chaque passage
- **Matériaux coûteux** : Vérifier avant de lancer les coupes

## Comment cadrer

### Méthode 1 : Contour uniquement

Tracer la limite du travail sans allumer le laser :

1. **Chargez votre design** dans Rayforge
2. **Placez le matériau** sur le lit du laser
3. **Cliquez sur le bouton Cadrer** dans la barre d'outils
4. **Observez la tête laser** tracer le rectangle de délimitation
5. **Vérifiez le positionnement** et ajustez le matériau si nécessaire

### Méthode 2 : Aperçu à faible puissance

Certaines machines prennent en charge le cadrage à faible puissance avec un
faisceau visible :

1. **Activez le mode faible puissance** dans les paramètres de la machine
2. **Réglez la puissance de cadrage** (généralement 1 à 5 %)
3. **Lancez l'opération de cadrage**
4. **Observez le contour** tracé sur la surface du matériau

:::warning Vérifiez votre machine
Tous les lasers ne prennent pas en charge le cadrage à faible puissance en
toute sécurité. Consultez la documentation de votre machine avant d'utiliser
cette fonctionnalité.
:::

## Paramètres de cadrage

Configurez le comportement du cadrage dans les paramètres de la tête laser de
votre machine :

- **Vitesse de cadrage** : La vitesse de déplacement de la tête laser pendant
  le cadrage. Elle est définie par tête laser, donc si votre machine possède
  plusieurs lasers, vous pouvez utiliser des vitesses différentes pour chacun.
- **Puissance de cadrage** : Puissance du laser pendant le cadrage (0 pour
  éteint, % faible pour trace visible)
- **Temps de pause aux coins** : Une brève pause à chaque coin du contour.
  Cela vous donne un instant pour voir exactement où se situe chaque coin —
  particulièrement utile à des vitesses de cadrage plus élevées.
- **Nombre de répétitions** : Nombre de fois que le contour est tracé. Une
  valeur supérieure à un rend le trajet plus facile à suivre visuellement.

## Utilisation des résultats du cadrage

Après le cadrage, vous pouvez :

- **Ajuster la position du matériau** si nécessaire
- **Recadrer** pour vérifier la nouvelle position
- **Lancer le travail** une fois satisfait du placement

## Conseils pour un cadrage efficace

- **Marquez les coins** : Placez de petits morceaux de ruban adhésif aux coins
  comme repères
- **Vérifiez l'espace** : Assurez-vous d'un espace suffisant autour du design
- **Confirmez l'orientation** : Vérifiez que le matériau est orienté
  correctement
- **Tenez compte du trait de scie** : Rappelez-vous que les coupes seront
  légèrement plus larges que les contours

## Cadrage avec caméra

Si votre machine prend en charge la caméra, vous pouvez :

1. **Capturer une image** du placement du matériau
2. **Superposer le design** sur la vue de la caméra
3. **Ajuster la position** virtuellement avant le cadrage
4. **Cadrer pour confirmer** l'alignement physique

Voir [Intégration de la caméra](../machine/camera) pour plus de détails.

## Dépannage

**Le cadre ne correspond pas au design** : Vérifiez l'origine du travail et
les paramètres du système de coordonnées

**Le laser tire pendant le cadrage** : Désactivez la puissance de cadrage ou
vérifiez les paramètres de la machine

**Le cadre est trop rapide** : Réduisez la vitesse de cadrage dans les
paramètres

**La tête n'atteint pas les coins** : Vérifiez que le design est dans la zone
de travail de la machine

## Consignes de sécurité

- **Ne laissez jamais la machine sans surveillance** pendant le cadrage
- **Vérifiez que le laser est éteint** lors du cadrage sans puissance
- **Gardez les mains à l'écart** du trajet de la tête laser
- **Surveillez les obstacles** pouvant interférer avec le mouvement

## Sujets connexes

- [Positionnement de la pièce](workpiece-positioning) - Guide complet de
  positionnement
- [Intégration de la caméra](../machine/camera)
- [Guide de démarrage rapide](../getting-started/quick-start)
