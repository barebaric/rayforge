# Cadrer Votre Travail

Apprenez à utiliser la fonction de cadrage pour prévisualiser les limites de votre travail laser et assurer un alignement correct avant la coupe.

## Aperçu

Le cadrage vous permet de prévisualiser les limites exactes de votre travail laser en traçant un contour avec le laser à faible puissance ou avec le laser éteint. Cela aide à vérifier le positionnement et à prévenir les erreurs coûteuses.

## Quand Utiliser le Cadrage

- **Configurations initiales** : Vérifier le placement du matériau
- **Positionnement précis** : S'assurer que le design tient dans les limites du matériau
- **Travaux multiples** : Confirmer l'alignement avant chaque exécution
- **Matériaux coûteux** : Double-vérification avant de s'engager dans les coupes

## Comment Cadrer

### Méthode 1 : Contour Uniquement

Tracez la limite du travail sans allumer le laser :

1. **Chargez votre design** dans Rayforge
2. **Positionnez le matériau** sur le lit laser
3. **Cliquez sur le bouton Cadrer** dans la barre d'outils
4. **Regardez la tête laser** tracer le rectangle de limite
5. **Vérifiez le positionnement** et ajustez le matériau si nécessaire

### Méthode 2 : Aperçu à Faible Puissance

Certaines machines supportent le cadrage à faible puissance avec un faisceau visible :

1. **Activez le mode faible puissance** dans les paramètres machine
2. **Définissez la puissance de cadrage** (typiquement 1-5%)
3. **Exécutez l'opération de cadrage**
4. **Observez le contour** tracé sur la surface du matériau

:::warning Vérifiez Votre Machine
Tous les lasers ne supportent pas le cadrage à faible puissance de manière sécurisée. Consultez la documentation de votre machine avant d'utiliser cette fonctionnalité.
:::


## Paramètres de Cadrage

Configurez le comportement de cadrage dans Paramètres → Machine :

- **Vitesse de cadrage** : À quelle vitesse la tête laser se déplace pendant le cadrage
- **Puissance de cadrage** : Puissance laser pendant le cadrage (0 pour éteint, faible % pour trace visible)
- **Pause aux coins** : Brève pause à chaque coin pour la visibilité
- **Nombre de répétitions** : Nombre de fois pour tracer le contour

## Utiliser les Résultats du Cadrage

Après le cadrage, vous pouvez :

- **Ajuster la position du matériau** si nécessaire
- **Recadrer** pour vérifier la nouvelle position
- **Procéder au travail** une fois satisfait du placement

## Conseils pour un Cadrage Efficace

- **Marquez les coins** : Placez de petits morceaux de ruban adhésif aux coins pour référence
- **Vérifiez le dégagement** : Assurez un espace adéquat autour de votre design
- **Vérifiez l'orientation** : Confirmez que le matériau est orienté correctement
- **Tenez compte du kerf** : Rappelez-vous que les coupes seront légèrement plus larges que les contours

## Cadrage avec Caméra

Si votre machine a un support caméra, vous pouvez :

1. **Capturez l'image caméra** du placement du matériau
2. **Superposez le design** sur la vue caméra
3. **Ajustez la position** virtuellement avant le cadrage
4. **Cadrez pour confirmer** l'alignement physique

Voir [Intégration Caméra](../machine/camera) pour plus de détails.

## Dépannage

**Le cadrage ne correspond pas au design** : Vérifiez l'origine du travail et les paramètres du système de coordonnées

**Le laser tire pendant le cadrage** : Désactivez la puissance de cadrage ou vérifiez les paramètres machine

**Cadrage trop rapide pour voir** : Réduisez la vitesse de cadrage dans les paramètres

**La tête n'atteint pas les coins** : Vérifiez que le design est dans la zone de travail machine

## Notes de Sécurité

- **Ne laissez jamais la machine sans surveillance** pendant le cadrage
- **Vérifiez que le laser est éteint** si vous utilisez le cadrage à puissance zéro
- **Gardez les mains éloignées** du parcours de la tête laser
- **Surveillez les obstructions** qui pourraient interférer avec le mouvement

## Sujets Connexes

- [Intégration Caméra](../machine/camera)
- [Guide de Démarrage Rapide](../getting-started/quick-start)
