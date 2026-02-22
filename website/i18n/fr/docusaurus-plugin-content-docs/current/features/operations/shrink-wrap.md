# Shrink Wrap

Shrink Wrap crée un parcours de coupe efficace autour de plusieurs objets en générant une limite qui se "rétracte" autour d'eux. C'est utile pour couper plusieurs pièces d'une feuille avec un minimum de gaspillage.

## Aperçu

Les opérations Shrink Wrap :

- Créent des parcours de limite autour de groupes d'objets
- Minimisent le gaspillage de matériau
- Réduisent le temps de coupe en combinant les parcours
- Supportent des distances de décalage pour le dégagement
- Fonctionnent avec toute combinaison de formes vectorielles

## Quand Utiliser Shrink Wrap

Utilisez shrink wrap pour :

- Couper plusieurs petites pièces d'une feuille
- Minimiser le gaspillage de matériau
- Créer des limites d'imbrication efficaces
- Séparer des groupes de pièces
- Réduire le temps de coupe total

**N'utilisez pas shrink wrap pour :**

- Objets uniques (utilisez [Contour](contour) à la place)
- Pièces nécessitant des limites individuelles
- Coupes rectangulaires précises

## Comment Fonctionne Shrink Wrap

Shrink wrap crée une limite en utilisant un algorithme de géométrie computationnelle :

1. **Démarrez** avec une enveloppe convexe autour de tous les objets
2. **Rétractez** la limite vers l'intérieur vers les objets
3. **Enveloppez** étroitement autour du groupe d'objets
4. **Décalez** vers l'extérieur de la distance spécifiée

Le résultat est un parcours de coupe efficace qui suit la forme globale de vos pièces tout en maintenant le dégagement.

## Créer une Opération Shrink Wrap

### Étape 1 : Organiser les Objets

1. Placez toutes les pièces que vous voulez envelopper sur le canevas
2. Positionnez-les avec l'espacement souhaité
3. Plusieurs groupes séparés peuvent être shrink-wrap ensemble

### Étape 2 : Sélectionner les Objets

1. Sélectionnez tous les objets à inclure dans le shrink wrap
2. Peuvent être différentes formes, tailles et types
3. Tous les objets sélectionnés seront enveloppés ensemble

### Étape 3 : Ajouter une Opération Shrink Wrap

- **Menu :** Opérations Ajouter Shrink Wrap
- **Clic droit :** Menu contextuel Ajouter Opération Shrink Wrap

### Étape 4 : Configurer les Paramètres

![Paramètres d'étape shrink wrap](/screenshots/step-settings-shrink-wrap-general.png)

## Paramètres Clés

### Puissance & Vitesse

Comme les autres opérations de coupe :

**Puissance (%) :**

- Intensité laser pour la coupe
- Identique à ce que vous utiliseriez pour la coupe [Contour](contour)

**Vitesse (mm/min) :**

- À quelle vitesse le laser se déplace
- Correspondez à la vitesse de coupe de votre matériau

**Passes :**

- Nombre de fois pour couper la limite
- Généralement 1-2 passes
- Identique à la coupe de contour pour votre matériau

### Distance de Décalage

**Décalage (mm) :**

- Combien de dégagement autour des pièces
- Distance des objets à la limite shrink-wrap
- Décalage plus grand = plus de matériau laissé autour des pièces

**Valeurs typiques :**

- **2-3mm :** Enveloppe serrée, gaspillage minimal
- **5mm :** Dégagement confortable
- **10mm+ :** Matériau supplémentaire pour la manipulation

**Pourquoi le décalage est important :**

- Trop petit : Risque de couper dans les pièces
- Trop grand : Gaspatillage de matériau
- Considérez : Largeur de kerf, précision de coupe

### Douceur

Contrôle à quel point la limite suit les formes des objets :

**Douceur élevée :**

- Suit les objets plus étroitement
- Parcours plus complexe
- Temps de coupe plus long
- Moins de gaspillage de matériau

**Douceur basse :**

- Parcours plus simple, plus arrondi
- Temps de coupe plus court
- Légèrement plus de gaspillage de matériau

**Recommandé :** Douceur moyenne pour la plupart des cas

## Cas d'Utilisation

### Production de Pièces par Lot

**Scénario :** Couper 20 petites pièces d'une grande feuille

**Sans shrink wrap :**

- Couper la limite de la feuille complète
- Gaspiller tout le matériau autour des pièces
- Temps de coupe long

**Avec shrink wrap :**

- Couper une limite serrée autour du groupe de pièces
- Sauvegarder le matériau pour d'autres projets
- Coupe plus rapide (périmètre plus court)

### Optimisation de l'Imbrication

**Flux de travail :**

1. Imbriquez les pièces efficacement sur la feuille
2. Groupez les pièces en sections
3. Shrink wrap chaque section
4. Coupez les sections séparément

**Avantages :**

- Peut retirer les sections finies pendant la continuation
- Manipulation plus facile des pièces coupées
- Risque réduit de mouvement des pièces

### Conservation du Matériau

**Exemple :** Petites pièces sur matériau coûteux

**Processus :**

1. Organisez les pièces de manière serrée
2. Shrink wrap avec décalage de 3mm
3. Coupez libre de la feuille
4. Sauvegardez le matériau restant

**Résultat :** Efficacité matériau maximum

## Combiner avec d'Autres Opérations

### Shrink Wrap + Contour

Flux de travail courant :

1. Opérations de **Contour** sur les pièces individuelles (couper les détails)
2. **Shrink wrap** autour du groupe (couper libre de la feuille)

**Ordre d'exécution :**

- D'abord : Couper les détails dans les pièces (pendant sécurisé)
- Dernier : Shrink wrap coupe le groupe libre

Voir [Flux de Travail Multi-Couches](../multi-layer) pour plus de détails.

### Shrink Wrap + Raster

**Exemple :** Pièces gravées et coupées

1. **Raster** gravure des logos sur les pièces
2. **Contour** coupe des contours des pièces
3. **Shrink wrap** autour du groupe entier

**Avantages :**

- Toute la gravure se produit pendant que le matériau est sécurisé
- Le shrink wrap final coupe tout le lot libre

## Conseils & Meilleures Pratiques

![Paramètres de post-traitement shrink wrap](/screenshots/step-settings-shrink-wrap-post.png)

### Espacement des Pièces

**Espacement optimal :**

- 5-10mm entre les pièces
- Suffisant pour que shrink wrap distingue les objets séparés
- Pas trop pour gaspiller du matériau

**Trop proche :**

- Les pièces peuvent être enveloppées ensemble
- Shrink wrap peut combler les écarts
- Difficile à séparer après la coupe

**Trop loin :**

- Gaspatillage de matériau
- Temps de coupe plus long
- Utilisation inefficace de la feuille

### Considérations de Matériau

**Idéal pour :**

- Production en série (plusieurs pièces identiques)
- Petites pièces de grandes feuilles
- Matériaux coûteux (minimiser le gaspillage)
- Travaux de coupe par lot

**Non idéal pour :**

- Pièces uniques grandes
- Pièces remplissant toute la feuille
- Quand vous avez besoin de la coupe de feuille complète

### Sécurité

**Toujours :**

- Vérifiez que la limite ne chevauche pas les pièces
- Vérifiez que le décalage est suffisant
- Prévisualisez en [Mode Simulation](../simulation-mode)
- Testez sur du rebut d'abord

**Surveillez :**

- Shrink wrap coupant dans les pièces (augmentez le décalage)
- Pièces bougeant avant la fin du shrink wrap
- Matériau gondolant tirant les pièces hors de position

## Techniques Avancées

### Shrink Wraps Multiples

Créez des limites séparées pour différents groupes :

**Processus :**

1. Organisez les pièces en groupes logiques
2. Shrink wrap Groupe 1 (pièces du haut)
3. Shrink wrap Groupe 2 (pièces du bas)
4. Coupez les groupes séparément

**Avantages :**

- Retirez les groupes finis pendant le travail
- Meilleure organisation
- Récupération des pièces plus facile

### Shrink Wraps Imbriqués

Shrink wrap à l'intérieur d'une limite plus grande :

**Exemple :**

1. Shrink wrap intérieur : Petites pièces détaillées
2. Shrink wrap extérieur : Inclut les pièces plus grandes
3. Contour : Limite de feuille complète

**Utilisation pour :** Dispositions multi-pièces complexes

### Test de Dégagement

Avant la production :

1. Créez un shrink wrap
2. Prévisualisez avec [Mode Simulation](../simulation-mode)
3. Vérifiez que le dégagement est adéquat
4. Vérifiez qu'aucune pièce n'est intersectée
5. Exécutez le test sur matériau de rebut

## Dépannage

### Shrink wrap coupe dans les pièces

- **Augmentez :** La distance de décalage
- **Vérifiez :** Les pièces ne sont pas trop proches les unes des autres
- **Vérifiez :** Le parcours shrink wrap dans l'aperçu
- **Tenez compte de :** La largeur du kerf (largeur du faisceau laser)

### La limite ne suit pas les formes

- **Augmentez :** Le paramètre de douceur
- **Vérifiez :** Les pièces sont correctement sélectionnées
- **Essayez :** Un décalage plus petit (pourrait envelopper trop loin)

### Les pièces sont enveloppées ensemble

- **Augmentez :** L'espacement entre les pièces
- **Ajoutez :** Des contours manuels autour des pièces individuelles
- **Divisez :** En plusieurs opérations shrink wrap

### La coupe prend trop de temps

- **Diminuez :** La douceur (parcours plus simple)
- **Augmentez :** Le décalage (limites plus droites)
- **Considérez :** Plusieurs shrink wraps plus petits

### Les pièces bougent pendant la coupe

- **Ajoutez :** De petits ponts pour maintenir les pièces (voir [Ponts de Maintien](../holding-tabs))
- **Utilisez :** Ordre de coupe : de l'intérieur vers l'extérieur
- **Assurez-vous :** Le matériau est plat et sécurisé
- **Vérifiez :** La feuille n'est pas gondolée

## Détails Techniques

### Algorithme

Shrink wrap utilise la géométrie computationnelle :

1. **Enveloppe convexe** - Trouver la limite extérieure
2. **Forme alpha** - Rétracter vers les objets
3. **Décalage** - Étendre par la distance de décalage
4. **Simplifier** - Selon le paramètre de douceur

### Optimisation du Parcours

Le parcours de limite est optimisé pour :

- Longueur totale minimum
- Courbes lisses (selon la douceur)
- Points de début/fin efficaces

### Système de Coordonnées

- **Unités :** Millimètres (mm)
- **Précision :** 0.01mm typique
- **Coordonnées :** Identiques à l'espace de travail

## Sujets Connexes

- **[Coupe de Contour](contour)** - Couper les contours d'objets individuels
- **[Flux de Travail Multi-Couches](../multi-layer)** - Combiner efficacement les opérations
- **[Ponts de Maintien](../holding-tabs)** - Maintenir les pièces sécurisées pendant la coupe
- **[Mode Simulation](../simulation-mode)** - Prévisualiser les parcours de coupe
- **[Grille de Test de Matériau](material-test-grid)** - Trouver les paramètres de coupe optimaux
