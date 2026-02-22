# Simuler Votre Travail

![Capture d'Écran Mode Simulation](/screenshots/main-simulation.png)

Apprenez à utiliser le mode simulation de Rayforge pour prévisualiser votre travail laser, identifier les problèmes potentiels et estimer le temps d'achèvement avant l'exécution sur le matériel réel.

## Aperçu

Le mode simulation vous permet de visualiser l'exécution de votre travail laser sans réellement faire fonctionner la machine. Cela aide à repérer les erreurs, optimiser les paramètres et planifier votre flux de travail.

## Avantages de la Simulation

- **Prévisualiser l'exécution du travail** : Voir exactement comment le laser se déplacera
- **Estimer le temps** : Obtenir des estimations précises de la durée du travail
- **Identifier les problèmes** : Repérer les chevauchements, les écarts ou le comportement inattendu
- **Optimiser l'ordre des parcours** : Visualiser la séquence de coupe
- **Apprendre le G-code** : Comprendre comment les opérations se traduisent en commandes machine

## Démarrer une Simulation

1. **Chargez ou créez votre design** dans Rayforge
2. **Configurez les opérations** avec les paramètres souhaités
3. **Cliquez sur le bouton Simuler** dans la barre d'outils (ou utilisez le raccourci clavier)
4. **Regardez la simulation** défiler votre travail

## Contrôles de Simulation

### Contrôles de Lecture

- **Lecture/Pause** : Démarrer ou mettre en pause la simulation
- **Avancer/Reculer** : Parcourir le travail une commande à la fois
- **Contrôle de Vitesse** : Ajuster la vitesse de lecture (0.5x à 10x)
- **Aller à la Position** : Passer à un pourcentage spécifique du travail
- **Redémarrer** : Commencer la simulation depuis le début

### Options de Visualisation

- **Afficher le parcours d'outil** : Afficher le parcours que la tête laser suivra
- **Afficher les mouvements de déplacement** : Visualiser les mouvements de positionnement rapide
- **Afficher la puissance laser** : Colorer les parcours par niveau de puissance
- **Mode carte de chaleur** : Visualiser le temps de stationnement et la densité de puissance

### Affichage des Informations

Pendant la simulation, surveillez :

- **Position actuelle** : Coordonnées X, Y de la tête laser
- **Progression du travail** : Pourcentage achevé
- **Temps restant estimé** : Basé sur la progression actuelle
- **Opération actuelle** : Quelle opération est en cours d'exécution
- **Puissance et vitesse** : Paramètres laser actuels

## Interpréter les Résultats de Simulation

### Que Rechercher

- **Efficacité du parcours** : Y a-t-il des mouvements de déplacement inutiles ?
- **Coupes qui se chevauchent** : Double-coupe involontaire des parcours
- **Ordre des opérations** : La séquence a-t-elle du sens ?
- **Distribution de puissance** : La puissance est-elle appliquée de manière cohérente ?
- **Mouvements inattendus** : Tout mouvement saccadé ou étrange

### Visualisation Carte de Chaleur

La carte de chaleur montre l'exposition cumulée au laser :

- **Couleurs froides (bleu/vert)** : Faible exposition
- **Couleurs chaudes (jaune/orange)** : Exposition modérée
- **Couleurs chaudes (rouge)** : Exposition élevée ou temps de stationnement

Utilisez cela pour identifier :

- **Points chauds** : Zones qui peuvent sur-brûler
- **Écarts** : Zones qui peuvent être sous-exposées
- **Problèmes de chevauchement** : Double-exposition involontaire

Voir [Mode Simulation](../features/simulation-mode) pour des informations détaillées.

## Utiliser la Simulation pour l'Optimisation

### Optimiser l'Ordre de Coupe

Si la simulation révèle un ordre de parcours inefficace :

1. **Activez l'optimisation de parcours** dans les paramètres d'opération
2. **Choisissez la méthode d'optimisation** (plus proche voisin, TSP)
3. **Resimulez** pour vérifier l'amélioration

### Ajuster le Timing

La simulation fournit des estimations de temps précises :

- **Temps de travail longs** : Envisagez d'optimiser les parcours ou d'augmenter la vitesse
- **Temps très courts** : Vérifiez que les paramètres sont corrects pour le matériau
- **Durée inattendue** : Vérifiez les opérations cachées ou les doublons

### Vérifier les Travaux Multi-Couches

Pour les projets complexes multi-couches :

1. **Simulez chaque calque** indépendamment
2. **Vérifiez l'ordre des opérations** à travers les calques
3. **Recherchez les conflits** entre les calques
4. **Estimez le temps total** pour le travail complet

## Simulation vs Exécution Réelle

### Différences à Noter

La simulation est très précise mais :

- **Ne prend pas en compte** : Imperfections mécaniques, jeu, vibrations
- **Peut différer légèrement** : Accélération/décélération réelle vs simulée
- **Ne montre pas** : Interaction avec le matériau, fumée, émanations
- **Estimations de temps** : Généralement précises à 5-10% près

### Quand Resimuler

- **Après avoir changé les paramètres** : Puissance, vitesse ou paramètres d'opération
- **Après avoir édité le design** : Tout changement de design
- **Avant les matériaux coûteux** : Double-vérification avant de s'engager
- **Lors du dépannage** : Vérifier les corrections aux problèmes identifiés

## Conseils pour une Simulation Efficace

- **Simulez toujours** avant d'exécuter les travaux importants
- **Utilisez une lecture plus lente** pour repérer les problèmes subtils
- **Activez la carte de chaleur** pour les travaux de gravure
- **Comparez plusieurs paramètres** en simulant des variations
- **Documentez les résultats** : Capture d'écran ou notez les problèmes trouvés

## Dépannage de la Simulation

**La simulation ne démarre pas** : Vérifiez que les opérations sont correctement configurées

**La simulation va trop vite** : Ajustez la vitesse de lecture à un paramètre plus lent

**Impossible de voir les détails** : Zoomez sur des zones spécifiques d'intérêt

**L'estimation de temps semble fausse** : Vérifiez que le profil machine a les vitesses max correctes

## Sujets Connexes

- [Fonctionnalité Mode Simulation](../features/simulation-mode)
- [Flux de Travail Multi-Couches](../features/multi-layer)
