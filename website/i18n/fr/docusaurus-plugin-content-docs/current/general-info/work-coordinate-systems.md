# Systèmes de Coordonnées de Travail (WCS)

Les Systèmes de Coordonnées de Travail (WCS) vous permettent de définir plusieurs points de référence sur la zone de travail de votre machine. Cela facilite l'exécution du même travail à différentes positions sans refaire le design ou repositionner vos pièces.

## Comprendre le WCS

Considérez le WCS comme des "points zéro" personnalisables pour votre travail. Alors que votre machine a une position d'origine fixe (déterminée par les contacteurs de fin de course), le WCS vous permet de définir où vous voulez que votre travail commence.

### Pourquoi Utiliser le WCS ?

- **Montages multiples** : Configurez plusieurs zones de travail sur votre lit et basculez entre elles
- **Positionnement répétable** : Exécutez le même travail à différents emplacements
- **Alignement rapide** : Définissez un point de référence basé sur votre matériau ou pièce
- **Flux de production** : Organisez plusieurs travaux à travers votre zone de travail

## Types de WCS

Rayforge supporte les systèmes de coordonnées suivants :

| Système | Type | Description |
| ------- | ---- | ----------- |
| **G53** | Machine | Coordonnées machine absolues (fixes, ne peut pas être changé) |
| **G54** | Travail 1 | Premier système de coordonnées de travail (défaut) |
| **G55** | Travail 2 | Deuxième système de coordonnées de travail |
| **G56** | Travail 3 | Troisième système de coordonnées de travail |
| **G57** | Travail 4 | Quatrième système de coordonnées de travail |
| **G58** | Travail 5 | Cinquième système de coordonnées de travail |
| **G59** | Travail 6 | Sixième système de coordonnées de travail |

### Coordonnées Machine (G53)

G53 représente la position absolue de votre machine, avec zéro à la position d'origine de la machine. C'est fixé par votre matériel et ne peut pas être changé.

**Quand utiliser :**

- Homing et calibration
- Positionnement absolu relatif aux limites machine
- Quand vous avez besoin de référencer la position physique de la machine

### Coordonnées de Travail (G54-G59)

Ce sont des systèmes de coordonnées décalés que vous pouvez définir. Chacun a son propre point zéro que vous pouvez définir n'importe où sur votre zone de travail.

**Quand utiliser :**

- Configurer plusieurs montages de travail
- Aligner sur des positions de matériau
- Exécuter le même travail à différents emplacements

## Visualiser le WCS dans l'Interface

### Canevas 2D

Le canevas 2D affiche votre origine WCS avec un marqueur vert :

- **Lignes vertes** : Indiquent la position de l'origine WCS actuelle (0, 0)
- **Alignement de grille** : Les lignes de grille sont alignées sur l'origine WCS, pas l'origine machine

Le marqueur d'origine se déplace quand vous changez le WCS actif ou son décalage, vous montrant exactement où votre travail commencera.

### Aperçu 3D

Dans l'aperçu 3D, le WCS est affiché différemment :

- **Grille et axes** : Toute la grille apparaît comme si l'origine WCS était l'origine monde
- **Vue isolée** : Le WCS est affiché "en isolation" - on dirait que la grille est centrée sur le WCS, pas la machine
- **Étiquettes** : Les étiquettes de coordonnées sont relatives à l'origine WCS

Cela facilite la visualisation de où votre travail s'exécutera par rapport au système de coordonnées de travail sélectionné.

## Sélectionner et Changer le WCS

### Via la Barre d'Outils

1. Localisez le menu déroulant WCS dans la barre d'outils principale (étiqueté "G53" par défaut)
2. Cliquez pour voir les systèmes de coordonnées disponibles
3. Sélectionnez le WCS que vous voulez utiliser

### Via le Panneau de Contrôle

1. Ouvrez le Panneau de Contrôle (Affichage → Panneau de Contrôle ou Ctrl+L)
2. Trouvez le menu déroulant WCS dans la section de statut machine
3. Sélectionnez le WCS souhaité dans le menu déroulant

## Définir les Décalages WCS

Vous pouvez définir où chaque origine WCS est située sur votre machine.

### Définir le Zéro à la Position Actuelle

1. Connectez-vous à votre machine
2. Sélectionnez le WCS que vous voulez configurer (ex : G54)
3. Déplacez la tête laser à la position que vous voulez être (0, 0)
4. Dans le Panneau de Contrôle, cliquez sur les boutons zéro :
   - **Zéro X** : Définit la position X actuelle comme 0 pour le WCS actif
   - **Zéro Y** : Définit la position Y actuelle comme 0 pour le WCS actif
   - **Zéro Z** : Définit la position Z actuelle comme 0 pour le WCS actif

Les décalages sont stockés dans le contrôleur de votre machine et persistent entre les sessions.

### Voir les Décalages Actuels

Le Panneau de Contrôle affiche les décalages actuels pour le WCS actif :

- **Décalages Actuels** : Affiche le décalage (X, Y, Z) depuis l'origine machine
- **Position Actuelle** : Affiche la position de la tête laser dans le WCS actif

## Le WCS dans Vos Travaux

Quand vous exécutez un travail, Rayforge utilise le WCS actif pour positionner votre travail :

1. Concevez votre travail dans le canevas
2. Sélectionnez le WCS que vous voulez utiliser
3. Exécutez le travail - il sera positionné selon le décalage WCS

Le même travail peut être exécuté à différentes positions simplement en changeant le WCS actif.

## Flux de Travail Pratiques

### Flux de Travail 1 : Positions de Montage Multiples

Vous avez un grand lit et voulez configurer trois zones de travail :

1. **Mettez votre machine à l'origine** pour établir une référence
2. **Déplacez-vous vers la première zone de travail** et définissez le décalage G54 (Zéro X, Zéro Y)
3. **Déplacez-vous vers la deuxième zone de travail** et définissez le décalage G55
4. **Déplacez-vous vers la troisième zone de travail** et définissez le décalage G56
5. Maintenant vous pouvez basculer entre G54, G55 et G56 pour exécuter des travaux dans chaque zone

### Flux de Travail 2 : Aligner au Matériau

Vous avez un morceau de matériau placé quelque part sur votre lit :

1. **Déplacez la tête laser** vers le coin de votre matériau
2. **Sélectionnez G54** (ou votre WCS préféré)
3. **Cliquez sur Zéro X et Zéro Y** pour définir le coin du matériau comme (0, 0)
4. **Concevez votre travail** avec (0, 0) comme origine
5. **Exécutez le travail** - il démarrera du coin du matériau

### Flux de Travail 3 : Grille de Production

Vous devez couper la même pièce 10 fois à différents emplacements :

1. **Concevez une pièce** dans Rayforge
2. **Configurez les décalages G54-G59** pour vos positions souhaitées
3. **Exécutez le travail** avec G54 actif
4. **Basculez vers G55** et exécutez à nouveau
5. **Répétez** pour chaque position WCS

## Notes Importantes

### Limitations du WCS

- **G53 ne peut pas être changé** : Les coordonnées machine sont fixées par le matériel
- **Les décalages persistent** : Les décalages WCS sont stockés dans le contrôleur de votre machine
- **Connexion requise** : Vous devez être connecté à une machine pour définir les décalages WCS

### WCS et Origine du Travail

Le WCS fonctionne indépendamment de vos paramètres d'origine de travail. L'origine du travail détermine où sur le canevas votre travail commence, tandis que le WCS détermine où cette position de canevas est mappée sur votre machine.

### Compatibilité Machine

Toutes les machines ne supportent pas toutes les fonctionnalités WCS :

- **GRBL (v1.1+)** : Support complet de G53-G59
- **Smoothieware** : Supporte G54-G59 (la lecture des décalages peut être limitée)
- **Contrôleurs personnalisés** : Varie selon l'implémentation

---

**Pages Connexes :**

- [Systèmes de Coordonnées](coordinate-systems) - Comprendre les systèmes de coordonnées
- [Panneau de Contrôle](../ui/control-panel) - Contrôle manuel et gestion WCS
- [Configuration Machine](../machine/general) - Configurez votre machine
- [Aperçu 3D](../ui/3d-preview) - Visualiser vos travaux
