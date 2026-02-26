# Systèmes de Coordonnées de Travail (WCS)

Les Systèmes de Coordonnées de Travail (WCS) vous permettent de définir plusieurs points de référence sur la zone de travail de votre machine. Cela facilite l'exécution du même travail à différentes positions sans reconcevoir ou repositionner vos pièces.

## Espaces de Coordonnées

Rayforge utilise trois espaces de coordonnées qui travaillent ensemble :

| Espace       | Description                                                                                                                             |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| **MACHINE**  | Coordonnées absolues relatives à la position d'origine de la machine. L'origine est fixée par le matériel.                              |
| **WORKAREA** | La zone utilisable dans votre machine, en tenant compte des marges autour du lit.                                                       |
| **WCS**      | Le système de coordonnées de votre travail. Origine configurable par l'utilisateur pour la conception et le positionnement des travaux. |

:::note Note pour les Développeurs
En interne, Rayforge utilise un système de coordonnées normalisé appelé espace WORLD.
L'espace WORLD décrit le même espace physique que l'espace MACHINE, mais avec une
convention fixe : Y vers le haut avec origine en bas à gauche. Cela simplifie les
calculs internes et le rendu. Les utilisateurs n'ont pas besoin d'interagir directement
avec l'espace WORLD.
:::

### Espace MACHINE

L'espace MACHINE est le système de coordonnées absolu relatif à la position d'origine
de votre machine. L'origine (0,0) est déterminée par la configuration de prise d'origine
de votre machine.

- **Origine** : Position d'origine de la machine (0,0,0) - fixée par le matériel
- **Objectif** : Référence pour tous les autres systèmes de coordonnées
- **Fixe** : Ne peut pas être modifié par logiciel

La direction des coordonnées dépend de la configuration de votre machine :

- **Coin d'origine** : Peut être en haut à gauche, en bas à gauche, en haut à droite ou en bas à droite
- **Direction des axes** : Les axes X et Y peuvent être inversés selon la configuration matérielle

### Espace WORKAREA

L'espace WORKAREA définit la zone utilisable dans votre machine, en tenant compte
des marges autour des bords de votre lit.

- **Origine** : Même coin que l'origine de l'espace MACHINE
- **Objectif** : Définit la zone réelle où les travaux peuvent être exécutés
- **Marges** : Des marges peuvent être appliquées (gauche, haut, droite, bas)

Par exemple, si votre machine fait 400×300mm mais a une marge de 10mm de tous les côtés,
la WORKAREA serait de 380×280mm commençant à la position (10, 10) dans l'espace MACHINE.

## Comprendre les WCS

Pensez aux WCS comme des "points zéro" personnalisables pour votre travail. Alors que votre machine a une position d'origine fixe (déterminée par les butées), les WCS vous permettent de définir où vous voulez que votre travail commence.

### Pourquoi Utiliser les WCS ?

- **Fixations multiples** : Configurez plusieurs zones de travail sur votre lit et passez de l'une à l'autre
- **Positionnement répétable** : Exécutez le même travail à différents endroits
- **Alignement rapide** : Définissez un point de référence basé sur votre matériau ou pièce
- **Flux de travail de production** : Organisez plusieurs travaux sur votre zone de travail

## Types de WCS

Rayforge prend en charge les systèmes de coordonnées suivants :

| Système | Type      | Description                                                         |
| ------- | --------- | ------------------------------------------------------------------- |
| **G53** | Machine   | Coordonnées machine absolues (fixes, ne peuvent pas être modifiées) |
| **G54** | Travail 1 | Premier système de coordonnées de travail (par défaut)              |
| **G55** | Travail 2 | Deuxième système de coordonnées de travail                          |
| **G56** | Travail 3 | Troisième système de coordonnées de travail                         |
| **G57** | Travail 4 | Quatrième système de coordonnées de travail                         |
| **G58** | Travail 5 | Cinquième système de coordonnées de travail                         |
| **G59** | Travail 6 | Sixième système de coordonnées de travail                           |

### Coordonnées Machine (G53)

G53 représente la position absolue de votre machine, avec zéro à la position d'origine de la machine. Ceci est fixé par votre matériel et ne peut pas être modifié.

**Quand l'utiliser :**

- Prise d'origine et calibration
- Positionnement absolu par rapport aux limites de la machine
- Quand vous avez besoin de référencer la position physique de la machine

### Coordonnées de Travail (G54-G59)

Ce sont des systèmes de coordonnées avec décalage que vous pouvez définir. Chacun a son propre point zéro que vous pouvez placer n'importe où sur votre zone de travail.

**Quand les utiliser :**

- Configurer plusieurs fixations de travail
- Aligner sur des positions de matériau
- Exécuter le même travail à différents endroits

## Visualiser les WCS dans l'Interface

### Canevas 2D

Le canevas 2D affiche votre origine WCS avec un marqueur vert :

- **Lignes vertes** : Indiquent la position de l'origine WCS actuelle (0, 0)
- **Alignement de la grille** : Les lignes de la grille sont alignées sur l'origine WCS, pas sur l'origine machine

Le marqueur d'origine se déplace lorsque vous changez le WCS actif ou son décalage, vous montrant exactement où votre travail commencera.

### Aperçu 3D

Dans l'aperçu 3D, les WCS sont affichés différemment :

- **Grille et axes** : Toute la grille apparaît comme si l'origine WCS était l'origine du monde
- **Vue isolée** : Le WCS est affiché "en isolation" - on dirait que la grille est centrée sur le WCS, pas sur la machine
- **Étiquettes** : Les étiquettes de coordonnées sont relatives à l'origine WCS

Cela facilite la visualisation de l'endroit où votre travail sera exécuté par rapport au système de coordonnées de travail sélectionné.

## Sélectionner et Changer les WCS

### Via la Barre d'Outils

1. Localisez le menu déroulant WCS dans la barre d'outils principale (étiqueté "G53" par défaut)
2. Cliquez pour voir les systèmes de coordonnées disponibles
3. Sélectionnez le WCS que vous voulez utiliser

### Via le Panneau de Contrôle

1. Ouvrez le Panneau de Contrôle (Affichage → Panneau de Contrôle ou Ctrl+L)
2. Trouvez le menu déroulant WCS dans la section d'état de la machine
3. Sélectionnez le WCS souhaité dans le menu déroulant

## Définir les Décalages WCS

Vous pouvez définir où chaque origine WCS est située sur votre machine.

### Définir Zéro à la Position Actuelle

1. Connectez-vous à votre machine
2. Sélectionnez le WCS que vous voulez configurer (ex., G54)
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

## Les WCS dans vos Travaux

Lorsque vous exécutez un travail, Rayforge utilise le WCS actif pour positionner votre travail :

1. Concevez votre travail dans le canevas
2. Sélectionnez le WCS que vous voulez utiliser
3. Exécutez le travail - il sera positionné selon le décalage WCS

Le même travail peut être exécuté à différentes positions simplement en changeant le WCS actif.

## Flux de Travail Pratiques

### Flux de Travail 1 : Plusieurs Positions de Fixation

Vous avez un grand lit et voulez configurer trois zones de travail :

1. **Faites une prise d'origine** pour établir une référence
2. **Déplacez-vous vers la première zone de travail** et définissez le décalage G54 (Zéro X, Zéro Y)
3. **Déplacez-vous vers la deuxième zone de travail** et définissez le décalage G55
4. **Déplacez-vous vers la troisième zone de travail** et définissez le décalage G56
5. Maintenant vous pouvez basculer entre G54, G55 et G56 pour exécuter des travaux dans chaque zone

### Flux de Travail 2 : Alignement sur le Matériau

Vous avez un morceau de matériau placé quelque part sur votre lit :

1. **Déplacez la tête laser** vers le coin de votre matériau
2. **Sélectionnez G54** (ou votre WCS préféré)
3. **Cliquez sur Zéro X et Zéro Y** pour définir le coin du matériau comme (0, 0)
4. **Concevez votre travail** avec (0, 0) comme origine
5. **Exécutez le travail** - il commencera depuis le coin du matériau

### Flux de Travail 3 : Grille de Production

Vous devez couper la même pièce 10 fois à différents endroits :

1. **Concevez une pièce** dans Rayforge
2. **Configurez les décalages G54-G59** pour vos positions souhaitées
3. **Exécutez le travail** avec G54 actif
4. **Passez à G55** et exécutez à nouveau
5. **Répétez** pour chaque position WCS

## Notes Importantes

### Limitations des WCS

- **G53 ne peut pas être modifié** : Les coordonnées machine sont fixées par le matériel
- **Les décalages persistent** : Les décalages WCS sont stockés dans le contrôleur de votre machine
- **Connexion requise** : Vous devez être connecté à une machine pour définir les décalages WCS

---

**Pages Liées :**

- [Panneau de Contrôle](../ui/control-panel) - Contrôle manuel et gestion des WCS
- [Configuration Machine](../machine/general) - Configurez votre machine
- [Aperçu 3D](../ui/3d-preview) - Visualiser vos travaux
