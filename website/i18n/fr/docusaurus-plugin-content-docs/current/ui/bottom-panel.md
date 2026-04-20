# Panneau inférieur

Le panneau inférieur en bas de la fenêtre Rayforge fournit un contrôle manuel
sur la position de ton découpeur laser, l'état de la machine en temps réel, une vue du journal
pour surveiller les opérations, un visualiseur G-code et un navigateur d'actifs.

## Aperçu

Le panneau inférieur combine plusieurs fonctions dans une interface pratique :

1. **Onglets amarrables** : Bascule entre Console, Visualiseur G-code et Actifs en utilisant
   la bande d'icônes sur la gauche
2. **Contrôles de déplacement** : Mouvement et positionnement manuels (toujours visibles)
3. **État de la machine** : Position et état de connexion en temps réel
4. **Système de coordonnées de travail (WCS)** : Sélection rapide du WCS (toujours visible)

Chaque zone du panneau possède une bande d'icônes d'onglets sur la gauche qui te permet
de basculer entre la **Console**, le **Visualiseur G-code** et le navigateur d'**Actifs**.
Les contrôles de déplacement et les contrôles WCS sur le côté droit restent visibles quel
que soit l'onglet actif. Les onglets peuvent être réordonnés par glisser-déposer dans
leur bande, et tu peux glisser des onglets entre les zones du panneau ou sur les
séparateurs pour réorganiser la disposition en plusieurs colonnes. Les colonnes vides
sont supprimées automatiquement.

![Panneau inférieur](/screenshots/bottom-panel-console.png)

## Accéder au panneau inférieur

Le panneau inférieur peut être basculé via :

- **Menu** : Affichage → Panneau inférieur
- **Raccourci clavier** : Ctrl+L

:::note Connexion requise
Les contrôles de déplacement ne sont disponibles que lorsque tu es connecté à une machine
qui prend en charge les opérations de déplacement.
:::

## Contrôles de déplacement

Les contrôles de déplacement fournissent un contrôle manuel sur la position de ton découpeur laser,
te permettant de déplacer précisément la tête laser pour la configuration, l'alignement et
les tests.

### Contrôles de mise à l'origine

Mets à l'origine les axes de ta machine pour établir une position de référence :

| Bouton       | Fonction                      | Description                                 |
| ------------ | ----------------------------- | ------------------------------------------- |
| Origine tout | Met à l'origine tous les axes | Met à l'origine tous les axes simultanément |

:::tip Séquence de mise à l'origine
Il est recommandé de mettre à l'origine tous les axes avant de commencer tout travail pour assurer
un positionnement précis.
:::

### Mouvement directionnel

Les contrôles de déplacement fournissent des boutons pour le mouvement directionnel :

```
  ↖  ↑  ↗
  ←  •  →
  ↙  ↓  ↘
```

| Bouton          | Mouvement                           | Raccourci clavier |
| --------------- | ----------------------------------- | ----------------- |
| ↑               | Y+ (Y- si la machine est Y-inversé) | Flèche haut       |
| ↓               | Y- (Y+ si la machine est Y-inversé) | Flèche bas        |
| ←               | X- (gauche)                         | Flèche gauche     |
| →               | X+ (droite)                         | Flèche droite     |
| ↖ (haut-gauche) | X- Y+/- (diagonale)                 | -                 |
| ↗ (haut-droite) | X+ Y+/- (diagonale)                 | -                 |
| ↙ (bas-gauche)  | X- Y-/+ (diagonale)                 | -                 |
| ↘ (bas-droite)  | X+ Y-/+ (diagonale)                 | -                 |
| Z+              | Axe Z vers le haut                  | Page Up           |
| Z-              | Axe Z vers le bas                   | Page Down         |

:::note Focus requis
Les raccourcis clavier ne fonctionnent que lorsque la fenêtre principale a le focus.
:::

### Retour visuel

Les boutons de déplacement fournissent un retour visuel :

- **Normal** : Le bouton est activé et sûr à utiliser
- **Avertissement (orange)** : Le mouvement approcherait ou dépasserait les limites logicielles
- **Désactivé** : Le mouvement n'est pas pris en charge ou la machine n'est pas connectée

### Paramètres de déplacement

Configure le comportement des opérations de déplacement :

**Vitesse de déplacement :**

- **Plage** : 1-60 000 mm/min
- **Par défaut** : 1 000 mm/min
- **Objectif** : Contrôle la vitesse de déplacement de la tête laser

:::tip Sélection de vitesse

- Utilise des vitesses plus faibles (100-500 mm/min) pour un positionnement précis
- Utilise des vitesses plus élevées (1 000-3 000 mm/min) pour des mouvements plus importants
- Des vitesses très élevées peuvent causer des pas perdus sur certaines machines
  :::

**Distance de déplacement :**

- **Plage** : 0.1-1 000 mm
- **Par défaut** : 10.0 mm
- **Objectif** : Contrôle la distance de déplacement de la tête laser par pression de bouton

:::tip Sélection de distance

- Utilise de petites distances (0.1-1.0 mm) pour le réglage fin
- Utilise des distances moyennes (5-20 mm) pour le positionnement général
- Utilise de grandes distances (50-100 mm) pour le repositionnement rapide
  :::

## Affichage de l'état de la machine

Le panneau de contrôle affiche des informations en temps réel sur ta machine :

### Position actuelle

Affiche la position de la tête laser dans le système de coordonnées actif :

- Les coordonnées sont relatives à l'origine du WCS sélectionné
- Se met à jour en temps réel lorsque tu déplaces ou exécutes des travaux
- Format : valeurs X, Y, Z en millimètres

### État de la connexion

- **Connecté** : Indicateur vert, la machine répond
- **Déconnecté** : Indicateur gris, pas de connexion machine
- **Erreur** : Indicateur rouge, problème de connexion ou de communication

### État de la machine

- **Inactif** : La machine est prête pour les commandes
- **En cours** : Le travail est en cours d'exécution
- **En pause** : Le travail est en pause
- **Alarme** : La machine est en état d'alarme
- **Origine** : Le cycle de mise à l'origine est en cours

## Système de coordonnées de travail (WCS)

Le panneau de contrôle fournit un accès rapide à la gestion du système de coordonnées de travail.

### Sélection du système actif

Sélectionne le système de coordonnées actuellement actif :

| Option          | Type        | Description                                            |
| --------------- | ----------- | ------------------------------------------------------ |
| G53 (Machine)   | Fixe        | Coordonnées machine absolues, ne peut pas être modifié |
| G54 (Travail 1) | Utilisateur | Premier système de coordonnées de travail              |
| G55 (Travail 2) | Utilisateur | Deuxième système de coordonnées de travail             |
| G56 (Travail 3) | Utilisateur | Troisième système de coordonnées de travail            |
| G57 (Travail 4) | Utilisateur | Quatrième système de coordonnées de travail            |
| G58 (Travail 5) | Utilisateur | Cinquième système de coordonnées de travail            |
| G59 (Travail 6) | Utilisateur | Sixième système de coordonnées de travail              |

### Définir l'origine du WCS

Définis où l'origine du WCS actif devrait être :

| Bouton               | Fonction      | Description                                                                                |
| -------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| Cliquer pour zéro    | Définir X,Y=0 | Clique sur l'icône du réticule, puis clique sur le canevas pour définir le zéro de travail |
| Éditer les décalages | Éditer        | Éditer manuellement les valeurs de décalage du WCS                                         |
| Zéro X               | Définir X=0   | Fait de la position X actuelle l'origine X pour le WCS actif                               |
| Zéro Y               | Définir Y=0   | Fait de la position Y actuelle l'origine Y pour le WCS actif                               |
| Zéro Z               | Définir Z=0   | Fait de la position Z actuelle l'origine Z pour le WCS actif                               |

:::note G53 ne peut pas être modifié
Les boutons Zéro sont désactivés lorsque G53 (Coordonnées machine) est sélectionné,
car les coordonnées machine sont fixes par le matériel.
:::

:::tip Flux de travail pour définir le WCS

1. Connecte-toi à ta machine et mets à l'origine tous les axes
2. Sélectionne le WCS que tu veux configurer (par exemple, G54)
3. Déplace la tête laser à la position d'origine souhaitée
4. Clique sur Zéro X et Zéro Y pour définir cette position comme (0, 0)
5. Le décalage est stocké dans le contrôleur de ta machine
   :::

## Onglet Console

L'onglet Console fournit une interface de type terminal interactif pour envoyer des
commandes G-code et surveiller la communication machine. Clique sur l'icône console
dans la bande d'onglets pour basculer vers cette vue.

### Saisie de commandes

La boîte de saisie de commandes te permet d'envoyer du G-code brut directement à la machine :

- **Support multi-lignes** : Colle ou tape plusieurs commandes
- **Entrée** : Envoie toutes les commandes
- **Shift+Entrée** : Insère une nouvelle ligne (pour l'édition avant l'envoi)
- **Historique** : Utilise les flèches Haut/Bas pour naviguer dans les commandes précédemment envoyées

### Affichage du journal

Le journal montre la communication entre Rayforge et ta machine avec
une coloration syntaxique pour une lecture facile :

- **Commandes utilisateur** (bleu) : Commandes que tu as saisies ou envoyées pendant les travaux
- **Horodatages** (gris) : Heure de chaque message
- **Erreurs** (rouge) : Messages d'erreur de la machine
- **Avertissements** (orange) : Messages d'avertissement
- **Interrogations d'état** (atténué) : Rapports de position/état en temps réel comme
  `&lt;Idle|WPos:0.000,0.000,0.000|...&gt;`

### Mode verbeux

Clique sur l'icône de terminal dans le coin supérieur droit de la console pour basculer
la sortie verbeuse :

- **Désactivé** (par défaut) : Masque les interrogations d'état fréquentes et les réponses « ok »
- **Activé** : Affiche toute la communication machine

### Comportement du défilement automatique

La console défile automatiquement pour afficher les nouveaux messages :

- Faire défiler vers le haut désactive le défilement automatique pour que tu puisses revoir l'historique
- Faire défiler vers le bas réactive le défilement automatique
- Les nouveaux messages apparaissent immédiatement lorsque le défilement automatique est actif

### Utiliser la console pour le dépannage

La console est inestimable pour diagnostiquer les problèmes :

- Vérifie que les commandes sont envoyées correctement
- Recherche les messages d'erreur du contrôleur
- Surveille l'état et la stabilité de la connexion
- Passe en revue la progression de l'exécution du travail en temps réel
- Envoie des commandes de diagnostic (par exemple, `$$` pour voir les paramètres GRBL)

## Onglet Visualiseur G-code

L'onglet Visualiseur G-code affiche le G-code généré pour les opérations actuelles.
Clique sur l'icône G-code dans la bande d'onglets pour basculer vers cette vue.

### Caractéristiques

- **Coloration syntaxique** : Les commandes G-code sont colorées pour une meilleure lisibilité
- **Surlignage de ligne** : La ligne actuellement exécutée est mise en surbrillance pendant
  l'exécution du travail
- **Actualisation automatique** : Le contenu du G-code se met à jour automatiquement lorsque
  les opérations ou les paramètres du document changent

## Onglet Actifs

L'onglet Actifs affiche tous les éléments de stock et croquis dans ton document. Clique sur
l'icône des actifs dans la bande d'onglets pour basculer vers cette vue.

Lorsque la liste des actifs est vide, elle affiche des boutons pour ajouter du stock ou créer
un nouveau croquis. Tu peux glisser les actifs depuis cette liste vers le canevas pour les
placer. Double-cliquer sur un actif de stock ouvre ses propriétés.

Lorsque l'onglet Console ou Visualiseur G-code est actif, tu peux aussi appuyer sur
<kbd>Ctrl+F</kbd> pour rechercher dans le contenu.

## Compatibilité machine

Le panneau inférieur s'adapte aux capacités de ta machine :

### Support des axes

- **Axe X/Y** : Pris en charge par pratiquement tous les découpeurs laser
- **Axe Z** : Disponible uniquement sur les machines avec contrôle de l'axe Z
- **Mouvement diagonal** : Nécessite le support des axes X et Y

### Types de machines

| Type de machine           | Support de déplacement | Notes                                                     |
| ------------------------- | ---------------------- | --------------------------------------------------------- |
| GRBL (v1.1+)              | Complet                | Prend en charge toutes les fonctionnalités de déplacement |
| Smoothieware              | Complet                | Prend en charge toutes les fonctionnalités de déplacement |
| Contrôleurs personnalisés | Variable               | Dépend de l'implémentation                                |

## Fonctionnalités de sécurité

### Limites logicielles

Lorsque les limites logicielles sont activées dans ton profil machine :

- Les boutons affichent un avertissement orange en approche des limites
- Le mouvement est automatiquement limité pour éviter de dépasser les bornes
- Fournit un retour visuel pour éviter les collisions

### État de la connexion

- Tous les contrôles sont désactivés lorsque tu n'es pas connecté à une machine
- Les boutons mettent à jour leur sensibilité en fonction de l'état de la machine
- Empêche le mouvement accidentel pendant l'opération

---

**Pages connexes :**

- [Systèmes de coordonnées de travail (WCS)](../general-info/coordinate-systems) - Gérer le WCS
- [Configuration machine](../machine/general) - Configure ta machine
- [Raccourcis clavier](../reference/shortcuts) - Référence complète des raccourcis
- [Fenêtre principale](main-window) - Aperçu de l'interface principale
- [Paramètres généraux](../machine/general) - Configuration de l'appareil
