# Panneau de contrôle

Le panneau de contrôle en bas de la fenêtre Rayforge fournit un contrôle manuel
sur la position de votre découpeur laser, l'état de la machine en temps réel, et une vue du journal
pour surveiller les opérations.

## Aperçu

Le panneau de contrôle combine plusieurs fonctions dans une interface pratique :

1. **Contrôles de déplacement** : Mouvement et positionnement manuels
2. **État de la machine** : Position et état de connexion en temps réel
3. **Console** : Terminal G-code interactif avec coloration syntaxique
4. **Système de coordonnées de travail (WCS)** : Sélection rapide du WCS

![Panneau de contrôle](/screenshots/control-panel.png)

## Accéder au panneau de contrôle

Le panneau de contrôle est toujours visible en bas de la fenêtre principale. Il peut
être basculé via :

- **Menu** : Affichage → Panneau de contrôle
- **Raccourci clavier** : Ctrl+L

:::note Connexion requise
Les contrôles de déplacement ne sont disponibles que lorsque vous êtes connecté à une machine
qui prend en charge les opérations de déplacement.
:::


## Contrôles de déplacement

Les contrôles de déplacement fournissent un contrôle manuel sur la position de votre découpeur laser,
vous permettant de déplacer précisément la tête laser pour la configuration, l'alignement et
les tests.

### Contrôles de mise à l'origine

Mettez à l'origine les axes de votre machine pour établir une position de référence :

| Bouton     | Fonction                | Description                              |
| ---------- | ----------------------- | ---------------------------------------- |
| Origine X  | Met à l'origine l'axe X | Déplace l'axe X vers sa position d'origine |
| Origine Y  | Met à l'origine l'axe Y | Déplace l'axe Y vers sa position d'origine |
| Origine Z  | Met à l'origine l'axe Z | Déplace l'axe Z vers sa position d'origine |
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

| Bouton            | Mouvement                       | Raccourci clavier |
| ----------------- | ------------------------------- | ----------------- |
| ↑                 | Y+ (Y- si la machine est Y-inversé) | Flèche haut    |
| ↓                 | Y- (Y+ si la machine est Y-inversé) | Flèche bas     |
| ←                 | X- (gauche)                     | Flèche gauche     |
| →                 | X+ (droite)                     | Flèche droite     |
| ↖ (haut-gauche)  | X- Y+/- (diagonale)             | -                 |
| ↗ (haut-droite)  | X+ Y+/- (diagonale)             | -                 |
| ↙ (bas-gauche)   | X- Y-/+ (diagonale)             | -                 |
| ↘ (bas-droite)   | X+ Y-/+ (diagonale)             | -                 |
| Z+                | Axe Z vers le haut              | Page Up           |
| Z-                | Axe Z vers le bas               | Page Down         |

:::note Focus requis
Les raccourcis clavier ne fonctionnent que lorsque la fenêtre principale a le focus.
:::


### Retour visuel

Les boutons de déplacement fournissent un retour visuel :

- **Normal** : Le bouton est activé et sûr à utiliser
- **Avertissement (orange)** : Le mouvement approcherait ou dépasserait les limites logicielles
- **Désactivé** : Le mouvement n'est pas pris en charge ou la machine n'est pas connectée

### Paramètres de déplacement

Configurez le comportement des opérations de déplacement :

**Vitesse de déplacement :**
- **Plage** : 1-10 000 mm/min
- **Par défaut** : 1 000 mm/min
- **Objectif** : Contrôle la vitesse de déplacement de la tête laser

:::tip Sélection de vitesse
- Utilisez des vitesses plus faibles (100-500 mm/min) pour un positionnement précis
- Utilisez des vitesses plus élevées (1 000-3 000 mm/min) pour des mouvements plus importants
- Des vitétés très élevées peuvent causer des pas perdus sur certaines machines
:::


**Distance de déplacement :**
- **Plage** : 0.1-1 000 mm
- **Par défaut** : 10.0 mm
- **Objectif** : Contrôle la distance de déplacement de la tête laser par pression de bouton

:::tip Sélection de distance
- Utilisez de petites distances (0.1-1.0 mm) pour le réglage fin
- Utilisez des distances moyennes (5-20 mm) pour le positionnement général
- Utilisez de grandes distances (50-100 mm) pour le repositionnement rapide
:::


## Affichage de l'état de la machine

Le panneau de contrôle affiche des informations en temps réel sur votre machine :

### Position actuelle

Affiche la position de la tête laser dans le système de coordonnées actif :

- Les coordonnées sont relatives à l'origine du WCS sélectionné
- Se met à jour en temps réel lorsque vous déplacez ou exécutez des travaux
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

Sélectionnez le système de coordonnées actuellement actif :

| Option        | Type  | Description                                     |
| ------------- | ----- | ----------------------------------------------- |
| G53 (Machine) | Fixe  | Coordonnées machine absolues, ne peut pas être modifié |
| G54 (Travail 1)| Utilisateur | Premier système de coordonnées de travail    |
| G55 (Travail 2)| Utilisateur | Deuxième système de coordonnées de travail   |
| G56 (Travail 3)| Utilisateur | Troisième système de coordonnées de travail   |
| G57 (Travail 4)| Utilisateur | Quatrième système de coordonnées de travail  |
| G58 (Travail 5)| Utilisateur | Cinquième système de coordonnées de travail   |
| G59 (Travail 6)| Utilisateur | Sixième système de coordonnées de travail    |

### Décalages actuels

Affiche les valeurs de décalage pour le WCS actif :

- Affiché en (X, Y, Z) en millimètres
- Représente la distance de l'origine machine à l'origine du WCS
- Se met à jour automatiquement lorsque les décalages du WCS changent

### Définir l'origine du WCS

Définissez où l'origine du WCS actif devrait être :

| Bouton | Fonction | Description                                          |
| ------ | -------- | ---------------------------------------------------- |
| Zéro X | Définir X=0 | Fait de la position X actuelle l'origine X pour le WCS actif |
| Zéro Y | Définir Y=0 | Fait de la position Y actuelle l'origine Y pour le WCS actif |
| Zéro Z | Définir Z=0 | Fait de la position Z actuelle l'origine Z pour le WCS actif |

:::note G53 ne peut pas être modifié
Les boutons Zéro sont désactivés lorsque G53 (Coordonnées machine) est sélectionné,
car les coordonnées machine sont fixes par le matériel.
:::


:::tip Flux de travail pour définir le WCS
1. Connectez-vous à votre machine et mettez à l'origine tous les axes
2. Sélectionnez le WCS que vous voulez configurer (par exemple, G54)
3. Déplacez la tête laser à la position d'origine souhaitée
4. Cliquez sur Zéro X et Zéro Y pour définir cette position comme (0, 0)
5. Le décalage est stocké dans le contrôleur de votre machine
:::


## Console

La console fournit une interface de type terminal interactif pour envoyer des commandes G-code
et surveiller la communication machine :

### Saisie de commandes

La boîte de saisie de commandes vous permet d'envoyer du G-code brut directement à la machine :

- **Support multi-lignes** : Collez ou tapez plusieurs commandes
- **Entrée** : Envoie toutes les commandes
- **Shift+Entrée** : Insère une nouvelle ligne (pour l'édition avant l'envoi)
- **Historique** : Utilisez les flèches Haut/Bas pour naviguer dans les commandes précédemment envoyées

### Affichage du journal

Le journal montre la communication entre Rayforge et votre machine avec
une coloration syntaxique pour une lecture facile :

- **Commandes utilisateur** (bleu) : Commandes que vous avez saisies ou envoyées pendant les travaux
- **Horodatages** (gris) : Heure de chaque message
- **Erreurs** (rouge) : Messages d'erreur de la machine
- **Avertissements** (orange) : Messages d'avertissement
- **Interrogations d'état** (atténué) : Rapports de position/état en temps réel comme
  `<Idle|WPos:0.000,0.000,0.000|...>`

### Mode verbeux

Cliquez sur l'icône de terminal dans le coin supérieur droit de la console pour basculer
la sortie verbeuse :

- **Désactivé** (par défaut) : Masque les interrogations d'état fréquentes et les réponses "ok"
- **Activé** : Affiche toute la communication machine

### Comportement du défilement automatique

La console défile automatiquement pour afficher les nouveaux messages :

- Faire défiler vers le haut désactive le défilement automatique pour que vous puissiez revoir l'historique
- Faire défiler vers le bas réactive le défilement automatique
- Les nouveaux messages apparaissent immédiatement lorsque le défilement automatique est actif

### Utiliser la console pour le dépannage

La console est inestimable pour diagnostiquer les problèmes :

- Vérifiez que les commandes sont envoyées correctement
- Recherchez les messages d'erreur du contrôleur
- Surveillez l'état et la stabilité de la connexion
- Passez en revue la progression de l'exécution du travail en temps réel
- Envoyez des commandes de diagnostic (par exemple, `$$` pour voir les paramètres GRBL)

## Compatibilité machine

Le panneau de contrôle s'adapte aux capacités de votre machine :

### Support des axes

- **Axe X/Y** : Pris en charge par pratiquement tous les découpeurs laser
- **Axe Z** : Disponible uniquement sur les machines avec contrôle de l'axe Z
- **Mouvement diagonal** : Nécessite le support des axes X et Y

### Types de machines

| Type de machine    | Support de déplacement | Notes                     |
| ------------------ | ---------------------- | ------------------------- |
| GRBL (v1.1+)       | Complet                | Prend en charge toutes les fonctionnalités de déplacement |
| Smoothieware       | Complet                | Prend en charge toutes les fonctionnalités de déplacement |
| Contrôleurs personnalisés | Variable         | Dépend de l'implémentation |

## Fonctionnalités de sécurité

### Limites logicielles

Lorsque les limites logicielles sont activées dans votre profil machine :

- Les boutons affichent un avertissement orange en approche des limites
- Le mouvement est automatiquement limité pour éviter de dépasser les bornes
- Fournit un retour visuel pour éviter les collisions

### État de la connexion

- Tous les contrôles sont désactivés lorsque vous n'êtes pas connecté à une machine
- Les boutons mettent à jour leur sensibilité en fonction de l'état de la machine
- Empêche le mouvement accidentel pendant l'opération

---

**Pages connexes :**

- [Systèmes de coordonnées de travail (WCS)](../general-info/coordinate-systems) - Gérer le WCS
- [Configuration machine](../machine/general) - Configurez votre machine
- [Raccourcis clavier](../reference/shortcuts) - Référence complète des raccourcis
- [Fenêtre principale](main-window) - Aperçu de l'interface principale
- [Paramètres généraux](../machine/general) - Configuration de l'appareil
