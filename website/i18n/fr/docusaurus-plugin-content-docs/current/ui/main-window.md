# Fenêtre principale

La fenêtre principale de Rayforge est votre espace de travail principal pour créer et gérer
les travaux laser.

## Disposition de la fenêtre

![Fenêtre principale](/screenshots/main-standard.png)

### 1. Barre de menu

Accédez à toutes les fonctions Rayforge via des menus organisés :

- **Fichier** : Ouvrir, enregistrer, importer, exporter et fichiers récents
- **Édition** : Annuler, rétablir, copier, coller, préférences
- **Affichage** : Zoom, grille, règles, panneaux et modes d'affichage
- **Opérations** : Ajouter, modifier et gérer les opérations
- **Machine** : Connecter, déplacer, mettre à l'origine, démarrer/arrêter les travaux
- **Aide** : À propos, Don, Enregistrer le journal de débogage

### 2. Barre d'outils

Accès rapide aux contrôles fréquemment utilisés :

- **Liste déroulante Machine** : Sélectionne ta machine, affiche l'état de connexion et
  montre le temps restant estimé pendant les travaux
- **Liste déroulante WCS** : Sélectionne le système de coordonnées de travail actif (G53-G59)
- **Basculer la simulation** : Active/désactive le mode simulation
- **Focus laser** : Active/désactive le mode de mise au point du laser
- **Contrôles de travail** : Boutons Origine, Cadrer, Envoyer, Pause et Annuler

La liste déroulante Machine affiche l'état de connexion de ta machine et son état actuel
(p. ex. Inactif, En cours) directement dans la barre d'outils. Pendant l'exécution d'un travail,
elle affiche aussi une estimation du temps restant.

La liste déroulante WCS te permet de basculer rapidement entre les systèmes de coordonnées.
Voir [Systèmes de coordonnées de travail](../general-info/coordinate-systems) pour
plus d'informations.

Les bascules de visibilité pour les pièces, les onglets, le flux caméra, les déplacements
rapides et d'autres éléments ont été déplacées vers des boutons superposés sur le canevas
lui-même, pour qu'ils soient toujours à portée de main pendant que tu travailles.

### 3. Canevas

L'espace de travail principal où tu peux :

- Importer et organiser des conceptions
- Prévisualiser les trajets d'outil
- Positionner des objets par rapport à l'origine machine
- Tester les limites de cadrage

**Contrôles du canevas :**

- **Panoramique** : Glisser clic milieu ou <kbd>espace</kbd> + glisser
- **Zoom** : Molette de souris ou <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Réinitialiser la vue** : <kbd>ctrl+0</kbd> ou Affichage → Réinitialiser le zoom

### 4. Panneau latéral

Le panneau latéral est un panneau flottant sur le côté droit du canevas. Il affiche
le flux de travail du calque actif sous forme de liste verticale d'étapes. Chaque
étape affiche son nom, un résumé (p. ex. puissance et vitesse), et des boutons pour
la visibilité, les paramètres et la suppression. Utilisez le bouton **+** pour ajouter
de nouvelles étapes. Les étapes peuvent être réorganisées par glisser-déposer.

En cliquant sur le bouton des paramètres d'une étape, une boîte de dialogue s'ouvre
vous permettant de configurer le type d'opération, la puissance du laser, la vitesse
de coupe, l'assistance par air, la largeur du faisceau et les options de
post-traitement. Les valeurs des curseurs sont modifiables — cliquez sur une valeur à
côté d'un curseur et tapez le nombre exact que vous souhaitez.

Le panneau peut être déplacé lorsqu'il n'est pas nécessaire.

### 5. Panneau inférieur

Le panneau inférieur fournit des onglets ancrables qui peuvent être réorganisés par
glisser-déposer et divisés en plusieurs colonnes. Les onglets disponibles sont :

- **Calques** : Affiche tous les calques sous forme de colonnes côte à côte. Chaque
  colonne possède un en-tête avec le nom du calque et des contrôles, un pipeline
  horizontal compact d'icônes d'étapes représentant le flux de travail, et une liste
  de pièces. Les calques et les pièces peuvent être réorganisés par glisser-déposer.
- **Actifs** : Liste les éléments de stock et les croquis de votre document.
- **Console** : Terminal interactif pour envoyer du G-code et surveiller la
  communication de la machine.
- **Visualiseur G-code** : Affiche le G-code généré avec coloration syntaxique.
- **Contrôles** : Contrôles de déplacement pour le positionnement manuel et la gestion
  du WCS.

Le temps estimé du travail est affiché dans l'en-tête de la liste des calques.

Voir [Panneau inférieur](bottom-panel) pour des informations détaillées.

## Gestion des fenêtres

### Panneaux

Afficher/masquer les panneaux selon les besoins :

- **Panneau inférieur** : Affichage → Panneau inférieur (<kbd>ctrl+l</kbd>)

### Mode plein écran

Concentrez-vous sur votre travail en plein écran :

- Entrer : <kbd>f11</kbd> ou Affichage → Plein écran
- Quitter : <kbd>f11</kbd> ou <kbd>échappe</kbd>

## Personnalisation

Personnalisez l'interface dans **Édition → Paramètres** :

- **Thème** : Clair, sombre ou système
- **Unités** : Millimètres ou pouces
- **Grille** : Afficher/masquer et configurer l'espacement de la grille
- **Règles** : Afficher/masquer les règles sur le canevas

---

**Pages connexes :**

- [Systèmes de coordonnées de travail](../general-info/coordinate-systems) - WCS
- [Outils du canevas](canvas-tools) - Outils pour manipuler les conceptions
- [Panneau inférieur](bottom-panel) - Contrôle manuel de la machine, état et journaux
- [Prévisualisation 3D](3d-preview) - Visualiser les trajets d'outil en 3D
