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
- **Aide** : Documentation, à propos et support

### 2. Barre d'outils

Accès rapide aux outils fréquemment utilisés :

- **Outil de sélection** : Sélectionner et déplacer des objets
- **Outil panoramique** : Naviguer sur le canevas
- **Outil zoom** : Zoomer/dézoomer sur des zones spécifiques
- **Outil de mesure** : Mesurer des distances et des angles
- **Outils d'alignement** : Aligner et distribuer des objets
- **Liste déroulante WCS** : Sélectionner le système de coordonnées de travail actif (G53-G59)

La liste déroulante WCS vous permet de basculer rapidement entre les systèmes de coordonnées.
Voir [Systèmes de coordonnées de travail](../general-info/work-coordinate-systems) pour
plus d'informations.

### 3. Canevas

L'espace de travail principal où vous :

- Importez et disposez des conceptions
- Prévisualisez les trajets d'outil
- Positionnez des objets par rapport à l'origine machine
- Testez les limites de cadrage

**Contrôles du canevas :**

- **Panoramique** : Glisser clic milieu ou <kbd>espace</kbd> + glisser
- **Zoom** : Molette de souris ou <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Réinitialiser la vue** : <kbd>ctrl+0</kbd> ou Affichage → Réinitialiser le zoom

### 4. Panneau Calques

Gérez les opérations et les affectations de calques :

- Visualisez toutes les opérations dans votre projet
- Affectez des opérations aux éléments de conception
- Réorganisez l'exécution des opérations
- Activez/désactivez des opérations individuelles
- Configurez les paramètres des opérations

### 5. Panneau Propriétés

Configurez les paramètres pour les objets ou opérations sélectionnés :

- Type d'opération (Contour, Raster, etc.)
- Paramètres de puissance et de vitesse
- Nombre de passes
- Options avancées (overscan, kerf, onglets)

### 6. Panneau de contrôle

Le panneau de contrôle en bas de la fenêtre fournit :

- **Contrôles de déplacement** : Mouvement et positionnement manuels de la machine
- **État de la machine** : Position et état de connexion en temps réel
- **Vue du journal** : Communication G-code et historique des opérations
- **Gestion du WCS** : Sélection et mise à zéro du système de coordonnées de travail

Voir [Panneau de contrôle](control-panel) pour des informations détaillées.

## Gestion des fenêtres

### Panneaux

Afficher/masquer les panneaux selon les besoins :

- **Panneau Calques** : Affichage → Panneau Calques (<kbd>ctrl+l</kbd>)
- **Panneau Propriétés** : Affichage → Panneau Propriétés (<kbd>ctrl+i</kbd>)

### Mode plein écran

Concentrez-vous sur votre travail en plein écran :

- Entrer : <kbd>f11</kbd> ou Affichage → Plein écran
- Quitter : <kbd>f11</kbd> ou <kbd>échappe</kbd>

## Personnalisation

Personnalisez l'interface dans **Édition → Préférences** :

- **Thème** : Clair, sombre ou système
- **Unités** : Millimètres ou pouces
- **Grille** : Afficher/masquer et configurer l'espacement de la grille
- **Règles** : Afficher/masquer les règles sur le canevas
- **Barre d'outils** : Personnaliser les boutons visibles

---

**Pages connexes :**

- [Systèmes de coordonnées de travail](../general-info/work-coordinate-systems) - WCS
- [Outils du canevas](canvas-tools) - Outils pour manipuler les conceptions
- [Panneau de contrôle](control-panel) - Contrôle manuel de la machine, état et journaux
- [Prévisualisation 3D](3d-preview) - Visualiser les trajets d'outil en 3D
