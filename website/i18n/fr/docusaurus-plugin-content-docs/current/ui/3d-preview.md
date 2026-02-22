# Prévisualisation 3D

La fenêtre de prévisualisation 3D vous permet de visualiser vos trajets d'outil G-code avant
de les envoyer à votre machine. Cette fonctionnalité puissante vous aide à détecter les erreurs
et à vérifier la configuration de votre travail.

![Prévisualisation 3D](/screenshots/main-3d.png)

## Ouvrir la prévisualisation 3D

Accédez à la prévisualisation 3D :

- **Menu** : Affichage → Prévisualisation 3D
- **Clavier** : <kbd>ctrl+3</kbd>
- **Après génération du G-code** : S'ouvre automatiquement (configurable)

## Navigation

### Contrôles de la souris

- **Rotation** : Clic gauche et glisser
- **Panoramique** : Clic droit et glisser, ou clic milieu et glisser
- **Zoom** : Molette de défilement, ou <kbd>ctrl</kbd> + clic gauche et glisser

### Contrôles clavier

- <kbd>r</kbd> : Réinitialiser la caméra à la vue par défaut
- <kbd>origine</kbd> : Réinitialiser le zoom et la position
- <kbd>f</kbd> : Ajuster la vue au trajet d'outil
- Touches fléchées : Rotation de la caméra

### Présélections de vue

Angles de caméra rapides :

- **Dessus** (<kbd>1</kbd>) : Vue à vol d'oiseau
- **Face** (<kbd>2</kbd>) : Élévation avant
- **Droite** (<kbd>3</kbd>) : Élévation côté droit
- **Isométrique** (<kbd>4</kbd>) : Vue isométrique 3D

## Affichage du système de coordonnées de travail

La prévisualisation 3D visualise le système de coordonnées de travail (WCS) actif
différemment du canevas 2D :

### Grille et axes

- **Affichage isolé** : La grille et les axes apparaissent comme si l'origine du WCS était
  l'origine monde
- **Décalage appliqué** : Toute la grille est décalée pour s'aligner avec le décalage
  du WCS sélectionné
- **Étiquettes relatives au WCS** : Les étiquettes de coordonnées affichent les positions relatives à
  l'origine du WCS, pas l'origine machine

Cet affichage "en isolation" facilite la compréhension de l'endroit où votre travail
s'exécutera par rapport au système de coordonnées de travail sélectionné, sans être confus
par la position absolue de la machine.

### Changer de WCS

La prévisualisation 3D se met à jour automatiquement lorsque vous changez le WCS actif :
- Sélectionnez un WCS différent dans la liste déroulante de la barre d'outils
- La grille et les axes se décalent pour refléter la nouvelle origine du WCS
- Les étiquettes se mettent à jour pour afficher les coordonnées relatives au nouveau WCS

:::tip WCS dans la prévisualisation 3D
La prévisualisation 3D affiche vos trajets d'outil par rapport au WCS sélectionné. Lorsque vous
changez de WCS, vous verrez les trajets d'outil sembler bouger parce que le point de référence
(la grille) a changé, non pas parce que les trajets d'outil eux-mêmes ont bougé.
:::


## Options d'affichage

### Visualisation du trajet d'outil

Personnalisez ce que vous voyez :

- **Afficher les déplacements rapides** : Afficher les déplacements de déplacement (lignes pointillées)
- **Afficher les déplacements de travail** : Afficher les déplacements de coupe/gravure (lignes pleines)
- **Couleur par opération** : Différentes couleurs pour chaque opération
- **Couleur par puissance** : Dégradé basé sur la puissance laser
- **Couleur par vitesse** : Dégradé basé sur la vitesse d'avance

### Visualisation de la machine

- **Afficher l'origine** : Afficher le point de référence (0,0)
- **Afficher la zone de travail** : Afficher les limites de la machine
- **Afficher la tête laser** : Afficher l'indicateur de position actuelle

### Paramètres de qualité

- **Largeur de ligne** : Épaisseur des lignes de trajet d'outil
- **Anti-crénelage** : Rendu de ligne lisse (peut impacter les performances)
- **Arrière-plan** : Clair, sombre, ou couleur personnalisée

## Contrôles de lecture

Simulez l'exécution du travail :

- **Lecture/Pause** (<kbd>espace</kbd>) : Animer l'exécution du trajet d'outil
- **Vitesse** : Ajuster la vitesse de lecture (0.5x - 10x)
- **Avancer/Reculer** : Avancer par commandes G-code individuelles
- **Aller à la position** : Cliquer sur la timeline pour aller à un point spécifique

### Timeline

La timeline affiche :

- Position actuelle dans le travail
- Limites des opérations (segments colorés)
- Temps estimé à n'importe quel point

## Outils d'analyse

### Mesure de distance

Mesurez des distances en 3D :

1. Activez l'outil de mesure
2. Cliquez sur deux points du trajet d'outil
3. Visualisez la distance dans les unités actuelles

### Panneau de statistiques

Visualisez les statistiques du travail :

- **Distance totale** : Somme de tous les mouvements
- **Distance de travail** : Distance de coupe/gravure uniquement
- **Distance rapide** : Déplacements de déplacement uniquement
- **Temps estimé** : Estimation de la durée du travail
- **Boîte englobante** : Dimensions globales

### Visibilité des calques

Basculez la visibilité des opérations :

- Cliquez sur le nom de l'opération pour afficher/masquer
- Concentrez-vous sur des opérations spécifiques pour l'inspection
- Isolez les problèmes sans régénérer le G-code

## Liste de vérification

Avant d'envoyer à la machine, vérifiez :

- [ ] **Le trajet d'outil est complet** : Pas de segments manquants
- [ ] **Dans la zone de travail** : Reste à l'intérieur des limites de la machine
- [ ] **Ordre des opérations correct** : Graver avant de couper
- [ ] **Pas de collisions** : La tête ne heurte pas les brides/fixations
- [ ] **Origine appropriée** : Commence à la position attendue
- [ ] **Positions des onglets** : Onglets de maintien aux bons emplacements (si utilisés)

## Conseils de performance

Pour les travaux volumineux ou complexes :

1. **Réduisez le détail des lignes** : Baissez la qualité d'affichage pour un rendu plus rapide
2. **Masquez les déplacements rapides** : Concentrez-vous sur les déplacements de travail uniquement
3. **Désactivez l'anti-crénelage** : Améliore le taux de rafraîchissement
4. **Fermez les autres applications** : Libérez des ressources GPU

## Dépannage

### La prévisualisation est vide ou noire

- Régénérez le G-code (<kbd>ctrl+g</kbd>)
- Vérifiez que les opérations sont activées
- Vérifiez que les objets ont des opérations attribuées

### Prévisualisation lente ou saccadée

- Réduisez la largeur de ligne
- Désactivez l'anti-crénelage
- Masquez les déplacements rapides
- Mettez à jour les pilotes graphiques

### Les couleurs ne s'affichent pas correctement

- Vérifiez le paramètre de couleur par (opération/puissance/vitesse)
- Assurez-vous que les opérations ont des couleurs différentes attribuées
- Réinitialisez les paramètres de vue aux valeurs par défaut

---

**Pages connexes :**

- [Systèmes de coordonnées de travail](../general-info/work-coordinate-systems) - WCS
- [Fenêtre principale](main-window) - Aperçu de l'interface principale
- [Paramètres](settings) - Préférences de l'application
