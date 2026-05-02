# Vue 3D

La vue 3D te permet de visualiser tes trajets d'outil G-code et de simuler l'exécution
du travail avant de l'envoyer à ta machine.

![Prévisualisation 3D](/screenshots/main-3d.png)

## Ouvrir la vue 3D

Accéde à la vue 3D :

- **Menu** : Affichage → Vue 3D
- **Clavier** : <kbd>F12</kbd>

## Navigation

### Contrôles de la souris

- **Rotation** : Clic gauche et glisser (axe Z), clic milieu et glisser (orbite 3 axes)
- **Panoramique** : <kbd>shift</kbd> + clic milieu et glisser
- **Zoom** : Molette de défilement

### Présélections de vue

Angles de caméra rapides :

- **Dessus** (<kbd>1</kbd>) : Vue à vol d'oiseau
- **Face** (<kbd>2</kbd>) : Élévation avant
- **Droite** (<kbd>3</kbd>) : Vue côté droit
- **Gauche** (<kbd>4</kbd>) : Vue côté gauche
- **Arrière** (<kbd>5</kbd>) : Élévation arrière
- **Isométrique** (<kbd>7</kbd>) : Vue isométrique 3D

## Affichage du système de coordonnées de travail

La vue 3D visualise le système de coordonnées de travail (WCS) actif
différemment du canevas 2D :

### Grille et axes

- **Affichage isolé** : La grille et les axes apparaissent comme si l'origine du WCS était
  l'origine monde
- **Décalage appliqué** : Toute la grille est décalée pour s'aligner avec le décalage
  du WCS sélectionné
- **Étiquettes relatives au WCS** : Les étiquettes de coordonnées affichent les positions relatives à
  l'origine du WCS, pas l'origine machine

Cet affichage « en isolation » facilite la compréhension de l'endroit où ton travail
s'exécutera par rapport au système de coordonnées de travail sélectionné, sans être perturbé
par la position absolue de la machine.

### Changer de WCS

La vue 3D se met à jour automatiquement lorsque tu changes le WCS actif :
- Sélectionne un WCS différent dans la liste déroulante de la barre d'outils
- La grille et les axes se décalent pour refléter la nouvelle origine du WCS
- Les étiquettes se mettent à jour pour afficher les coordonnées relatives au nouveau WCS

:::tip WCS dans la vue 3D
La vue 3D affiche tes trajets d'outil par rapport au WCS sélectionné. Lorsque tu
changes de WCS, tu verras les trajets d'outil sembler bouger parce que le point de référence
(la grille) a changé, non pas parce que les trajets d'outil eux-mêmes ont bougé.
:::


## Options d'affichage

Les bascules de visibilité se trouvent sous forme de boutons superposés en haut à droite
du canevas 3D :

- **Modèle** : Basculer la visibilité du modèle 3D de la machine
- **Déplacements rapides** : Basculer la visibilité des déplacements rapides
- **Zones interdites** : Basculer la visibilité des zones interdites

### Visualisation du trajet d'outil

Personnalise ce que tu vois :

- **Afficher les déplacements rapides** : Afficher les déplacements de déplacement (lignes pointillées)
- **Afficher les déplacements de travail** : Afficher les déplacements de coupe/gravure (lignes pleines)
- **Couleur par opération** : Différentes couleurs pour chaque opération

:::tip Couleurs par Laser
Lors de l'utilisation de machines avec plusieurs têtes laser, chaque laser peut
avoir ses propres couleurs de coupe et de raster configurées dans
[Paramètres Laser](../machine/laser). Cela facilite l'identification du laser
qui effectuera chaque opération.
:::

### Modèle de tête laser

La vue 3D affiche un modèle de ta tête laser qui se déplace le long du trajet d'outil
pendant la simulation. Tu peux attribuer un modèle 3D à chaque tête laser dans la page
[Paramètres Laser](../machine/laser) des Paramètres Machine. L'échelle, la rotation
et la distance focale du modèle peuvent être ajustées pour correspondre à ton
installation physique.

Pendant la simulation, un faisceau laser lumineux est tracé depuis la tête vers le bas
lorsque le laser est actif.

## Simulation

La vue 3D inclut un simulateur intégré avec des contrôles de lecture superposés
en bas du canevas.

### Contrôles de lecture

- **Lecture/Pause** (<kbd>espace</kbd>) : Animer l'exécution du trajet d'outil
- **Avancer/Reculer d'une étape** : Avancer ou reculer d'une opération à la fois
- **Vitesse** : Parcourir les vitesses de lecture (1x, 2x, 4x, 8x, 16x)
- **Curseur de timeline** : Glisser pour naviguer dans le travail

### Visualiseur G-code synchronisé

La simulation reste synchronisée avec le visualiseur G-code dans le panneau inférieur.
Avancer dans la simulation met en surbrillance la ligne correspondante dans le
visualiseur G-code, et cliquer sur une ligne dans le visualiseur G-code fait sauter
la simulation à ce point.

### Visibilité des calques

Bascule la visibilité des calques individuels :

- Clique sur le nom d'un calque pour l'afficher ou le masquer
- Concentre-toi sur des calques spécifiques pour l'inspection

## Liste de vérification

Avant d'envoyer à la machine, vérifie :

- [ ] Le trajet d'outil est complet sans segments manquants
- [ ] Les opérations de gravure s'exécutent avant les coupes
- [ ] Le travail commence à la position attendue
- [ ] Les onglets de maintien sont aux bons emplacements

Certaines vérifications supplémentaires sont effectuées automatiquement.
Lorsque tu exécutes ou exportes un travail, Rayforge effectue des
[vérifications de cohérence](../features/sanity-checks) qui vérifient les
limites de la machine, les contours de la zone de travail et les collisions
avec les zones interdites.

## Conseils de performance

Pour les travaux volumineux ou complexes :

1. Masque les déplacements rapides pour te concentrer sur les déplacements de travail uniquement
2. Réduis le nombre de calques visibles
3. Ferme les autres applications pour libérer des ressources GPU

## Dépannage

### La prévisualisation est vide ou noire

- Vérifie que les opérations sont activées
- Vérifie que les objets ont des opérations attribuées

### Prévisualisation lente ou saccadée

- Masque les déplacements rapides
- Masque les modèles 3D
- Réduis le nombre de calques visibles

---

**Pages connexes :**

- [Systèmes de coordonnées de travail](../general-info/coordinate-systems) - WCS
- [Fenêtre principale](main-window) - Aperçu de l'interface principale
- [Paramètres](settings) - Préférences de l'application
