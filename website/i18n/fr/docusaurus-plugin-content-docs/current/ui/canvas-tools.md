# Outils du canevas

Le canevas fournit un ensemble complet d'outils pour manipuler les conceptions, mesurer et préparer vos travaux laser.

## Outil de sélection

Sélectionnez, déplacez et transformez des objets sur le canevas.

**Utilisation :**

- Cliquez pour sélectionner un seul objet
- <kbd>ctrl</kbd> + clic pour sélectionner plusieurs objets
- Glissez pour créer une boîte de sélection
- Cliquez et faites glisser les objets sélectionnés pour les déplacer

**Poignées de transformation :**

- **Poignées d'angle** : Mise à l'échelle proportionnelle (maintenez <kbd>shift</kbd> pour mise à l'échelle non proportionnelle)
- **Poignées de bord** : Mise à l'échelle dans une direction
- **Poignée de rotation** : Rotation autour du point central

**Raccourcis clavier :**

- <kbd>touches-fléchées</kbd> : Déplacer les objets sélectionnés de 1 unité
- <kbd>shift+touches-fléchées</kbd> : Déplacer de 10 unités
- <kbd>ctrl+d</kbd> : Dupliquer la sélection

## Outil Panoramique

Naviguez sur le canevas sans déplacer accidentellement les objets.

**Utilisation :**

- Cliquez et faites glisser pour panorama
- Alternativement, maintenez <kbd>espace</kbd> avec n'importe quel outil actif et faites glisser

## Outil Zoom

Zoomer sur des zones spécifiques de votre conception.

**Utilisation :**

- Cliquez pour zoomer à un point
- <kbd>alt</kbd> + clic pour dézoomer
- Cliquez et faites glisser pour zoomer sur une zone spécifique

**Raccourcis :**

- <kbd>ctrl+"+"</kbd> : Zoomer
- <kbd>ctrl+"-"</kbd> : Dézoomer
- <kbd>ctrl+0</kbd> : Réinitialiser le zoom (ajuster à la fenêtre)
- Molette de souris : Zoomer/dézoomer à la position du curseur

## Outil de mesure

Mesurez des distances et des angles sur le canevas.

**Utilisation :**

- Cliquez sur le point de départ
- Cliquez sur le point d'arrivée pour terminer la mesure
- La mesure s'affiche dans les unités actuelles (mm ou pouces)

**Fonctionnalités :**

- Distance entre deux points
- Angle relatif à l'horizontale
- Aperçu en temps réel pendant la mesure

## Outils d'alignement

Alignez et distribuez plusieurs objets.

**Options d'alignement :**

- Aligner les bords gauches
- Aligner les bords droits
- Aligner les bords supérieurs
- Aligner les bords inférieurs
- Aligner les centres horizontaux
- Aligner les centres verticaux

**Options de distribution :**

- Distribuer horizontalement
- Distribuer verticalement
- Espacement égal

**Utilisation :**

1. Sélectionnez plusieurs objets
2. Choisissez l'option d'alignement/distribution dans la barre d'outils ou le menu Édition
3. Les objets s'alignent/se distribuent immédiatement

## Grille et aimantation

Aidez le positionnement précis avec la grille et l'aimantation.

**Grille :**

- Basculer : Affichage → Afficher la grille (<kbd>ctrl+g</kbd>)
- Configurez l'espacement dans les Préférences
- Guide visuel uniquement (aimantation optionnelle)

**Aimantation :**

- **Aimanter à la grille** : Aligner les objets sur les points de grille
- **Aimanter aux objets** : Aligner sur les bords d'autres objets
- **Aimanter à l'origine** : Aligner sur l'origine machine (0,0)

Basculer l'aimantation : Affichage → Aimanter (<kbd>ctrl+shift+g</kbd>)

## Transformation d'objet

Transformez les objets numériquement pour plus de précision.

**Accessible via le panneau Propriétés :**

- **Position (X, Y)** : Coordonnées exactes
- **Taille (L, H)** : Dimensions exactes
- **Rotation** : Degrés par rapport à l'horizontale
- **Échelle** : Pourcentage de la taille originale

## Opérations booléennes

Combinez ou soustrayez des formes :

- **Union** : Fusionner les formes qui se chevauchent
- **Différence** : Soustraire une forme d'une autre
- **Intersection** : Garder uniquement les zones qui se chevauchent
- **Exclusion** : Supprimer les zones qui se chevauchent

**Utilisation :**

1. Sélectionnez deux objets vectoriels ou plus
2. Choisissez l'opération booléenne dans le menu Édition
3. Le résultat remplace les objets sélectionnés

## Conseils pour une utilisation efficace du canevas

1. **Utilisez les raccourcis clavier** : Beaucoup plus rapide que les clics sur la barre d'outils
2. **Maîtrisez le panoramique et le zoom** : Essentiel pour les conceptions volumineuses ou détaillées
3. **Aimanter à la grille** : Accélère l'alignement pour les dispositions rectangulaires
4. **Mesurez d'abord** : Vérifiez les dimensions avant de générer le G-code
5. **Groupez les objets connexes** : Plus facile à déplacer et organiser (<kbd>ctrl+g</kbd> pour grouper)

---

**Suivant** : [Prévisualisation 3D →](3d-preview)
