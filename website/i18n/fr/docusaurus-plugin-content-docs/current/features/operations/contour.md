# Coupe de Contour

La coupe de contour trace le contour des formes vectorielles pour les couper librement du matériau. C'est l'opération laser la plus courante pour créer des pièces, des enseignes et des pièces décoratives.

## Aperçu

Les opérations de contour :

- Suivent les parcours vectoriels (lignes, courbes, formes)
- Coupe le long du périmètre des objets
- Supportent des passes uniques ou multiples pour les matériaux épais
- Peuvent utiliser des parcours de coupe intérieur, extérieur ou sur la ligne
- Fonctionnent avec toute forme vectorielle fermée ou ouverte


## Quand Utiliser le Contour

Utilisez la coupe de contour pour :

- Couper des pièces libres du matériau de stock
- Créer des contours et bordures
- Couper des formes dans le bois, l'acrylique, le carton
- Percer ou marquer (avec puissance réduite)
- Créer des pochoirs et modèles

**N'utilisez pas le contour pour :**

- Remplir des zones (utilisez [Gravure](engrave) à la place)
- Les images bitmap (convertissez d'abord en vecteurs)

## Créer une Opération de Contour

### Étape 1 : Sélectionner les Objets

1. Importez ou dessinez des formes vectorielles sur le canevas
2. Sélectionnez les objets que vous voulez couper
3. Assurez-vous que les formes sont des parcours fermés pour des coupes complètes

### Étape 2 : Ajouter une Opération de Contour

- **Menu :** Opérations Ajouter Contour
- **Raccourci :** <kbd>ctrl+shift+c</kbd>
- **Clic droit :** Menu contextuel Ajouter Opération Contour

### Étape 3 : Configurer les Paramètres

![Paramètres d'étape de contour](/screenshots/step-settings-contour-general.png)

## Paramètres Clés

### Puissance & Vitesse

**Puissance (%) :**

- Intensité laser de 0-100%
- Puissance plus élevée pour les matériaux plus épais
- Puissance plus basse pour le marquage ou le scorring

**Vitesse (mm/min) :**

- À quelle vitesse le laser se déplace
- Plus lent = plus d'énergie = coupe plus profonde
- Plus rapide = moins d'énergie = coupe plus légère

### Coupe Multi-Passes

Pour les matériaux plus épais qu'une seule passe peut couper :

**Passes :**

- Nombre de fois pour répéter la coupe
- Chaque passe coupe plus profond

**Profondeur de Passe (Z-step) :**

- De combien abaisser l'axe Z par passe (si supporté)
- Nécessite le contrôle de l'axe Z sur votre machine
- Crée une vraie coupe 2.5D
- Définissez à 0 pour des passes multiples à la même profondeur

:::warning Axe Z Requis
:::

La profondeur de passe fonctionne uniquement si votre machine a le contrôle de l'axe Z. Pour les machines sans axe Z, utilisez des passes multiples à la même profondeur.

### Décalage de Parcours

Contrôle où le laser coupe par rapport au parcours vectoriel :

| Décalage | Description | Utilisation |
| -------- | ----------- | ----------- |
| **Sur la Ligne** | Coupe directement sur le parcours | Coupes sur la ligne centrale, marquage |
| **Intérieur** | Coupe à l'intérieur de la forme | Pièces qui doivent correspondre à la taille exacte |
| **Extérieur** | Coupe à l'extérieur de la forme | Trous dans lesquels les pièces s'insèrent |

**Distance de Décalage :**

- Jusqu'où décaler vers l'intérieur/extérieur (mm)
- Typiquement défini à la moitié de votre largeur de kerf
- Kerf = largeur du matériau retiré par le laser
- Exemple : décalage de 0.15mm pour un kerf de 0.3mm

### Direction de Coupe

**Horaire vs Anti-Horaire :**

- Affecte quel côté de la coupe reçoit plus de chaleur
- Généralement horaire pour la règle de la main droite
- Changez si un côté brûle plus que l'autre

**Optimiser l'Ordre :**

- Trie automatiquement les parcours pour un déplacement minimum
- Réduit le temps de travail
- Empêche les coupes manquées

## Fonctionnalités Avancées

![Paramètres de post-traitement du contour](/screenshots/step-settings-contour-post.png)

### Ponts de Maintien

Les ponts gardent les pièces coupées attachées au matériau de stock pendant la coupe :

- Ajoutez des ponts pour empêcher les pièces de tomber
- Les ponts sont de petites sections non coupées
- Cassez les ponts après l'achèvement du travail
- Voir [Ponts de Maintien](../holding-tabs) pour plus de détails

### Compensation de Kerf

Le kerf est la largeur du matériau retiré par le faisceau laser :

**Pourquoi c'est important :**

- Un cercle coupé "sur la ligne" sera légèrement plus petit que conçu
- Le laser retire ~0.2-0.4mm de matériau (selon la largeur du faisceau)

**Comment compenser :**

1. Mesurez votre kerf sur des tests de coupe
2. Utilisez le décalage de parcours = kerf/2
3. Pour les pièces : décalez **vers l'intérieur** de kerf/2
4. Pour les trous : décalez **vers l'extérieur** de kerf/2

Voir [Kerf](../kerf) pour un guide détaillé.

### Entrée/Sortie

Les entrées et sorties contrôlent où les coupes commencent et finissent :

**Entrée :**

- Entrée progressive dans la coupe
- Empêche les marques de brûlure au point de départ
- Déplace le laser à pleine vitesse avant de toucher le bord du matériau

**Sortie :**

- Sortie progressive de la coupe
- Empêche les dommages au point de fin
- Courant pour les métaux et acryliques

**Configuration :**

- Longueur : Jusqu'où s'étend l'entrée (mm)
- Angle : Direction du parcours d'entrée
- Type : Ligne droite, arc ou spirale

## Conseils & Meilleures Pratiques

### Test de Matériau

**Testez toujours d'abord :**

1. Coupez de petites formes de test sur du rebut
2. Commencez avec des paramètres conservateurs (puissance plus basse, vitesse plus lente)
3. Augmentez progressivement la puissance ou diminuez la vitesse
4. Enregistrez les paramètres réussis

### Ordre de Coupe

**Meilleures pratiques :**

- Gravez avant de couper (garde le matériau sécurisé)
- Coupez les caractéristiques intérieures avant le périmètre extérieur
- Utilisez des ponts de maintien pour les pièces qui pourraient bouger
- Coupez les plus petites pièces d'abord (moins de vibrations)

## Dépannage

### Les coupes ne traversent pas le matériau

- **Augmentez :** Le paramètre de puissance
- **Diminuez :** Le paramètre de vitesse
- **Ajoutez :** Plus de passes
- **Vérifiez :** La mise au point est correcte
- **Vérifiez :** Le faisceau est propre (lentille sale)

### Brunissage ou brûlure excessifs

- **Diminuez :** Le paramètre de puissance
- **Augmentez :** Le paramètre de vitesse
- **Utilisez :** L'assistance air
- **Essayez :** Passes multiples plus rapides au lieu d'une lente
- **Vérifiez :** Le matériau est approprié pour la découpe laser

### Les pièces tombent pendant la coupe

- **Ajoutez :** [Ponts de maintien](../holding-tabs)
- **Utilisez :** L'optimisation de l'ordre de coupe
- **Coupez :** Les caractéristiques intérieures avant l'extérieur
- **Assurez-vous :** Le matériau est plat et sécurisé

### Profondeur de coupe incohérente

- **Vérifiez :** L'épaisseur du matériau est uniforme
- **Vérifiez :** Le matériau est plat (pas gondolé)
- **Vérifiez :** La distance de mise au point est cohérente
- **Vérifiez :** La puissance laser est stable

### Coins ou courbes manqués

- **Diminuez :** La vitesse (surtout sur les coins)
- **Vérifiez :** Les paramètres d'accélération machine
- **Vérifiez :** Les courroies sont tendues
- **Réduisez :** La complexité du parcours (simplifiez les courbes)

## Détails Techniques

### Système de Coordonnées

Les opérations de contour fonctionnent en :

- **Unités :** Millimètres (mm)
- **Origine :** Dépend de la configuration de la machine et du travail
- **Coordonnées :** Plan X/Y (Z pour la profondeur multi-passe)

### Génération de Parcours

Rayforge convertit les formes vectorielles en G-code :

1. Décalage du parcours (si coupe intérieur/extérieur)
2. Optimisation de l'ordre du parcours (minimiser le déplacement)
3. Insertion des entrées/sorties (si configurées)
4. Ajout des ponts de maintien (si configurés)
5. Génération des commandes G-code

### Commandes G-code

G-code de contour typique :

```gcode
G0 X10 Y10          ; Déplacement rapide au départ
M3 S204             ; Laser activé à 80% puissance
G1 X50 Y10 F500     ; Coupe vers le point à 500 mm/min
G1 X50 Y50 F500     ; Coupe vers le point suivant
G1 X10 Y50 F500     ; Continue la coupe
G1 X10 Y10 F500     ; Complète le carré
M5                  ; Laser éteint
```

## Sujets Connexes

- **[Gravure](engrave)** - Remplir des zones avec des motifs de gravure
- **[Ponts de Maintien](../holding-tabs)** - Maintenir les pièces sécurisées pendant la coupe
- **[Kerf](../kerf)** - Améliorer la précision de coupe
- **[Grille de Test de Matériau](material-test-grid)** - Trouver les paramètres puissance/vitesse optimaux
