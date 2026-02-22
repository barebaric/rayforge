# Gravure

Les opérations de gravure remplissent des zones avec des lignes de balayage raster, supportant plusieurs modes pour différents effets de gravure. Des photos en niveaux de gris lisses aux effets de relief 3D, choisissez le mode qui convient le mieux à votre design et matériau.

## Aperçu

Les opérations de gravure :

- Remplissent les formes fermées avec des lignes de balayage
- Supportent plusieurs modes de gravure pour différents effets
- Fonctionnent avec les formes vectorielles et les images bitmap
- Utilisent le balayage bidirectionnel pour la vitesse
- Créent des marques permanentes sur de nombreux matériaux

## Modes de Gravure

### Mode Puissance Variable

Le mode Puissance Variable fait varier la puissance laser continuellement selon la luminosité de l'image, créant une gravure en niveaux de gris lisse avec des transitions graduelles.

**Idéal Pour :**

- Photos et images en niveaux de gris lisses
- Dégradés et transitions naturels
- Portraits et œuvres d'art
- Gravure sur bois et cuir

**Caractéristiques Clés :**

- Modulation continue de la puissance
- Contrôle puissance min/max
- Dégradés lisses
- Meilleure qualité tonale que le tramage

### Mode Puissance Constante

Le mode Puissance Constante grave à pleine puissance, avec un seuil déterminant quels pixels sont gravés. Cela crée des résultats noir/blanc propres.

**Idéal Pour :**

- Texte et logos
- Graphiques à contraste élevé
- Gravures noir/blanc propres
- Formes et motifs simples

**Caractéristiques Clés :**

- Gravure basée sur un seuil
- Sortie de puissance cohérente
- Plus rapide que le mode puissance variable
- Bords nets

### Mode Tramage

Le mode Tramage convertit les images en niveaux de gris en motifs binaires en utilisant des algorithmes de tramage, permettant une gravure photo de haute qualité avec une meilleure reproduction tonale que les méthodes basées sur un simple seuil.

**Idéal Pour :**

- Gravure de photographies sur bois ou cuir
- Création d'œuvres d'art style demi-teinte
- Images avec dégradés lisses
- Quand le raster standard ne capture pas assez de détails

**Caractéristiques Clés :**

- Choix multiples d'algorithmes de tramage
- Meilleure préservation des détails
- Tons continus perçus
- Idéal pour les photographies

### Mode Profondeurs Multiples

Le mode Profondeurs Multiples crée des effets de relief 3D en faisant varier la puissance laser selon la luminosité de l'image, avec des passes multiples pour un creusage plus profond.

**Idéal Pour :**

- Création de portraits et œuvres d'art 3D
- Cartes de terrain et topographiques
- Lithophanes (images 3D transmettant la lumière)
- Logos et designs en relief
- Sculptures en relief

**Caractéristiques Clés :**

- Mappage de profondeur depuis la luminosité de l'image
- Profondeur min/max configurable
- Dégradés lisses
- Passes multiples pour une gravure plus profonde
- Avance en Z entre les passes

## Quand Utiliser la Gravure

Utilisez les opérations de gravure pour :

- Graver du texte et des logos
- Créer des images et photos sur bois/cuir
- Remplir des zones solides avec de la texture
- Marquer des pièces et produits
- Créer des effets de relief 3D
- Œuvres d'art style demi-teinte

**N'utilisez pas la gravure pour :**

- Couper à travers le matériau (utilisez [Contour](contour) à la place)
- Contours précis (le raster crée des zones remplies)
- Travail de ligne fine (les vecteurs sont plus propres)

## Créer une Opération de Gravure

### Étape 1 : Préparer le Contenu

La gravure fonctionne avec :

- **Formes vectorielles** - Remplies avec des lignes de balayage
- **Texte** - Converti en parcours remplis
- **Images** - Converties en niveaux de gris et gravées

### Étape 2 : Ajouter une Opération de Gravure

- **Menu :** Opérations → Ajouter Gravure
- **Raccourci :** <kbd>ctrl+shift+e</kbd>
- **Clic droit :** Menu contextuel → Ajouter Opération → Gravure

### Étape 3 : Choisir le Mode

Sélectionnez le mode de gravure qui convient le mieux à vos besoins :

- **Puissance Variable** - Gravure en niveaux de gris lisse
- **Puissance Constante** - Gravure noir/blanc propre
- **Tramage** - Gravure photo de haute qualité
- **Profondeurs Multiples** - Effets de relief 3D

### Étape 4 : Configurer les Paramètres

![Paramètres d'étape de gravure](/screenshots/step-settings-engrave-general-variable.png)

## Paramètres Courants

### Puissance & Vitesse

**Puissance (%) :**

- Intensité laser pour la gravure
- Puissance plus basse pour un marquage plus léger
- Puissance plus élevée pour une gravure plus profonde

**Vitesse (mm/min) :**

- À quelle vitesse le laser balaie
- Plus rapide = plus clair, plus lent = plus foncé

### Intervalle de Ligne

**Intervalle de Ligne (mm) :**

- Espacement entre les lignes de balayage
- Plus petit = qualité plus élevée, temps de travail plus long
- Plus grand = plus rapide, lignes visibles

| Intervalle | Qualité | Vitesse | Utilisation |
| ---------- | ------- | ------- | ----------- |
| 0.05mm | La plus élevée | La plus lente | Photos, détails fins |
| 0.1mm | Élevée | Moyenne | Texte, logos, graphiques |
| 0.2mm | Moyenne | Rapide | Remplissages solides, textures |
| 0.3mm+ | Basse | La plus rapide | Brouillon, test |

**Recommandé :** 0.1mm pour un usage général

:::tip Correspondance de Résolution
:::

Pour les images, l'intervalle de ligne devrait correspondre ou dépasser la résolution de l'image. Si votre image est de 10 pixels/mm (254 DPI), utilisez un intervalle de ligne de 0.1mm ou moins.

### Direction de Balayage

**Angle de Balayage (degrés) :**

- Direction des lignes de balayage
- 0 = horizontal (de gauche à droite)
- 90 = vertical (de haut en bas)
- 45 = diagonal

**Pourquoi changer l'angle ?**

- Grain du bois : Gravez perpendiculairement au grain pour de meilleurs résultats
- Orientation du motif : Correspondre à l'esthétique du design
- Réduire le banding : Un angle différent peut masquer les imperfections

**Balayage Bidirectionnel :**

- **Activé :** Le laser grave dans les deux directions (plus rapide)
- **Désactivé :** Le laser ne grave que de gauche à droite (plus lent, plus cohérent)

Pour une meilleure qualité, désactivez le bidirectionnel. Pour la vitesse, activez-le.

### Overscan

**Distance d'Overscan (mm) :**

- Jusqu'où au-delà du design le laser voyage avant de faire demi-tour
- Permet au laser d'atteindre sa pleine vitesse avant d'entrer dans le design
- Empêche les marques de brûlure aux débuts/fins de ligne

**Valeurs typiques :**

- 2-5mm pour la plupart des travaux
- Plus grand pour les vitesses élevées

Voir [Overscan](../overscan) pour plus de détails.

## Paramètres Spécifiques au Mode

### Paramètres du Mode Puissance Variable

![Paramètres du mode Puissance Variable](/screenshots/step-settings-engrave-general-variable.png)

**Puissance Min (%) :**

- Puissance laser pour les zones les plus claires (pixels blancs)
- Généralement 0-20%
- Définissez plus élevé pour éviter les zones très peu profondes

**Puissance Max (%) :**

- Puissance laser pour les zones les plus foncées (pixels noirs)
- Généralement 40-80% selon le matériau
- Plus bas = relief subtil, plus haut = profondeur dramatique

**Exemples de Plage de Puissance :**

| Min | Max | Effet |
| --- | --- | ----- |
| 0% | 40% | Relief subtil, léger |
| 10% | 60% | Profondeur moyenne, sûr |
| 20% | 80% | Profond, relief dramatique |

**Inverser :**

- **Off** (défaut) : Blanc = peu profond, Noir = profond
- **On** : Blanc = profond, Noir = peu profond

Utilisez inverser pour les lithophanes (les zones claires doivent être fines) ou le gaufrage (zones en relief).

**Plage de Luminosité :**

Contrôle comment les valeurs de luminosité de l'image sont mappées à la puissance laser. L'histogramme montre la distribution des valeurs de luminosité dans votre image.

- **Niveaux Auto** (défaut) : Ajuste automatiquement les points noir et blanc selon le contenu de l'image. Les valeurs sous le point noir sont traitées comme noires, les valeurs au-dessus du point blanc sont traitées comme blanches. Cela étire le contraste de l'image pour utiliser toute la plage de puissance.
- **Mode Manuel** : Désactivez Niveaux Auto pour définir manuellement les points noir et blanc en glissant les marqueurs sur l'histogramme.

C'est particulièrement utile pour :
- Images à faible contraste nécessitant une amélioration du contraste
- Images avec une plage tonale limitée
- Assurer des résultats cohérents entre différentes images sources

### Paramètres du Mode Puissance Constante

![Paramètres du mode Puissance Constante](/screenshots/step-settings-engrave-general-constant_power.png)

**Seuil (0-255) :**

- Cutoff de luminosité pour la séparation noir/blanc
- Plus bas = plus de noir gravé
- Plus haut = plus de blanc gravé

**Valeurs typiques :**

- 128 (seuil 50% gris)
- Ajustez selon le contraste de l'image

### Paramètres du Mode Tramage

![Paramètres du mode Tramage](/screenshots/step-settings-engrave-general-dither.png)

**Algorithme de Tramage :**

Choisissez l'algorithme qui convient le mieux à votre image et matériau :

| Algorithme | Qualité | Vitesse | Idéal Pour |
| ---------- | ------- | ------- | ---------- |
| Floyd-Steinberg | La plus élevée | La plus lente | Photos, portraits, dégradés lisses |
| Bayer 2x2 | Basse | La plus rapide | Effet demi-teinte grossier |
| Bayer 4x4 | Moyenne | Rapide | Demi-teinte équilibrée |
| Bayer 8x8 | Élevée | Moyenne | Détails fins, motifs subtils |

**Floyd-Steinberg** est le défaut et recommandé pour la plupart des gravures photo. Il utilise la diffusion d'erreur pour distribuer les erreurs de quantification aux pixels voisins, créant des résultats d'apparence naturelle.

**Le tramage Bayer** crée des motifs réguliers qui peuvent produire des effets artistiques ressemblant à l'impression demi-teinte traditionnelle.

### Paramètres du Mode Profondeurs Multiples

![Paramètres du mode Profondeurs Multiples](/screenshots/step-settings-engrave-general-multi_pass.png)

**Nombre de Niveaux de Profondeur :**

- Nombre de niveaux de profondeur discrets
- Plus de niveaux = dégradés plus lisses
- Typique : 5-10 niveaux

**Avance en Z par Niveau (mm) :**

- De combien avancer en Z entre les passes de profondeur
- Crée une profondeur totale plus profonde avec des passes multiples
- Typique : 0.1-0.5mm

**Angle de Rotation par Passe :**

- Degrés pour faire pivoter chaque passe successive
- Crée un effet 3D type croisé
- Typique : 0-45 degrés

**Inverser :**

- **Activé :** Blanc = profond, Noir = peu profond
- **Désactivé :** Noir = profond, Blanc = peu profond

Utilisez inverser pour les lithophanes (les zones claires doivent être fines) ou le gaufrage (zones en relief).

## Conseils & Meilleures Pratiques

![Paramètres de post-traitement de gravure](/screenshots/step-settings-engrave-post.png)

### Sélection du Matériau

**Meilleurs matériaux pour la gravure :**

- Bois (les variations naturelles créent de beaux résultats)
- Cuir (brûle au brun foncé/noir)
- Aluminium anodisé (retire le revêtement, révèle le métal)
- Métaux revêtus (retire la couche de revêtement)
- Certains plastiques (testez d'abord !)

**Matériaux difficiles :**

- Acrylique clair (ne montre pas bien la gravure)
- Métaux sans revêtement (nécessite des composés de marquage spéciaux)
- Verre (nécessite des paramètres/revêtements spéciaux)

### Paramètres de Qualité

**Pour une meilleure qualité :**

- Utilisez un intervalle de ligne plus petit (0.05-0.1mm)
- Désactivez le balayage bidirectionnel
- Augmentez l'overscan (3-5mm)
- Utilisez une puissance plus basse, passes multiples
- Assurez-vous que le matériau est plat et sécurisé

**Pour une gravure plus rapide :**

- Utilisez un intervalle de ligne plus grand (0.15-0.2mm)
- Activez le balayage bidirectionnel
- Overscan minimum (1-2mm)
- Passe unique à puissance plus élevée

### Problèmes Courants

**Marques de brûlure aux fins de ligne :**

- Augmentez la distance d'overscan
- Vérifiez les paramètres d'accélération
- Réduisez légèrement la puissance

**Lignes de balayage visibles :**

- Diminuez l'intervalle de ligne
- Réduisez la puissance (la surbrûlure crée des écarts)
- Vérifiez que le matériau est plat

**Gravure inégale :**

- Assurez-vous que le matériau est plat
- Vérifiez la cohérence de la mise au point
- Vérifiez la stabilité de la puissance laser
- Nettoyez la lentille laser

**Banding (rayures claires/foncées) :**

- Désactivez le balayage bidirectionnel
- Vérifiez la tension des courroies
- Réduisez la vitesse
- Essayez un angle de balayage différent

## Dépannage

### Gravure trop claire

- **Augmentez :** Le paramètre de puissance
- **Diminuez :** Le paramètre de vitesse
- **Vérifiez :** La mise au point est correcte
- **Essayez :** Des passes multiples

### Gravure trop foncée/brûlée

- **Diminuez :** Le paramètre de puissance
- **Augmentez :** Le paramètre de vitesse
- **Augmentez :** L'intervalle de ligne
- **Vérifiez :** Le matériau est approprié

### Obscurité incohérente

- **Vérifiez :** Le matériau est plat
- **Vérifiez :** La distance de mise au point est cohérente
- **Vérifiez :** Le faisceau laser est propre
- **Testez :** Différente zone du matériau (le grain varie)

### L'image semble pixelisée

- **Diminuez :** L'intervalle de ligne
- **Vérifiez :** La résolution de l'image source
- **Essayez :** Un intervalle de ligne plus petit (0.05mm)
- **Vérifiez :** L'image n'est pas suréchantillonnée

### Lignes de balayage visibles

- **Diminuez :** L'intervalle de ligne
- **Réduisez :** La puissance (la surbrûlure crée des écarts)
- **Essayez :** Un angle de balayage différent
- **Assurez-vous :** La surface du matériau est lisse

## Sujets Connexes

- **[Coupe de Contour](contour)** - Contours et formes de coupe
- **[Overscan](../overscan)** - Améliorer la qualité de gravure
- **[Grille de Test de Matériau](material-test-grid)** - Trouver les paramètres optimaux
- **[Flux de Travail Multi-Couches](../multi-layer)** - Combiner gravure avec d'autres opérations
