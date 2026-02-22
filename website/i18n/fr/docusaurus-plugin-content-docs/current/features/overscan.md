# Overscan

L'overscan étend les lignes de gravure raster au-delà de la zone de contenu réelle pour s'assurer que le laser atteint une vitesse constante pendant la gravure, éliminant les artefacts d'accélération.

## Le Problème : Marques d'Accélération

Sans overscan, la gravure raster souffre d'**artefacts d'accélération** :

- **Bords clairs** où l'accélération commence (laser se déplaçant trop vite pour le niveau de puissance)
- **Bords sombres** où la décélération se produit (laser s'attardant plus longtemps)
- **Profondeur/foncé de gravure incohérent** à travers chaque ligne
- Bandes ou rayures visibles aux bords des lignes

## Comment Fonctionne l'Overscan

L'overscan **étend le parcours d'outil** avant et après chaque ligne raster :

**Processus :**

1. **Entrée :** Le laser se déplace à une position _avant_ le début de la ligne
2. **Accélère :** Le laser accélère à la vitesse cible (laser ÉTEINT)
3. **Grave :** Le laser s'allume et grave à vitesse constante
4. **Décélère :** Le laser s'éteint et décélère _après_ la fin de la ligne

**Résultat :** Toute la zone gravée reçoit une puissance constante à vitesse constante.

**Avantages :**

- Profondeur de gravure uniforme à travers toute la ligne raster
- Pas de bords clairs/sombres
- Gravure photo de meilleure qualité
- Résultats d'apparence professionnelle

## Configurer l'Overscan

L'overscan est un **transformateur** dans le pipeline de flux de travail Rayforge.

**Pour activer :**

1. **Sélectionnez le calque** avec la gravure raster
2. **Ouvrez les paramètres de flux de travail** (ou paramètres d'opération)
3. **Ajoutez le transformateur Overscan** s'il n'est pas déjà présent
4. **Configurez la distance**

**Paramètres :**

| Paramètre | Description | Valeur Typique |
| --------- | ----------- | -------------- |
| **Activé** | Basculer overscan on/off | ON (pour raster) |
| **Distance (mm)** | Jusqu'où étendre les lignes | 2-5 mm |

## Choisir la Distance d'Overscan

La distance d'overscan devrait permettre à la machine d'**accélérer complètement** à la vitesse cible.

**Directives pratiques :**

| Vitesse Max | Accélération | Overscan Recommandé |
| ----------- | ------------ | ------------------- |
| 3000 mm/min (50 mm/s) | Faible | 5 mm |
| 3000 mm/min (50 mm/s) | Moyenne | 3 mm |
| 3000 mm/min (50 mm/s) | Élevée | 2 mm |
| 6000 mm/min (100 mm/s) | Faible | 10 mm |
| 6000 mm/min (100 mm/s) | Moyenne | 6 mm |
| 6000 mm/min (100 mm/s) | Élevée | 4 mm |

**Facteurs affectant la distance requise :**

- **Vitesse :** Vitesse plus élevée = besoin de plus de distance pour accélérer
- **Accélération :** Accélération plus faible = besoin de plus de distance
- **Mécanique machine :** Transmission par courroie vs direct-drive affecte l'accélération

**Réglage :**

- **Trop peu :** Marques d'accélération encore visibles aux bords
- **Trop :** Perte de temps, peut atteindre les limites de la machine
- **Commencez avec 3mm** et ajustez selon les résultats

## Tester les Paramètres d'Overscan

**Procédure de test :**

1. **Créez une gravure de test :**
   - Rectangle plein (50mm x 20mm)
   - Utilisez vos paramètres de gravure typiques
   - Activez overscan à 3mm

2. **Gravez le test :**
   - Exécutez le travail
   - Laissez terminer

3. **Examinez les bords :**
   - Regardez les bords gauche et droit du rectangle
   - Vérifiez la variation de foncé aux bords
   - Comparez le foncé des bords au foncé du centre

4. **Ajustez :**
   - **Si les bords sont plus clairs/foncés :** Augmentez l'overscan
   - **Si les bords correspondent au centre :** L'overscan est suffisant
   - **Si les bords sont parfaits :** Essayez de réduire légèrement l'overscan pour gagner du temps

## Quand Utiliser l'Overscan

**Utilisez toujours pour :**

- Gravure photo (raster)
- Motifs de remplissage
- Tout travail raster à haute précision
- Gravure d'image en niveaux de gris
- Gravure de texte (mode raster)

**Optionnel pour :**

- Coupe vectorielle (non nécessaire)
- Gravure très lente (accélération moins noticeable)
- Grandes formes simples (bords moins critiques)

**Désactivez pour :**

- Opérations vectorielles
- Très petites zones de travail (peut dépasser les limites)
- Lorsque la qualité des bords n'est pas importante

---

## Sujets Connexes

- [Opérations de Gravure](./operations/engrave) - Configurer les paramètres de gravure
- [Grille de Test de Matériau](./operations/material-test-grid) - Trouver la puissance/vitesse optimales
