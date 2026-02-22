# Grille de Test de Matériau

Le générateur de Grille de Test de Matériau crée des motifs de test paramétriques pour vous aider à trouver les paramètres laser optimaux pour différents matériaux.

## Aperçu

Les tests de matériau sont essentiels pour le travail laser - différents matériaux nécessitent différents paramètres de puissance et vitesse. La Grille de Test de Matériau automatise ce processus en :

- Générant des grilles de test avec des plages de vitesse/puissance configurables
- Fournissant des préréglages pour les types de laser courants (Diode, CO2)
- Optimisant l'ordre d'exécution pour la sécurité (vitesses les plus rapides d'abord)
- Ajoutant des étiquettes pour identifier les paramètres de chaque cellule de test

## Créer une Grille de Test de Matériau

### Étape 1 : Ouvrir le Générateur

Accédez au générateur de Grille de Test de Matériau :

- Menu : **Outils → Grille de Test de Matériau**
- Cela crée une pièce spéciale qui génère le motif de test

### Étape 2 : Choisir un Préréglage (Optionnel)

Rayforge inclut des préréglages pour des scénarios courants :

| Préréglage | Plage de Vitesse | Plage de Puissance | Utilisation |
| ---------- | ---------------- | ------------------ | ----------- |
| **Gravure Diode** | 1000-10000 mm/min | 10-100% | Gravure laser diode |
| **Coupe Diode** | 100-5000 mm/min | 50-100% | Coupe laser diode |
| **Gravure CO2** | 3000-20000 mm/min | 10-50% | Gravure laser CO2 |
| **Coupe CO2** | 1000-20000 mm/min | 30-100% | Coupe laser CO2 |

Les préréglages sont des points de départ - vous pouvez ajuster tous les paramètres après en avoir sélectionné un.

### Étape 3 : Configurer les Paramètres

Ajustez les paramètres de la grille de test dans la boîte de dialogue des paramètres :

![Paramètres Grille de Test de Matériau](/screenshots/material-test-grid.png)

#### Type de Test

- **Gravure** : Remplit les carrés avec un motif raster
- **Coupe** : Coupe le contour des carrés

#### Plage de Vitesse

- **Vitesse Min** : Vitesse la plus lente à tester (mm/min)
- **Vitesse Max** : Vitesse la plus rapide à tester (mm/min)
- Les colonnes dans la grille représentent différentes vitesses

#### Plage de Puissance

- **Puissance Min** : Puissance la plus basse à tester (%)
- **Puissance Max** : Puissance la plus élevée à tester (%)
- Les lignes dans la grille représentent différents niveaux de puissance

#### Dimensions de la Grille

- **Colonnes** : Nombre de variations de vitesse (typiquement 3-7)
- **Lignes** : Nombre de variations de puissance (typiquement 3-7)

#### Taille & Espacement

- **Taille de la Forme** : Taille de chaque carré de test en mm (défaut : 20mm)
- **Espacement** : Espace entre les carrés en mm (défaut : 5mm)

#### Étiquettes

- **Inclure les Étiquettes** : Activer/désactiver les étiquettes d'axe montrant les valeurs de vitesse et puissance
- Les étiquettes apparaissent sur les bords gauche et supérieur
- Les étiquettes sont gravées à 10% puissance, 1000 mm/min

### Étape 4 : Générer la Grille

Cliquez sur **Générer** pour créer le motif de test. La grille apparaît sur votre canevas comme une pièce spéciale.

## Comprendre la Disposition de la Grille

### Organisation de la Grille

```
Puissance (%)     Vitesse (mm/min) →
    ↓      1000   2500   5000   7500   10000
  100%     [  ]   [  ]   [  ]   [  ]   [  ]
   75%     [  ]   [  ]   [  ]   [  ]   [  ]
   50%     [  ]   [  ]   [  ]   [  ]   [  ]
   25%     [  ]   [  ]   [  ]   [  ]   [  ]
   10%     [  ]   [  ]   [  ]   [  ]   [  ]
```

- **Colonnes** : La vitesse augmente de gauche à droite
- **Lignes** : La puissance augmente de bas en haut
- **Étiquettes** : Montrent les valeurs exactes pour chaque ligne/colonne

### Calcul de la Taille de Grille

**Sans étiquettes :**

- Largeur = colonnes × (taille_forme + espacement) - espacement
- Hauteur = lignes × (taille_forme + espacement) - espacement

**Avec étiquettes :**

- Ajoutez une marge de 15mm à gauche et en haut pour l'espace des étiquettes

**Exemple :** Grille 5×5 avec carrés de 20mm et espacement de 5mm :

- Sans étiquettes : 120mm × 120mm
- Avec étiquettes : 135mm × 135mm

## Ordre d'Exécution (Optimisation du Risque)

Rayforge exécute les cellules de test dans un **ordre optimisé par risque** pour prévenir les dommages au matériau :

1. **Vitesse la plus élevée d'abord** : Les vitesses rapides sont plus sûres (moins d'accumulation de chaleur)
2. **Puissance la plus basse à l'intérieur de la vitesse** : Minimise le risque à chaque niveau de vitesse

Cela prévient le brunissage ou le feu de commencer avec des combinaisons lentes et à haute puissance.

**Exemple d'ordre d'exécution pour une grille 3×3 :**

```
Ordre :  1  2  3
         4  5  6  ← Vitesse la plus élevée, puissance croissante
         7  8  9

(Vitesse la plus rapide/puissance la plus basse exécutée en premier)
```

## Utiliser les Résultats de Test de Matériau

### Étape 1 : Exécuter le Test

1. Chargez votre matériau dans le laser
2. Faites la mise au point du laser correctement
3. Exécutez le travail de grille de test de matériau
4. Surveillez le test - arrêtez si une cellule cause des problèmes

### Étape 2 : Évaluer les Résultats

Après l'achèvement du test, examinez chaque cellule :

- **Trop clair** : Augmentez la puissance ou diminuez la vitesse
- **Trop foncé/brûlé** : Diminuez la puissance ou augmentez la vitesse
- **Parfait** : Notez la combinaison vitesse/puissance

### Étape 3 : Enregistrer les Paramètres

Documentez vos paramètres réussis pour référence future :

- Type et épaisseur du matériau
- Type d'opération (gravure ou coupe)
- Combinaison vitesse et puissance
- Nombre de passes
- Toutes notes spéciales

:::tip Base de Données Matériaux
Envisagez de créer un document de référence avec vos résultats de test de matériau pour une recherche rapide dans les projets futurs.
:::

## Utilisation Avancée

### Combiner avec d'Autres Opérations

Les grilles de test de matériau sont des pièces régulières - vous pouvez les combiner avec d'autres opérations :

**Exemple de flux de travail :**

1. Créez une grille de test de matériau
2. Ajoutez une coupe de contour autour de toute la grille
3. Exécutez le test, coupez libre, évaluez les résultats

C'est utile pour couper la pièce de test libre du matériau de stock.

### Plages de Test Personnalisées

Pour un réglage fin, créez des tests à plage étroite :

**Test grossier** (trouver l'à peu près) :

- Vitesse : 1000-10000 mm/min (5 colonnes)
- Puissance : 10-100% (5 lignes)

**Test fin** (optimiser) :

- Vitesse : 4000-6000 mm/min (5 colonnes)
- Puissance : 35-45% (5 lignes)

### Différents Matériaux, Même Grille

Exécutez la même configuration de grille sur différents matériaux pour construire votre bibliothèque de matériaux plus rapidement.

## Conseils & Meilleures Pratiques

### Conception de Grille

✅ **Commencez avec les préréglages** - Bons points de départ pour les scénarios courants
✅ **Utilisez des grilles 5×5** - Bon équilibre entre détail et temps de test
✅ **Activez les étiquettes** - Essentiel pour identifier les résultats
✅ **Gardez les carrés ≥20mm** - Plus facile à voir et mesurer les résultats

### Stratégie de Test

✅ **Testez sur du rebut d'abord** - Ne testez jamais sur le matériau final
✅ **Une variable à la fois** - Testez la plage de vitesse OU de puissance, pas les deux extrêmes
✅ **Permettez le refroidissement** - Attendez entre les tests sur le même matériau
✅ **Mise au point cohérente** - Même distance de mise au point pour tous les tests

### Sécurité

⚠️ **Surveillez les tests** - Ne laissez jamais les tests en cours sans surveillance
⚠️ **Commencez de manière conservatrice** - Commencez avec des plages de puissance plus basses
⚠️ **Vérifiez la ventilation** - Assurez une extraction des fumées appropriée
⚠️ **Surveillez le feu** - Ayez un extincteur prêt

## Dépannage

### Les cellules de test s'exécutent dans le mauvais ordre

- Rayforge utilise l'ordre optimisé par risque (vitesses les plus rapides d'abord)
- C'est intentionnel et ne peut pas être changé
- Voir [Ordre d'Exécution](#ordre-dexécution-optimisation-du-risque) ci-dessus

### Les résultats sont incohérents

- **Vérifiez** : Le matériau est plat et correctement sécurisé
- **Vérifiez** : La mise au point est cohérente sur toute la zone de test
- **Vérifiez** : La puissance laser est stable (vérifiez l'alimentation)
- **Essayez** : Une grille plus petite pour réduire la zone de test

## Sujets Connexes

- **[Mode Simulation](../simulation-mode)** - Prévisualiser l'exécution du test avant de l'exécuter
- **[Gravure](engrave)** - Comprendre les opérations de gravure
- **[Coupe de Contour](contour)** - Comprendre les opérations de coupe
