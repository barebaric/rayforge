# Grille de Test de Matériau

Le générateur de Grille de Test de Matériau crée des motifs de test paramétriques pour t'aider à trouver les paramètres laser optimaux pour différents matériaux.

## Aperçu

Les tests de matériau sont essentiels pour le travail laser - différents matériaux nécessitent différents paramètres de puissance et vitesse. La Grille de Test de Matériau automatise ce processus en :

- Générant des grilles de test avec des plages de vitesse/puissance configurables
- Fournissant des préréglages pour les types de laser courants (Diode, CO2)
- Optimisant l'ordre d'exécution pour la sécurité (vitesses les plus rapides d'abord)
- Ajoutant des étiquettes pour identifier les paramètres de chaque cellule de test

## Créer une Grille de Test de Matériau

### Étape 1 : Ouvrir le Générateur

Accéde au générateur de Grille de Test de Matériau :

- Menu : **Outils → Grille de Test de Matériau**
- Cela crée une pièce spéciale qui génère le motif de test

### Étape 2 : Choisir un Préréglage (Optionnel)

Rayforge inclut des préréglages pour des scénarios courants :

| Préréglage        | Plage de Vitesse  | Plage de Puissance | Utilisation         |
| ----------------- | ----------------- | ------------------ | ------------------- |
| **Gravure Diode** | 1000-10000 mm/min | 10-100%            | Gravure laser diode |
| **Coupe Diode**   | 100-5000 mm/min   | 50-100%            | Coupe laser diode   |
| **Gravure CO2**   | 3000-20000 mm/min | 10-50%             | Gravure laser CO2   |
| **Coupe CO2**     | 1000-20000 mm/min | 30-100%            | Coupe laser CO2     |

Les préréglages sont des points de départ - tu peux ajuster tous les paramètres après en avoir sélectionné un.

### Étape 3 : Configurer les Paramètres

Ajuste les paramètres de la grille de test dans la boîte de dialogue des paramètres :

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
- **Puissance des étiquettes (%)** : Paramètre de puissance pour graver les étiquettes
- **Vitesse des étiquettes (mm/min)** : Vitesse pour graver les étiquettes (défaut : 1000 mm/min)

Les étiquettes sont gravées en premier, avant la grille de test, pour ne pas être masquées par le motif de test.

#### Intervalle de ligne (Test gravure uniquement)

- **Intervalle de ligne (mm)** : Espacement entre les lignes de balayage en mode gravure
- Des valeurs plus petites créent des remplissages plus denses mais prennent plus de temps
- Valeurs typiques : 0.1-0.3mm

### Étape 4 : Générer la Grille

Clique sur **Générer** pour créer le motif de test. La grille apparaît sur ton canevas comme une pièce spéciale.

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

- Ajoute une marge de 15mm à gauche et en haut pour l'espace des étiquettes

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

1. Charge ton matériau dans le laser
2. Fais la mise au point du laser correctement
3. Exécute le travail de grille de test de matériau
4. Surveille le test - arrête si une cellule cause des problèmes

### Étape 2 : Évaluer les Résultats

Après l'achèvement du test, examine chaque cellule :

- **Trop clair** : Augmente la puissance ou diminue la vitesse
- **Trop foncé/brûlé** : Diminue la puissance ou augmente la vitesse
- **Parfait** : Note la combinaison vitesse/puissance

### Étape 3 : Enregistrer les Paramètres

Documente tes paramètres réussis pour référence future :

- Type et épaisseur du matériau
- Type d'opération (gravure ou coupe)
- Combinaison vitesse et puissance
- Nombre de passes
- Toutes notes spéciales

:::tip Base de Données Matériaux
Envisage de créer un document de référence avec tes résultats de test de matériau pour une recherche rapide dans les projets futurs.
:::

## Utilisation Avancée

### Combiner avec d'Autres Opérations

Les grilles de test de matériau sont des pièces régulières - tu peux les combiner avec d'autres opérations :

**Exemple de flux de travail :**

1. Crée une grille de test de matériau
2. Ajoute une coupe de contour autour de toute la grille
3. Exécute le test, coupe libre, évalue les résultats

C'est utile pour couper la pièce de test libre du matériau de stock.

### Plages de Test Personnalisées

Pour un réglage fin, crée des tests à plage étroite :

**Test grossier** (trouver l'à peu près) :

- Vitesse : 1000-10000 mm/min (5 colonnes)
- Puissance : 10-100% (5 lignes)

**Test fin** (optimiser) :

- Vitesse : 4000-6000 mm/min (5 colonnes)
- Puissance : 35-45% (5 lignes)

### Différents Matériaux, Même Grille

Exécute la même configuration de grille sur différents matériaux pour construire ta bibliothèque de matériaux plus rapidement.

## Conseils & Meilleures Pratiques

### Conception de Grille

✅ **Commence avec les préréglages** - Bons points de départ pour les scénarios courants
✅ **Utilise des grilles 5×5** - Bon équilibre entre détail et temps de test
✅ **Active les étiquettes** - Essentiel pour identifier les résultats
✅ **Garde les carrés ≥20mm** - Plus facile à voir et mesurer les résultats

### Stratégie de Test

✅ **Teste sur du rebut d'abord** - Ne teste jamais sur le matériau final
✅ **Une variable à la fois** - Teste la plage de vitesse OU de puissance, pas les deux extrêmes
✅ **Permet le refroidissement** - Attends entre les tests sur le même matériau
✅ **Mise au point cohérente** - Même distance de mise au point pour tous les tests

### Sécurité

⚠️ **Surveille les tests** - Ne laisse jamais les tests en cours sans surveillance
⚠️ **Commence de manière conservatrice** - Commence avec des plages de puissance plus basses
⚠️ **Vérifie la ventilation** - Assure une extraction des fumées appropriée
⚠️ **Surveille le feu** - Aie un extincteur prêt

## Dépannage

### Les cellules de test s'exécutent dans le mauvais ordre

- Rayforge utilise l'ordre optimisé par risque (vitesses les plus rapides d'abord)
- C'est intentionnel et ne peut pas être changé
- Voir [Ordre d'Exécution](#ordre-dexécution-optimisation-du-risque) ci-dessus

### Les résultats sont incohérents

- **Vérifie** : Le matériau est plat et correctement sécurisé
- **Vérifie** : La mise au point est cohérente sur toute la zone de test
- **Vérifie** : La puissance laser est stable (vérifie l'alimentation)
- **Essayez** : Une grille plus petite pour réduire la zone de test

## Sujets Connexes

- **[Mode Simulation](../simulation-mode)** - Prévisualiser l'exécution du test avant de l'exécuter
- **[Gravure](engrave)** - Comprendre les opérations de gravure
- **[Coupe de Contour](contour)** - Comprendre les opérations de coupe
