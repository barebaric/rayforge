# Mode Simulation

![Mode Simulation](/screenshots/main-simulation.png)

Le Mode Simulation fournit une visualisation en temps réel de l'exécution de votre travail laser avant de l'exécuter sur la machine réelle. Il affiche l'ordre d'exécution, les variations de vitesse et les niveaux de puissance à travers une superposition interactive dans la vue 2D.

## Aperçu

Le Mode Simulation vous aide à :

- **Visualiser l'ordre d'exécution** - Voir la séquence exacte des opérations
- **Identifier les variations de vitesse** - La carte de chaleur couleur montre les mouvements lents (bleu) à rapides (rouge)
- **Vérifier les niveaux de puissance** - La transparence indique la puissance (faible=pâle, élevée=gras)
- **Valider les tests de matériau** - Confirmer l'ordre d'exécution de la grille de test
- **Repérer les erreurs tôt** - Identifier les problèmes avant de gaspiller du matériau
- **Comprendre le timing** - Voir combien de temps prennent différentes opérations


## Activer le Mode Simulation

Il y a trois façons d'entrer dans le Mode Simulation :

### Méthode 1 : Raccourci Clavier
Appuyez sur <kbd>f7</kbd> pour basculer le mode simulation on/off.

### Méthode 2 : Menu
- Naviguez vers **Affichage → Simuler l'Exécution**
- Cliquez pour basculer on/off

### Méthode 3 : Barre d'Outils (si disponible)
- Cliquez sur le bouton mode simulation dans la barre d'outils

:::note Vue 2D Uniquement
Le mode simulation fonctionne dans la vue 2D. Si vous êtes dans la vue 3D (<kbd>f6</kbd>), basculez vers la vue 2D (<kbd>f5</kbd>) d'abord.
:::


## Comprendre la Visualisation

### Carte de Chaleur de Vitesse

Les opérations sont colorées selon leur vitesse :

| Couleur | Vitesse | Signification |
| ------- | ------- | ------------- |
| 🔵 **Bleu** | La plus lente | Vitesse minimum dans votre travail |
| 🔵 **Cyan** | Lente | Vitesse sous la moyenne |
| 🟢 **Vert** | Moyenne | Vitesse moyenne |
| 🟡 **Jaune** | Rapide | Vitesse au-dessus de la moyenne |
| 🔴 **Rouge** | La plus rapide | Vitesse maximum dans votre travail |

La carte de chaleur est **normalisée** à la plage de vitesse réelle de votre travail :
- Si votre travail fonctionne à 100-1000 mm/min, bleu=100, rouge=1000
- Si votre travail fonctionne à 5000-10000 mm/min, bleu=5000, rouge=10000


### Transparence de Puissance

L'opacité de la ligne indique la puissance laser :

- **Lignes faibles** (10% opacité) = Faible puissance (0%)
- **Translucide** (50% opacité) = Puissance moyenne (50%)
- **Lignes solides** (100% opacité) = Pleine puissance (100%)

Cela aide à identifier :
- Mouvements de déplacement (0% puissance) - Très faible
- Opérations de gravure - Opacité modérée
- Opérations de coupe - Lignes solides, grasses

### Indicateur de Tête Laser

La position du laser est affichée avec une croix :

- 🔴 Croix rouge (lignes 6mm)
- Contour de cercle (rayon 3mm)
- Point central (0.5mm)

L'indicateur se déplace pendant la lecture, montrant exactement où se trouve le laser dans la séquence d'exécution.

## Contrôles de Lecture

Lorsque le mode simulation est actif, les contrôles de lecture apparaissent au bas du canevas :


### Bouton Lecture/Pause

- **▶️ Lecture** : Démarre la lecture automatique
- **⏸️ Pause** : S'arrête à la position actuelle
- **Lecture automatique** : La lecture démarre automatiquement quand vous activez le mode simulation

### Curseur de Progression

- **Glissez** pour parcourir l'exécution
- **Cliquez** pour sauter à un point spécifique
- Affiche l'étape actuelle / étapes totales
- Supporte les positions fractionnelles pour un parcours fluide

### Affichage de la Plage de Vitesse

Affiche les vitesses minimum et maximum dans votre travail :

```
Plage de vitesse : 100 - 5000 mm/min
```

Cela vous aide à comprendre les couleurs de la carte de chaleur.

## Utiliser le Mode Simulation

### Valider l'Ordre d'Exécution

La simulation montre l'ordre exact dans lequel les opérations s'exécuteront :

1. Activez le mode simulation (<kbd>f7</kbd>)
2. Regardez la lecture
3. Vérifiez que les opérations s'exécutent dans la séquence attendue
4. Vérifiez que les coupes se produisent après la gravure (si applicable)

**Exemple :** Grille de test de matériau
- Observez l'ordre optimisé par risque (vitesses les plus rapides d'abord)
- Confirmez que les cellules à faible puissance s'exécutent avant celles à haute puissance
- Validez que le test s'exécute en séquence sécuritaire

### Vérifier les Variations de Vitesse

Utilisez la carte de chaleur pour identifier les changements de vitesse :

- **Couleur cohérente** = Vitesse uniforme (bon pour la gravure)
- **Changements de couleur** = Variations de vitesse (attendu aux coins)
- **Zones bleues** = Mouvements lents (vérifiez si intentionnel)

### Estimer le Temps de Travail

La durée de lecture est mise à l'échelle à 5 secondes pour le travail complet :

- Regardez la vitesse de lecture
- Estimez le temps réel : Si la lecture semble fluide, le travail sera rapide
- Si la lecture saute rapidement, le travail a beaucoup de petits segments

:::tip Temps Réel
 Pour le temps de travail réel pendant l'exécution (non simulation), vérifiez la section droite de la barre d'état après la génération du G-code.
:::


### Déboguer les Tests de Matériau

Pour les grilles de test de matériau, la simulation montre :

1. **Ordre d'exécution** - Vérifiez que les cellules s'exécutent de la plus rapide→plus lente
2. **Carte de chaleur de vitesse** - Chaque colonne devrait être d'une couleur différente
3. **Transparence de puissance** - Chaque ligne devrait avoir une opacité différente

Cela aide à confirmer que le test s'exécutera correctement avant d'utiliser du matériau.

## Édition Pendant la Simulation

Contrairement à beaucoup d'outils CAO, Rayforge vous permet d'**éditer les pièces pendant la simulation** :

- Déplacer, mettre à l'échelle, faire pivoter les objets ✅
- Changer les paramètres d'opération ✅
- Ajouter/supprimer des pièces ✅
- Zoomer et panoramique ✅

**Mise à jour automatique :** La simulation se rafraîchit automatiquement lorsque vous changez les paramètres.

:::note Pas de Changement de Contexte
Vous pouvez rester en mode simulation pendant l'édition - pas besoin de basculer aller-retour.
:::


## Conseils & Meilleures Pratiques

### Quand Utiliser la Simulation

✅ **Simulez toujours avant :**
- D'exécuter des matériaux coûteux
- Des travaux longs (>30 minutes)
- Des grilles de test de matériau
- Des travaux avec ordre d'exécution complexe

✅ **Utilisez la simulation pour :**
- Vérifier l'ordre des opérations
- Vérifier les mouvements de déplacement inattendus
- Valider les paramètres vitesse/puissance
- Former de nouveaux utilisateurs

### Lire la Visualisation

✅ **Recherchez :**
- Couleurs cohérentes dans les opérations (bon)
- Transitions fluides entre les segments (bon)
- Zones bleues inattendues (investiguez - pourquoi si lent ?)
- Lignes faibles dans les zones de coupe (mauvais - vérifiez les paramètres de puissance)

⚠️ **Drapeaux rouges :**
- Coupe avant gravure (la pièce peut bouger)
- Sections bleues (lentes) très longues (inefficace)
- Changements de puissance en milieu d'opération (vérifiez les paramètres)

### Conseils de Performance

- La simulation se met à jour automatiquement sur les changements
- Pour les travaux très complexes (1000+ opérations), la simulation peut ralentir
- Désactivez la simulation (<kbd>f7</kbd>) lorsqu'elle n'est pas nécessaire pour de meilleures performances

## Raccourcis Clavier

| Raccourci | Action |
| --------- | ------ |
| <kbd>f7</kbd> | Basculer le mode simulation on/off |
| <kbd>f5</kbd> | Basculer vers la vue 2D (requis pour la simulation) |
| <kbd>espace</kbd> | Lecture/Pause de la lecture |
| <kbd>gauche</kbd> | Retour arrière |
| <kbd>droite</kbd> | Avancer |
| <kbd>home</kbd> | Aller au début |
| <kbd>end</kbd> | Aller à la fin |

## Sujets Connexes

- **[Aperçu 3D](../ui/3d-preview)** - Visualisation 3D du parcours d'outil
- **[Grille de Test de Matériau](operations/material-test-grid)** - Utiliser la simulation pour valider les tests
- **[Simuler Votre Travail](simulating-your-job)** - Guide de simulation détaillé
