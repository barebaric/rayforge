# Paramètres Laser

La page Laser dans les Paramètres Machine configure vos têtes laser et leurs propriétés.

![Paramètres Laser](/screenshots/machine-laser.png)

## Têtes Laser

Rayforge supporte les machines avec plusieurs têtes laser. Chaque tête laser a sa propre configuration.

### Ajouter une Tête Laser

Cliquez sur le bouton **Ajouter Laser** pour créer une nouvelle configuration de tête laser.

### Propriétés de la Tête Laser

Chaque tête laser a les paramètres suivants :

#### Nom

Un nom descriptif pour cette tête laser.

Exemples :
- "Diode 10W"
- "Tube CO2"
- "Laser Infrarouge"

#### Numéro d'Outil

L'index d'outil pour cette tête laser. Utilisé dans le G-code avec la commande T.

- Machines mono-tête : Utilisez 0
- Machines multi-têtes : Assignez des numéros uniques (0, 1, 2, etc.)

#### Puissance Maximum

La valeur de puissance maximum pour votre laser.

- **GRBL typique** : 1000 (plage S0-S1000)
- **Certains contrôleurs** : 255 (plage S0-S255)
- **Mode pourcentage** : 100 (plage S0-S100)

Cette valeur doit correspondre au paramètre $30 de votre firmware.

#### Puissance de Cadrage

Le niveau de puissance utilisé pour les opérations de cadrage (traçage du
contour sans couper).

- Définissez à 0 pour désactiver le cadrage
- Ajustez selon votre laser et matériau

#### Puissance de Focus

Le niveau de puissance utilisé lorsque le mode focus est activé. Le mode focus
allume le laser à faible puissance pour agir comme un "pointeur laser" pour le
positionnement.

- Définissez à 0 pour désactiver la fonction de mode focus
- Utilisez pour l'alignement visuel et le positionnement

:::tip Utiliser le Mode Focus
Cliquez sur le bouton focus (icône laser) dans la barre d'outils pour activer
le mode focus. Le laser s'allumera à ce niveau de puissance, vous aidant à voir
exactement où le laser est positionné. Consultez
[Positionnement de la Pièce](../features/workpiece-positioning) pour plus
d'informations.
:::

#### Taille du Spot

La taille physique de votre faisceau laser focalisé en millimètres.

- Entrez les dimensions X et Y
- La plupart des lasers ont un spot circulaire (ex : 0.1 x 0.1)
- Affecte les calculs de qualité de gravure

:::tip Mesurer la Taille du Spot
Pour mesurer la taille de votre spot :
1. Tirez une impulsion courte à faible puissance sur un matériau de test
2. Mesurez la marque résultante avec un pied à coulisse
3. Utilisez la moyenne de plusieurs mesures
:::

#### Couleur de Coupe

La couleur utilisée pour afficher les opérations de coupe pour ce laser dans le
canevas et la prévisualisation 3D. Cela vous aide à distinguer visuellement quel
laser effectuera chaque opération de coupe lorsque vous travaillez avec
plusieurs têtes laser.

- Cliquez sur l'échantillon de couleur pour ouvrir un sélecteur de couleur
- Choisissez une couleur qui contraste bien avec l'aperçu de votre matériau
- Les couleurs par défaut sont attribuées automatiquement

#### Couleur de Raster

La couleur utilisée pour afficher les opérations de raster/gravure pour ce laser
dans le canevas et la prévisualisation 3D.

- Cliquez sur l'échantillon de couleur pour ouvrir un sélecteur de couleur
- Utile pour différencier les opérations de raster des coupes
- Chaque laser peut avoir sa propre couleur de raster distincte

:::tip Workflows Multi-Laser
Lors de l'utilisation de plusieurs têtes laser, l'attribution de couleurs
différentes à chaque laser facilite la visualisation des opérations effectuées
par chaque laser.
Par exemple, utilisez le rouge pour votre laser de coupe principal et le bleu
pour un laser de gravure secondaire.
:::

## Voir Aussi

- [Paramètres de l'Appareil](device) - Paramètres du mode laser GRBL
- [Positionnement de la Pièce](../features/workpiece-positioning) -
  Utilisation du mode focus et autres méthodes de positionnement
