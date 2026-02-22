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

Le niveau de puissance utilisé pour les opérations de cadrage (traçage du contour sans couper).

- Définissez à 0 pour désactiver le cadrage
- Valeurs typiques : 5-20 (juste visible, ne marquera pas le matériau)
- Ajustez selon votre laser et matériau

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

## Voir Aussi

- [Paramètres de l'Appareil](device) - Paramètres du mode laser GRBL
