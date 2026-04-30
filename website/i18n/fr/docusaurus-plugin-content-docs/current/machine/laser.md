# Paramètres Laser

La page Laser dans les Paramètres Machine configure tes têtes laser et leurs
propriétés.

![Paramètres Laser](/screenshots/machine-laser.png)

## Têtes Laser

Rayforge supporte les machines avec plusieurs têtes laser. Chaque tête laser a
sa propre configuration.

### Ajouter une Tête Laser

Clique sur le bouton **Ajouter Laser** pour créer une nouvelle configuration
de tête laser.

### Propriétés de la Tête Laser

Chaque tête laser a les paramètres suivants :

#### Nom

Un nom descriptif pour cette tête laser.

Exemples :
- "Diode 10W"
- "Tube CO2"
- "Laser Infrarouge"

#### Numéro d'Outil

L'index d'outil pour cette tête laser. Utilisé dans le G-code avec la
commande T.

- Machines mono-tête : Utilise 0
- Machines multi-têtes : Assigne des numéros uniques (0, 1, 2, etc.)

#### Puissance Maximum

La valeur de puissance maximum pour ton laser.

- **GRBL typique** : 1000 (plage S0-S1000)
- **Certains contrôleurs** : 255 (plage S0-S255)
- **Mode pourcentage** : 100 (plage S0-S100)

Cette valeur doit correspondre au paramètre $30 de ton firmware.

#### Puissance de Cadrage

Le niveau de puissance utilisé pour les opérations de cadrage (traçage du
contour sans couper).

- Définis à 0 pour désactiver le cadrage
- Ajuste selon ton laser et matériau

#### Vitesse de Cadrage

La vitesse à laquelle la tête laser se déplace pendant le cadrage. Elle est
définie par tête laser, ainsi si ta machine possède plusieurs lasers aux
caractéristiques différentes, tu peux choisir une vitesse appropriée pour
chacun. Des vitesses plus lentes rendent le trajet de cadrage plus facile à
suivre visuellement.

#### Puissance de Focus

Le niveau de puissance utilisé lorsque le mode focus est activé. Le mode focus
allume le laser à faible puissance pour agir comme un « pointeur laser » pour le
positionnement.

- Définis à 0 pour désactiver la fonction de mode focus
- Utilise pour l'alignement visuel et le positionnement

:::tip Utiliser le Mode Focus
Clique sur le bouton focus (icône laser) dans la barre d'outils pour activer
le mode focus. Le laser s'allumera à ce niveau de puissance, t'aidant à
voir exactement où le laser est positionné. Consulte
[Positionnement de la Pièce](../features/workpiece-positioning) pour plus
d'informations.
:::

#### Taille du Spot

La taille physique de ton faisceau laser focalisé en millimètres.

- Entre les dimensions X et Y
- La plupart des lasers ont un spot circulaire (ex : 0.1 x 0.1)
- Affecte les calculs de qualité de gravure

:::tip Mesurer la Taille du Spot
Pour mesurer la taille de ton spot :
1. Tire une impulsion courte à faible puissance sur un matériau de test
2. Mesure la marque résultante avec un pied à coulisse
3. Utilise la moyenne de plusieurs mesures
:::

#### Couleur de Coupe

La couleur utilisée pour afficher les opérations de coupe pour ce laser dans
le canevas et la prévisualisation 3D. Cela t'aide à distinguer visuellement
quel laser effectuera chaque opération de coupe lorsque tu travailles avec
plusieurs têtes laser.

- Clique sur l'échantillon de couleur pour ouvrir un sélecteur de couleur
- Choisis une couleur qui contraste bien avec l'aperçu de ton matériau
- Les couleurs par défaut sont attribuées automatiquement

#### Couleur de Raster

La couleur utilisée pour afficher les opérations de raster/gravure pour ce
laser dans le canevas et la prévisualisation 3D.

- Clique sur l'échantillon de couleur pour ouvrir un sélecteur de couleur
- Utile pour différencier les opérations de raster des coupes
- Chaque laser peut avoir sa propre couleur de raster distincte

:::tip Workflows Multi-Laser
Lors de l'utilisation de plusieurs têtes laser, l'attribution de couleurs
différentes à chaque laser facilite la visualisation des opérations effectuées
par chaque laser. Par exemple, utilise le rouge pour ton laser de coupe
principal et le bleu pour un laser de gravure secondaire.
:::

#### Type de Laser

Choisis le type de tête laser dans le menu déroulant :

- **Diode** : Lasers diode standards (les plus courants pour les machines de loisir)
- **CO2** : Lasers à tube CO2
- **Fiber** : Lasers fibrés

Lorsque CO2 ou Fiber est sélectionné, des **paramètres PWM** supplémentaires
apparaissent (voir ci-dessous). Pour les lasers diode, la section PWM est
masquée car elle ne s'applique pas.

#### Paramètres PWM

Lorsqu'un type de laser CO2 ou Fiber est sélectionné, les contrôles PWM
suivants apparaissent :

- **Fréquence PWM** : La fréquence PWM par défaut en Hz pour cette tête laser.
  Les valeurs typiques vont de 500 Hz à plusieurs kHz selon ton contrôleur
  et ton alimentation.
- **Fréquence PWM max** : La limite supérieure du réglage de fréquence. Cela
  empêche d'entrer des valeurs que ton matériel ne peut pas gérer.
- **Largeur d'impulsion** : La largeur d'impulsion par défaut en microsecondes.
  Cela contrôle la durée d'activation de chaque impulsion pendant un cycle.
- **Largeur d'impulsion min/max** : Les limites pour le réglage de la largeur
  d'impulsion.

Ces valeurs par défaut sont transmises à tes étapes d'opération, où elles
peuvent être remplacées par étape si nécessaire.

#### Modèle 3D

Chaque tête laser peut avoir un modèle 3D attribué. Ce modèle est affiché dans
la [vue 3D](../ui/3d-preview) et suit le trajet d'outil pendant la simulation.

Clique sur la ligne de sélection du modèle pour parcourir les modèles disponibles. Une fois un modèle
sélectionné, tu peux ajuster son échelle, sa rotation (X/Y/Z) et sa distance focale pour
correspondre à ta tête laser physique.

## Voir Aussi

- [Paramètres de l'Appareil](device) - Paramètres du mode laser GRBL
- [Positionnement de la Pièce](../features/workpiece-positioning) -
  Utilisation du mode focus et autres méthodes de positionnement
