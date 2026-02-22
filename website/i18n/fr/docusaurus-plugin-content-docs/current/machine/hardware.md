# Paramètres Matériels

La page Matériel dans les Paramètres Machine configure les dimensions physiques, le système de coordonnées et les limites de mouvement de votre machine.

![Paramètres Matériels](/screenshots/machine-hardware.png)

## Axes

Configurez l'étendue des axes et le système de coordonnées de votre machine.

### Étendue X/Y

La plage de déplacement complète de chaque axe en unités machine.

- Mesurez la zone de coupe réelle, pas l'extérieur de la machine
- Prenez en compte les obstructions ou limites
- Exemple : 400 pour un laser K40 typique

### Origine des Coordonnées

Sélectionnez où l'origine des coordonnées (0,0) de votre machine est située. Cela détermine comment les coordonnées sont interprétées.

- **Bas Gauche** : Le plus courant pour les appareils GRBL. X augmente vers la droite, Y augmente vers le haut.
- **Haut Gauche** : Courant pour certaines machines de style CNC. X augmente vers la droite, Y augmente vers le bas.
- **Haut Droit** : X augmente vers la gauche, Y augmente vers le bas.
- **Bas Droit** : X augmente vers la gauche, Y augmente vers le haut.

#### Trouver Votre Origine

1. Mettez votre machine à l'origine en utilisant le bouton Home
2. Observez où la tête laser se déplace
3. Cette position est votre origine (0,0)

:::info
Le paramètre d'origine des coordonnées affecte la façon dont le G-code est généré. Assurez-vous qu'il correspond à la configuration de homing de votre firmware.
:::

### Direction des Axes

Inversez la direction de n'importe quel axe si nécessaire :

- **Inverser la Direction de l'Axe X** : Rend les valeurs de coordonnées X négatives
- **Inverser la Direction de l'Axe Y** : Rend les valeurs de coordonnées Y négatives
- **Inverser la Direction de l'Axe Z** : Activez si une commande Z positive (ex : G0 Z10) déplace la tête vers le bas

## Zone de Travail

Les marges définissent l'espace inutilisable autour des bords de l'étendue de vos axes. C'est utile lorsque votre machine a des zones où le laser ne peut pas atteindre (ex : en raison de l'assemblage de la tête laser, des chaînes de câbles ou d'autres obstructions).

- **Marge Gauche/Haut/Droite/Bas** : L'espace inutilisable depuis chaque bord en unités machine

Lorsque des marges sont définies, la zone de travail (espace utilisable) est calculée comme l'étendue des axes moins les marges.

## Limites Logicielles

Limites de sécurité configurables pour le déplacement de la tête machine. Lorsqu'activées, les commandes de déplacement empêcheront tout mouvement en dehors de ces limites.

- **Activer les Limites Logicielles Personnalisées** : Basculer pour utiliser des limites personnalisées au lieu des limites de la surface de travail
- **X/Y Min** : Coordonnée minimum pour chaque axe
- **X/Y Max** : Coordonnée maximum pour chaque axe

Les limites logicielles sont automatiquement contraintes pour rester dans l'étendue des axes (0 à la valeur d'étendue).

:::tip
Utilisez les limites logicielles pour protéger les zones de votre surface de travail qui ne doivent jamais être atteintes lors du déplacement, comme les zones avec des fixations ou des équipements sensibles.
:::

## Voir Aussi

- [Paramètres Généraux](general) - Nom de machine et paramètres de vitesse
- [Paramètres de l'Appareil](device) - Paramètres GRBL de homing et d'axes
