# Paramètres Avancés

La page Avancé dans les Paramètres Machine contient des options de configuration supplémentaires pour des cas d'utilisation spécialisés.

![Paramètres Avancés](/screenshots/machine-advanced.png)

## Comportement de Connexion

Paramètres qui contrôlent comment Rayforge interagit avec votre machine pendant la connexion.

### Homing à la Connexion

Lorsqu'activé, Rayforge envoie automatiquement une commande de homing ($H) lors de la connexion à la machine.

- **Activez si** : Votre machine a des contacteurs de fin de course fiables
- **Désactivez si** : Votre machine n'a pas de contacteurs de fin de course ou si le homing n'est pas fiable

### Effacer les Alarmes à la Connexion

Lorsqu'activé, Rayforge efface automatiquement tout état d'alarme lors de la connexion.

- **Activez si** : Votre machine démarre fréquemment en état d'alarme
- **Désactivez si** : Vous voulez enquêter manuellement sur les alarmes avant de les effacer

### Permettre le Homing d'Axe Unique

Lorsqu'activé, vous pouvez effectuer le homing d'axes individuels indépendamment (X, Y ou Z) au lieu de requiring tous les axes ensemble. Ceci est utile pour les machines où un axe peut déjà être correctement positionné.

## Paramètres d'Arcs et de Courbes

Paramètres pour contrôler comment les chemins courbes sont convertis en mouvements G-code.

### Supporter les Arcs

Lorsqu'activé, Rayforge génère des commandes d'arc (G2/G3) pour les chemins courbes au lieu de les diviser en nombreux petits mouvements linéaires. Cela produit un G-code plus compact et un mouvement plus fluide sur la plupart des contrôleurs.

Lorsque désactivé, toutes les courbes sont converties en segments linéaires (commandes G1), ce qui fournit une compatibilité maximale avec les contrôleurs qui ne supportent pas les arcs.

### Supporter les Courbes Bézier

Lorsqu'activé, Rayforge génère des commandes Bézier cubiques natives (comme la commande G5 utilisée par LinuxCNC) pour les chemins courbes. Cela produit un mouvement très fluide et un G-code compact sur les contrôleurs qui le supportent. Vous devriez désactiver ce paramètre si le firmware de votre machine ne comprend pas les commandes Bézier, auquel cas les courbes seront décomposées en segments linéaires.

### Tolérance d'Arc et de Courbe

Ce paramètre contrôle la déviation maximale autorisée lors de l'ajustement des arcs et des courbes aux chemins courbes, spécifiée en millimètres. Une valeur plus petite produit des chemins plus précis mais peut nécessiter plus de commandes. Une valeur plus grande permet plus de déviation mais génère moins de commandes.

Les valeurs typiques vont de 0,01mm pour un travail de précision à 0,1mm pour un traitement plus rapide.

## Voir Aussi

- [Paramètres Matériels](hardware) - Configuration de l'origine des axes et inversion
- [Paramètres de l'Appareil](device) - Paramètres spécifiques GRBL
