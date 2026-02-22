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

## Inverser les Axes

Ces paramètres inversent la direction des mouvements d'axe.

### Inverser l'Axe X

Inverse la direction de l'axe X. Lorsqu'activé, X positif se déplace vers la gauche au lieu de la droite.

### Inverser l'Axe Y

Inverse la direction de l'axe Y. Lorsqu'activé, Y positif se déplace vers le bas au lieu du haut.

:::info
L'inversion des axes est utile lorsque :
- Le système de coordonnées de votre machine ne correspond pas au comportement attendu
- Vous avez câblé vos moteurs à l'envers
- Vous voulez correspondre au comportement d'une autre machine
:::

## Voir Aussi

- [Paramètres Matériels](hardware) - Configuration de l'origine des axes
- [Paramètres de l'Appareil](device) - Paramètres de direction d'axe GRBL
