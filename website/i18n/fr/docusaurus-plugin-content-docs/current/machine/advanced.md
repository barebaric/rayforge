# Paramètres Avancés

La page Avancé dans les Paramètres Machine contient des options de configuration supplémentaires
pour des cas d'utilisation spécialisés.

![Paramètres Avancés](/screenshots/machine-settings-advanced.webp)

## Comportement de Connexion

Paramètres qui contrôlent comment Rayforge interagit avec votre machine pendant la connexion.

### Homing à la Connexion

Lorsqu'activé, Rayforge envoie automatiquement une commande de homing ($H) lors de la connexion à la
machine.

- **Activez si** : Votre machine a des contacteurs de fin de course fiables
- **Désactivez si** : Votre machine n'a pas de contacteurs de fin de course ou si le homing n'est
  pas fiable

### Effacer les Alarmes à la Connexion

Lorsqu'activé, Rayforge efface automatiquement tout état d'alarme lors de la connexion.

- **Activez si** : Votre machine démarre fréquemment en état d'alarme
- **Désactivez si** : Vous voulez enquêter manuellement sur les alarmes avant de les effacer

### Permettre le Homing d'Axe Unique

Lorsqu'activé, vous pouvez effectuer le homing d'axes individuels indépendamment (X, Y ou Z) au lieu
de requiring tous les axes ensemble. Ceci est utile pour les machines où un axe peut déjà être
correctement positionné.

## Paramètres d'Arcs et de Courbes

Paramètres pour contrôler comment les chemins courbes sont convertis en mouvements G-code.

### Supporter les Arcs

Lorsqu'activé, Rayforge génère des commandes d'arc (G2/G3) pour les chemins courbes au lieu de les
diviser en nombreux petits mouvements linéaires. Cela produit un G-code plus compact et un mouvement
plus fluide sur la plupart des contrôleurs.

Lorsque désactivé, toutes les courbes sont converties en segments linéaires (commandes G1), ce qui
fournit une compatibilité maximale avec les contrôleurs qui ne supportent pas les arcs.

### Supporter les Courbes Bézier

Lorsqu'activé, Rayforge génère des commandes Bézier cubiques natives (comme la commande G5 utilisée
par LinuxCNC) pour les chemins courbes. Cela produit un mouvement très fluide et un G-code compact
sur les contrôleurs qui le supportent. Vous devriez désactiver ce paramètre si le firmware de votre
machine ne comprend pas les commandes Bézier, auquel cas les courbes seront décomposées en segments
linéaires.

### Tolérance d'Arc et de Courbe

Ce paramètre contrôle la déviation maximale autorisée lors de l'ajustement des arcs et des courbes
aux chemins courbes, spécifiée en millimètres. Une valeur plus petite produit des chemins plus
précis mais peut nécessiter plus de commandes. Une valeur plus grande permet plus de déviation mais
génère moins de commandes.

Les valeurs typiques vont de 0,01mm pour un travail de précision à 0,1mm pour un traitement plus
rapide.

## Décalage de Balayage Bidirectionnel par Vitesse {#bidirectional-scan-offset-by-speed}

Le tableau de décalage de balayage bidirectionnel par vitesse corrige le désalignement horizontal
entre les passes raster de gauche à droite et de droite à gauche. Sur beaucoup de machines, ce
décalage n'est pas constant : plus la tête laser se déplace vite, plus les passes de droite à gauche
dérivent hors de l'alignement, ce qui se manifeste par un banding qui s'aggrave aux vitesses de
gravure élevées.

![Décalage de Balayage Bidirectionnel par Vitesse](/screenshots/machine-settings-advanced-bidir-offset-table.webp)

Chaque ligne du tableau associe une vitesse de gravure au décalage, en millimètres, dont les passes
de droite à gauche ont besoin à cette vitesse :

- Utilisez **Ajouter une Ligne** pour ajouter une ligne et le bouton de corbeille d'une ligne pour
  la supprimer
- Les valeurs positives décalent les passes de droite à gauche vers la droite, les valeurs négatives
  vers la gauche
- Les lignes sont conservées triées par vitesse ; si deux lignes ont la même vitesse, la dernière
  gagne

Lorsqu'une étape de gravure s'exécute, Rayforge y recherche la vitesse de l'étape et interpole le
décalage linéairement entre les lignes adjacentes. Les vitesses inférieures à la première ligne ou
supérieures à la dernière utilisent le décalage de la ligne la plus proche, et un tableau vide
signifie que toutes les vitesses utilisent un décalage de 0.

**Sans correction** — le désalignement augmente avec la vitesse de gravure :

![Passes raster désalignées à 1000, 4000 et 8000 mm/min](/images/bidir-scan-offset-uncorrected.webp)

**Avec le tableau de décalage appliqué :**

![Passes raster alignées à toutes les vitesses](/images/bidir-scan-offset-table-applied.webp)

Le tableau ne s'applique qu'aux étapes de gravure dont le propre **Décalage de Balayage
Bidirectionnel** vaut 0 ; un décalage par étape non nul a toujours la priorité. Consultez
[Décalage de Balayage Bidirectionnel](../features/operations/engrave.md#bidirectional-scan-offset)
pour voir à quoi ressemble le désalignement et comment le calibrer.

## Voir Aussi

- [Paramètres Matériels](hardware) - Configuration de l'origine des axes et inversion
- [Paramètres de l'Appareil](device) - Paramètres spécifiques GRBL
