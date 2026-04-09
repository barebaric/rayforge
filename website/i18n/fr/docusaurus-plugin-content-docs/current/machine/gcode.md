# Paramètres G-code

La page G-code dans les Paramètres Machine configure comment Rayforge génère le G-code pour votre machine.

![Paramètres G-code](/screenshots/machine-gcode.png)

## Dialecte G-code

Sélectionnez le dialecte G-code qui correspond au firmware de votre contrôleur. Différents contrôleurs utilisent des commandes et des formats légèrement différents.

### Dialectes Disponibles

- **Grbl (Compat)** : Dialecte GRBL standard pour les découpeuses laser de loisir. Utilise M3/M5 pour le contrôle laser.
- **Grbl (Compat, no Z axis)** : Identique à Grbl (Compat) mais sans commandes d'axe Z. Pour machines 2D uniquement.
- **GRBL Dynamic** : Utilise le mode de puissance laser dynamique de GRBL pour la gravure à puissance variable.
- **GRBL Dynamic (no Z axis)** : Mode dynamique sans commandes d'axe Z.
- **LinuxCNC** : Pour les contrôleurs LinuxCNC. Prend en charge les courbes Bézier cubiques (G5) natives.
- **Smoothieware** : Pour Smoothieboard et contrôleurs similaires.
- **Marlin** : Pour contrôleurs basés sur Marlin.

:::info
Le dialecte affecte la façon dont la puissance laser, les mouvements et autres commandes sont formatés dans le G-code de sortie.
:::

## Préambule et Postscript du Dialecte

Chaque dialecte inclut un préambule et un postscript G-code personnalisables qui s'exécutent au début et à la fin des travaux.

### Préambule

Commandes G-code exécutées au début de chaque travail, avant toute opération de coupe. Les utilisations courantes incluent la définition des unités (G21 pour mm), le mode de positionnement (G90 pour absolu) et l'initialisation de l'état de la machine.

### Postscript

Commandes G-code exécutées à la fin de chaque travail, après toutes les opérations de coupe. Les utilisations courantes incluent l'extinction du laser (M5), le retour à l'origine (G0 X0 Y0) et le stationnement de la tête.

## Voir Aussi

- [Bases du G-code](../general-info/gcode-basics) - Comprendre le G-code
- [Dialectes G-code](../reference/gcode-dialects) - Différences détaillées des dialectes
- [Hooks & Macros](hooks-macros) - Points d'injection de G-code personnalisé
