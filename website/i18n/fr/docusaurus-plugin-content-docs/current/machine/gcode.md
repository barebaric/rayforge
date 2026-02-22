# Paramètres G-code

La page G-code dans les Paramètres Machine configure comment Rayforge génère le G-code pour votre machine.

![Paramètres G-code](/screenshots/machine-gcode.png)

## Dialecte G-code

Sélectionnez le dialecte G-code qui correspond au firmware de votre contrôleur. Différents contrôleurs utilisent des commandes et des formats légèrement différents.

### Dialectes Disponibles

- **GRBL** : Le plus courant pour les découpeuses laser de loisir. Utilise M3/M5 pour le contrôle laser.
- **Smoothieware** : Pour Smoothieboard et contrôleurs similaires.
- **Marlin** : Pour contrôleurs basés sur Marlin.
- **GRBL-compatible** : Pour les contrôleurs qui suivent majoritairement la syntaxe GRBL.

:::info
Le dialecte affecte la façon dont la puissance laser, les mouvements et autres commandes sont formatés dans le G-code de sortie.
:::

## G-code Personnalisé

Vous pouvez personnaliser le G-code que Rayforge génère à des points spécifiques du travail.

### Début de Programme

Commandes G-code exécutées au début de chaque travail, avant toute opération de coupe.

Utilisations courantes :
- Définir les unités (G21 pour mm)
- Définir le mode de positionnement (G90 pour absolu)
- Initialiser l'état de la machine

### Fin de Programme

Commandes G-code exécutées à la fin de chaque travail, après toutes les opérations de coupe.

Utilisations courantes :
- Éteindre le laser (M5)
- Retourner à l'origine (G0 X0 Y0)
- Parquer la tête

### Changement d'Outil

Commandes G-code exécutées lors du basculement entre têtes laser (pour machines multi-lasers).

## Voir Aussi

- [Bases du G-code](../general-info/gcode-basics) - Comprendre le G-code
- [Dialectes G-code](../reference/gcode-dialects) - Différences détaillées des dialectes
- [Hooks & Macros](hooks-macros) - Points d'injection de G-code personnalisé
