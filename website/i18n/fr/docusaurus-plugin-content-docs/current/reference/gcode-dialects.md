# Support des dialectes G-code

Rayforge prend en charge plusieurs dialectes G-code pour fonctionner avec différents firmwares de contrôleur.

## Dialectes pris en charge

Rayforge prend actuellement en charge ces dialectes G-code :

| Dialecte                           | Firmware     | Utilisation courante            | Statut                            |
| ---------------------------------- | ------------ | ------------------------------- | --------------------------------- |
| **GRBL (universel)**               | GRBL 1.1+    | Lasers à diode, CNC de loisir   | ✓ Principal, entièrement pris en charge |
| **GRBL (sans axe Z)**              | GRBL 1.1+    | Découpeurs laser 2D sans Z      | ✓ Variante optimisée              |
| **GRBL Dynamique (sensible à la profondeur)** | GRBL 1.1+    | Gravure laser sensible à la profondeur | ✓ Recommandé pour la puissance dynamique |
| **GRBL Dynamique (sans axe Z)**    | GRBL 1.1+    | Gravure laser sensible à la profondeur | ✓ Variante optimisée              |
| **Smoothieware**                   | Smoothieware | Découpeurs laser, CNC           | ⚠ Expérimental                    |
| **Marlin**                         | Marlin 2.0+  | Imprimantes 3D avec laser       | ⚠ Expérimental                    |

:::note Dialectes recommandés
:::

**GRBL (universel)** est le dialecte le plus testé et recommandé pour les applications laser standard.

    **GRBL Dynamique (sensible à la profondeur)** est recommandé pour la gravure laser sensible à la profondeur où la puissance varie pendant les coupes (par exemple, gravure à profondeur variable).
---

## Créer un dialecte personnalisé

Pour créer un dialecte G-code personnalisé basé sur un dialecte intégré :

1. Ouvrez **Paramètres de la machine** → **Dialecte G-code**
2. Cliquez sur l'icône **Copier** sur un dialecte intégré pour créer un nouveau dialecte personnalisé
3. Modifiez les paramètres du dialecte selon vos besoins
4. Enregistrez votre dialecte personnalisé

Les dialectes personnalisés sont stockés dans votre répertoire de configuration et peuvent être partagés.

---

## Pages connexes

- [Exporter du G-code](../files/exporting) - Paramètres d'exportation
- [Compatibilité des firmwares](firmware) - Versions de firmware
- [Paramètres de l'appareil](../machine/device) - Configuration GRBL
- [Macros et Hooks](../machine/hooks-macros) - Injection de G-code personnalisé
