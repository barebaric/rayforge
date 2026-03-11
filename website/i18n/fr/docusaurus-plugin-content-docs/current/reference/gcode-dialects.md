# Support des dialectes G-code

Rayforge prend en charge plusieurs dialectes G-code pour fonctionner avec différents firmwares de contrôleur.

## Dialectes pris en charge

Rayforge prend actuellement en charge ces dialectes G-code :

| Dialecte                                      | Firmware     | Utilisation courante                   |
| --------------------------------------------- | ------------ | -------------------------------------- |
| **GRBL (universel)**                          | GRBL 1.1+    | Lasers à diode, CNC de loisir          |
| **GRBL (sans axe Z)**                         | GRBL 1.1+    | Découpeurs laser 2D sans Z             |
| **GRBL Dynamique (sensible à la profondeur)** | GRBL 1.1+    | Gravure laser sensible à la profondeur |
| **GRBL Dynamique (sans axe Z)**               | GRBL 1.1+    | Gravure laser sensible à la profondeur |
| **Mach4 (M67 Analog)**                        | Mach4        | Gravure raster haute vitesse           |
| **Smoothieware**                              | Smoothieware | Découpeurs laser, CNC                  |
| **Marlin**                                    | Marlin 2.0+  | Imprimantes 3D avec laser              |

:::note Dialectes recommandés
:::

**GRBL (universel)** est le dialecte le plus testé et recommandé pour les applications laser standard.

**GRBL Dynamique (sensible à la profondeur)** est recommandé pour la gravure laser sensible à la profondeur où la puissance varie pendant les coupes (par exemple, gravure à profondeur variable).

---

## Mach4 (M67 Analog)

Le dialecte **Mach4 (M67 Analog)** est conçu pour la gravure raster haute vitesse avec les contrôleurs Mach4. Il utilise la commande M67 avec sortie analogique pour un contrôle précis de la puissance laser.

### Caractéristiques principales

- **Sortie analogique M67** : Utilise `M67 E0 Q<0-255>` pour la puissance laser au lieu des commandes S en ligne
- **Pression tampon réduite** : En séparant les commandes de puissance des commandes de mouvement, le tampon du contrôleur est moins sollicité pendant les opérations haute vitesse
- **Raster haute vitesse** : Optimisé pour les opérations de gravure raster rapides

### Quand l'utiliser

Utilisez ce dialecte lorsque :

- Vous avez un contrôleur Mach4 avec capacité de sortie analogique
- Vous avez besoin de gravure raster haute vitesse
- Votre contrôleur subit des débordements de tampon avec les commandes S en ligne standard

### Format de commande

Le dialecte génère du G-code comme :

```gcode
M67 E0 Q127  ; Définir la puissance laser à 50% (127/255)
G1 X100 Y200 F1000  ; Déplacer vers la position
M67 E0 Q0    ; Éteindre le laser
```

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
