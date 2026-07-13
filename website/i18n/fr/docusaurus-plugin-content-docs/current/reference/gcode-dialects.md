# Support des dialectes G-code

Rayforge prend en charge plusieurs dialectes G-code pour fonctionner avec
différents firmwares de contrôleur.

## Dialectes pris en charge

Rayforge prend actuellement en charge ces dialectes G-code :

| Dialecte                                      | Firmware     | Utilisation courante                   |
| --------------------------------------------- | ------------ | -------------------------------------- |
| **Grbl (Compat)**                             | GRBL 1.1+    | Lasers à diode, CNC de loisir          |
| **Grbl (Compat, sans axe Z)**                 | GRBL 1.1+    | Découpeurs laser 2D sans Z             |
| **Grbl Raster**                               | GRBL 1.1+    | Optimisé pour le travail raster        |
| **GRBL Dynamique (sensible à la profondeur)** | GRBL 1.1+    | Gravure laser sensible à la profondeur |
| **GRBL Dynamique (sans axe Z)**               | GRBL 1.1+    | Gravure laser sensible à la profondeur |
| **LinuxCNC**                                  | LinuxCNC     | Prise en charge native des Bézier (G5) |
| **Mach4 (M67 Analog)**                        | Mach4        | Gravure raster haute vitesse           |
| **Smoothieware**                              | Smoothieware | Découpeurs laser, CNC                  |
| **Marlin**                                    | Marlin 2.0+  | Imprimantes 3D avec laser              |

:::note Dialectes recommandés
:::

**Grbl (Compat)** est le dialecte le plus testé et recommandé pour les
applications laser standard.

**Grbl Raster** est optimisé pour la gravure raster sur les contrôleurs GRBL. Il garde
le laser en mode de puissance dynamique (M4) en continu et omet les commandes de vitesse
redondantes, ce qui produit un G-code plus fluide et plus compact.

**GRBL Dynamique (sensible à la profondeur)** est recommandé pour la gravure
laser sensible à la profondeur où la puissance varie pendant les coupes
(par exemple, gravure à profondeur variable).

**LinuxCNC** prend en charge les courbes Bézier cubiques natives via la
commande G5, ce qui produit un G-code très fluide et compact pour les trajectoires
courbes. Lorsque tu utilises ce dialecte, active l'option « Prise en charge
des courbes Bézier » dans les paramètres avancés de la machine pour profiter
de la sortie G5.

---

## Mach4 (M67 Analog)

Le dialecte **Mach4 (M67 Analog)** est conçu pour la gravure raster haute
vitesse avec les contrôleurs Mach4. Il utilise la commande M67 avec sortie
analogique pour un contrôle précis de la puissance laser.

### Caractéristiques principales

- **Sortie analogique M67** : Utilise `M67 E0 Q<0-255>` pour la puissance
  laser au lieu des commandes S en ligne
- **Pression tampon réduite** : En séparant les commandes de puissance des
  commandes de mouvement, le tampon du contrôleur est moins sollicité pendant
  les opérations haute vitesse
- **Raster haute vitesse** : Optimisé pour les opérations de gravure raster
  rapides

### Quand l'utiliser

Utilise ce dialecte lorsque :

- Tu as un contrôleur Mach4 avec capacité de sortie analogique
- Tu as besoin de gravure raster haute vitesse
- Ton contrôleur subit des débordements de tampon avec les commandes S en
  ligne standard

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
2. Cliquez sur l'icône **Copier** sur un dialecte intégré pour créer un nouveau
   dialecte personnalisé
3. Modifiez les paramètres du dialecte selon vos besoins
4. Enregistrez votre dialecte personnalisé

Chaque dialecte personnalisé est une copie indépendante. La modification d'un
dialecte n'affecte jamais les autres, tu peux donc expérimenter librement
sans risquer de perturber une configuration existante. Les dialectes
personnalisés sont stockés dans ton répertoire de configuration et peuvent
être partagés.

### Paramètres du dialecte

Lors de l'édition d'un dialecte personnalisé, la page Paramètres offre ces options :

**Mode laser continu** garde le laser en mode de puissance dynamique (M4) actif
pendant tout le travail au lieu de basculer M4/M5 entre les segments. C'est
utile pour la gravure raster où le laser doit rester allumé en continu
pendant les lignes de balayage.

**Vitesse modale** omet le paramètre de vitesse (F) des commandes de mouvement lorsqu'il
n'a pas changé depuis la dernière commande. Cela produit un G-code plus compact
et réduit la quantité de données envoyées au contrôleur.

### Commande d'allumage laser séparée pour la mise au point

Certains dialectes prennent en charge la configuration d'une commande séparée
pour allumer le laser à faible puissance, ce qui est utile pour le mode mise au
point. Cela te permet d'utiliser une commande différente pour le comportement
visuel de « pointeur laser » que celle utilisée pendant la découpe ou la gravure
réelle. Consulte la page des paramètres de ton dialecte pour cette option.

---

## Espaces réservés des modèles

Lors de la création ou de la modification d'un dialecte personnalisé, chaque
modèle de commande utilise des
[chaînes de formatage Python](https://docs.python.org/3/library/string.html#format-string-syntax)
avec des espaces réservés pour injecter des valeurs dynamiques. Utilise la
syntaxe `{nom}` ou `{nom:.0f}` (par ex. `{power:.0f}` pour formater en nombre
entier sans décimales).

### Espaces réservés disponibles par modèle

| Modèle                 | Espaces réservés                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Laser On**           | `power`                                                                                                      |
| **Focus Laser On**     | `power`                                                                                                      |
| **Laser Off**          | _(aucun)_                                                                                                    |
| **Changement d'outil** | `tool_number`                                                                                                |
| **Réglage vitesse**    | `speed`                                                                                                      |
| **Déplacement rapide** | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`                              |
| **Mouvement linéaire** | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `power`           |
| **Arc (CW)**           | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `power`           |
| **Arc (CCW)**          | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `power`           |
| **Bézier Cubique**     | `x`, `y`, `z`, `x_cmd`, `y_cmd`, `z_cmd`, `extra_cmd`, `f_command`, `s_command`, `i`, `j`, `p`, `q`, `power` |
| **Air On/Off**         | _(aucun)_                                                                                                    |
| **Origine tous**       | _(aucun)_                                                                                                    |
| **Origine axe**        | `axis_letter`                                                                                                |
| **Déplacer vers**      | `speed`, `x`, `y`, `z`                                                                                       |
| **Jog**                | `speed`                                                                                                      |
| **Effacer alarme**     | _(aucun)_                                                                                                    |
| **Décalage WCS**       | `p_num`, `x`, `y`, `z`                                                                                       |
| **Cycle de palpage**   | `axis_letter`, `max_travel`, `feed_rate`                                                                     |
| **Temporisation**      | `seconds`, `milliseconds`                                                                                    |

### Référence des espaces réservés

#### Coordonnées

| Espace réservé | Description                                                                                                                   |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `x`            | Coordonnée X cible en virgule flottante (par ex. `100.0`)                                                                     |
| `y`            | Coordonnée Y cible en virgule flottante (par ex. `200.0`)                                                                     |
| `z`            | Coordonnée Z cible en virgule flottante (par ex. `5.0`)                                                                       |
| `x_cmd`        | Chaîne de commande de l'axe X, par ex. `" X100.0"`. Omise si inchangée (si « Omettre les coordonnées inchangées » est activé) |
| `y_cmd`        | Chaîne de commande de l'axe Y, par ex. `" Y200.0"`. Omise si inchangée                                                        |
| `z_cmd`        | Chaîne de commande de l'axe Z, par ex. `" Z5.0"`. Omise si inchangée                                                          |
| `extra_cmd`    | Chaîne de commande pour axes supplémentaires (A, B, C), par ex. `" A90.0"`. Vide si aucun axe supplémentaire n'est configuré  |

#### Mouvement

| Espace réservé | Description                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `f_command`    | Chaîne de commande de vitesse d'avance, par ex. `" F3000"`. Omise si modale et inchangée                                 |
| `s_command`    | Chaîne de commande de broche/puissance, par ex. `" S500"`. Utilisée en modes dynamique/rastrage et en mode laser continu |
| `i`            | Décalage X du point de contrôle de l'arc ou Bézier par rapport à la position de départ                                   |
| `j`            | Décalage Y du point de contrôle de l'arc ou Bézier par rapport à la position de départ                                   |
| `p`            | Décalage X du deuxième point de contrôle Bézier par rapport à la position finale (Bézier Cubique uniquement)             |
| `q`            | Décalage Y du deuxième point de contrôle Bézier par rapport à la position finale (Bézier Cubique uniquement)             |

#### Puissance et vitesse

| Espace réservé | Description                                                                                                                  |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `power`        | Valeur absolue de puissance laser en virgule flottante. Prend en charge le formatage, par ex. `{power:.0f}` pour des entiers |
| `speed`        | Valeur de vitesse (pour les commandes Déplacer vers et Jog)                                                                  |
| `tool_number`  | Numéro de l'outil/tête laser                                                                                                 |

#### Machine et palpage

| Espace réservé | Description                                                                    |
| -------------- | ------------------------------------------------------------------------------ |
| `axis_letter`  | Lettre d'axe unique, par ex. `"X"`, `"Y"`, `"Z"` (pour Origine axe et palpage) |
| `p_num`        | Numéro P du WCS (par ex. `1` pour G54)                                         |
| `max_travel`   | Distance maximale de déplacement du palpage (Cycle de palpage uniquement)      |
| `feed_rate`    | Vitesse d'avance du palpage (Cycle de palpage uniquement)                      |

#### Temporisation

| Espace réservé | Description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| `seconds`      | Durée de temporisation en secondes en virgule flottante (par ex. `1.5`)   |
| `milliseconds` | Durée de temporisation en millisecondes en nombre entier (par ex. `1500`) |

### Conseils

- Les **spécifications de formatage** sont prises en charge : `{power:.0f}` formate la puissance
  en nombre entier, `{power:.2f}` avec deux décimales.
- Le paramètre **« Omettre les coordonnées inchangées »** contrôle si `x_cmd`, `y_cmd`
  et `z_cmd` sont laissés vides lorsque la position de l'axe n'a pas changé depuis
  la dernière commande. Cela réduit la taille du G-code.
- Le paramètre **« Vitesse modale »** contrôle si `f_command` est omise lorsque
  la vitesse d'avance n'a pas changé.
- Laissez un champ de modèle **vide** pour ignorer complètement cette commande
  (par ex., régler `bezier_cubic` sur `""` désactive la sortie Bézier native
  et utilise la linéarisation à la place).

---

## Pages connexes

- [Exporter du G-code](../files/exporting.md) - Paramètres d'exportation
- [Compatibilité des firmwares](firmware) - Versions de firmware
- [Paramètres de l'appareil](../machine/device.md) - Configuration GRBL
- [Macros et Hooks](../machine/hooks-macros.md) - Injection de G-code personnalisé
