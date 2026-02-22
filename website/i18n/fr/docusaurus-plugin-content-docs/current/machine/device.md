# Paramètres de l'Appareil

La page Appareil dans les Paramètres Machine vous permet de lire et d'appliquer des paramètres directement à votre appareil connecté (contrôleur). Ce sont aussi appelés paramètres "dollar" ou paramètres `$$` dans GRBL.

![Paramètres de l'Appareil](/screenshots/machine-device.png)

:::warning Attention Lors du Changement de Paramètres
Des paramètres de firmware incorrects peuvent faire se comporter votre machine de manière imprévisible, perdre sa position ou même endommager le matériel. Enregistrez toujours les valeurs originales avant d'effectuer des changements, et modifiez un paramètre à la fois.
:::

## Aperçu

La page Appareil fournit un accès direct aux paramètres du firmware de votre contrôleur. C'est ici que vous pouvez :

- Lire les paramètres actuels depuis l'appareil
- Modifier des paramètres individuels
- Appliquer les changements à l'appareil

Les paramètres du firmware contrôlent :

- **Paramètres de mouvement** : Limites de vitesse, accélération, calibration
- **Contacteurs de fin de course** : Comportement de homing, limites logicielles/matérielles
- **Contrôle laser** : Plage de puissance, activation du mode laser
- **Configuration électrique** : Inversions de broches, pullups
- **Rapport** : Format et fréquence des messages de statut

Ces paramètres sont stockés sur votre contrôleur (pas dans Rayforge) et persistent après les cycles d'alimentation.

## Lire les Paramètres

Cliquez sur le bouton **Lire depuis l'Appareil** pour récupérer les paramètres actuels depuis votre contrôleur connecté. Cela nécessite :

- Que la machine soit connectée
- Que le pilote supporte la lecture des paramètres de l'appareil

## Appliquer les Paramètres

Après avoir modifié les paramètres, les changements sont appliqués à l'appareil. L'appareil peut :

- Redémarrer temporairement
- Se déconnecter et se reconnecter
- Nécessiter un cycle d'alimentation pour certains changements

## Accès Console

Vous pouvez aussi voir/modifier les paramètres via la console G-code :

**Voir tous les paramètres :**
```
$$
```

**Voir un seul paramètre :**
```
$100
```

**Modifier un paramètre :**
```
$100=80.0
```

**Restaurer les valeurs par défaut :**
```
$RST=$
```

:::danger Restaurer les Défauts Efface Tous les Paramètres
La commande `$RST=$` réinitialise tous les paramètres GRBL aux valeurs d'usine par défaut. Vous perdrez toute calibration et tout réglage. Sauvegardez vos paramètres d'abord !
:::

---

## Paramètres Critiques pour les Lasers

Ces paramètres sont les plus importants pour le fonctionnement laser :

### $32 - Mode Laser

**Valeur :** 0 = Désactivé, 1 = Activé

**Objectif :** Active les fonctionnalités spécifiques au laser dans GRBL

**Lorsqu'activé (1) :**
- Le laser s'éteint automatiquement pendant les mouvements G0 (rapides)
- La puissance s'ajuste dynamiquement pendant l'accélération/décélération
- Empêche les brûlures accidentelles pendant le positionnement

**Lorsqu'désactivé (0) :**
- Le laser se comporte comme une broche (mode CNC)
- Ne s'éteint pas pendant les rapides
- **Dangereux pour l'utilisation laser !**

:::warning Activez Toujours le Mode Laser
$32 devrait **toujours** être défini à 1 pour les découpeuses laser. Le mode laser désactivé peut causer des brûlures involontaires et des risques d'incendie.
:::

### $30 & $31 - Plage de Puissance Laser

**$30 - Puissance Laser Maximum (RPM)**
**$31 - Puissance Laser Minimum (RPM)**

**Objectif :** Définit la plage de puissance pour les commandes S

**Valeurs typiques :**
- $30=1000, $31=0 (plage S0-S1000, le plus courant)
- $30=255, $31=0 (plage S0-S255, certains contrôleurs)

:::tip Correspondance avec la Configuration Rayforge
Le paramètre "Puissance Max" dans vos [Paramètres Laser](laser) devrait correspondre à votre valeur $30. Si $30=1000, définissez la puissance max à 1000 dans Rayforge.
:::

### $130 & $131 - Déplacement Maximum

**$130 - Déplacement Maximum X (mm)**
**$131 - Déplacement Maximum Y (mm)**

**Objectif :** Définit la zone de travail de votre machine

**Pourquoi c'est important :**
- Les limites logicielles ($20) utilisent ces valeurs pour éviter les crashes
- Définit les limites du système de coordonnées
- Doit correspondre à la taille physique de votre machine

---

## Référence des Paramètres

### Configuration des Moteurs Pas à Pas ($0-$6)

Contrôle les signaux électriques et le timing des moteurs pas à pas.

| Paramètre | Description | Valeur Typique |
| --------- | ----------- | -------------- |
| $0 | Temps d'impulsion de pas (μs) | 10 |
| $1 | Délai d'inactivité du pas (ms) | 25 |
| $2 | Inversion d'impulsion de pas (masque) | 0 |
| $3 | Inversion de direction de pas (masque) | 0 |
| $4 | Inverser la broche d'activation de pas | 0 |
| $5 | Inverser les broches de limite | 0 |
| $6 | Inverser la broche de sonde | 0 |

### Limites & Homing ($20-$27)

Contrôle les contacteurs de fin de course et le comportement de homing.

| Paramètre | Description | Valeur Typique |
| --------- | ----------- | -------------- |
| $20 | Activation des limites logicielles | 0 ou 1 |
| $21 | Activation des limites matérielles | 0 |
| $22 | Activation du cycle de homing | 0 ou 1 |
| $23 | Inversion de direction de homing | 0 |
| $24 | Vitesse de localisation de homing (mm/min) | 25 |
| $25 | Vitesse de recherche de homing (mm/min) | 500 |
| $26 | Délai anti-rebond de homing (ms) | 250 |
| $27 | Distance de retrait de homing (mm) | 1.0 |

### Broche & Laser ($30-$32)

| Paramètre | Description | Valeur Laser |
| --------- | ----------- | ------------ |
| $30 | Vitesse maximum de la broche | 1000.0 |
| $31 | Vitesse minimum de la broche | 0.0 |
| $32 | Activation du mode laser | 1 |

### Calibration des Axes ($100-$102)

Définit combien de pas de moteur pas à pas égalent un millimètre de mouvement.

| Paramètre | Description | Notes |
| --------- | ----------- | ----- |
| $100 | Pas/mm X | Dépend du rapport poulie/courroie |
| $101 | Pas/mm Y | Généralement identique à X |
| $102 | Pas/mm Z | Non utilisé sur la plupart des lasers |

**Calcul des pas/mm :**
```
pas/mm = (pas_moteur_par_rev × micropas) / (dents_poulie × pas_courroie)
```

**Exemple :** 200 pas/rev, 16 micropas, poulie 20 dents, courroie GT2 :
```
pas/mm = (200 × 16) / (20 × 2) = 3200 / 40 = 80
```

### Vitesse & Accélération des Axes ($110-$122)

| Paramètre | Description | Valeur Typique |
| --------- | ----------- | -------------- |
| $110 | Taux max X (mm/min) | 5000.0 |
| $111 | Taux max Y (mm/min) | 5000.0 |
| $112 | Taux max Z (mm/min) | 500.0 |
| $120 | Accélération X (mm/s²) | 500.0 |
| $121 | Accélération Y (mm/s²) | 500.0 |
| $122 | Accélération Z (mm/s²) | 100.0 |

### Déplacement des Axes ($130-$132)

| Paramètre | Description | Notes |
| --------- | ----------- | ----- |
| $130 | Déplacement max X (mm) | Largeur de la zone de travail |
| $131 | Déplacement max Y (mm) | Profondeur de la zone de travail |
| $132 | Déplacement max Z (mm) | Déplacement Z (si applicable) |

---

## Exemple de Configuration Courante

### Laser à Diode Typique (300×400mm)

```gcode
$0=10          ; Impulsion de pas 10μs
$1=255         ; Délai d'inactivité du pas 255ms
$2=0           ; Pas d'inversion de pas
$3=0           ; Pas d'inversion de direction
$4=0           ; Pas d'inversion d'activation
$5=0           ; Pas d'inversion de limite
$10=1          ; Rapporter WPos
$11=0.010      ; Déviation de jonction 0.01mm
$12=0.002      ; Tolérance d'arc 0.002mm
$13=0          ; Rapporter en mm
$20=1          ; Limites logicielles activées
$21=0          ; Limites matérielles désactivées
$22=1          ; Homing activé
$23=0          ; Home vers min
$24=50.0       ; Vitesse de homing 50mm/min
$25=1000.0     ; Recherche de homing 1000mm/min
$26=250        ; Anti-rebond de homing 250ms
$27=2.0        ; Retrait de homing 2mm
$30=1000.0     ; Puissance max S1000
$31=0.0        ; Puissance min S0
$32=1          ; Mode laser ACTIVÉ
$100=80.0      ; Pas/mm X
$101=80.0      ; Pas/mm Y
$102=80.0      ; Pas/mm Z
$110=5000.0    ; Taux max X
$111=5000.0    ; Taux max Y
$112=500.0     ; Taux max Z
$120=500.0     ; Accélération X
$121=500.0     ; Accélération Y
$122=100.0     ; Accélération Z
$130=400.0     ; Déplacement max X
$131=300.0     ; Déplacement max Y
$132=0.0       ; Déplacement max Z
```

---

## Sauvegarde des Paramètres

### Procédure de Sauvegarde

1. **Via Rayforge :**
   - Ouvrez le panneau Paramètres de l'Appareil
   - Cliquez sur "Exporter les Paramètres"
   - Sauvegardez le fichier sous `grbl-backup-AAAA-MM-JJ.txt`

2. **Via console :**
   - Envoyez la commande `$$`
   - Copiez toute la sortie dans un fichier texte
   - Sauvegardez avec la date

### Procédure de Restauration

1. Ouvrez le fichier de sauvegarde
2. Envoyez chaque ligne (`$100=80.0`, etc.) via console
3. Vérifiez avec la commande `$$`

:::tip Sauvegardes Régulières
Sauvegardez vos paramètres après toute calibration ou réglage. Stockez les sauvegardes dans un emplacement sûr.
:::

---

## Voir Aussi

- [Paramètres Généraux](general) - Nom de machine et paramètres de vitesse
- [Paramètres Laser](laser) - Configuration de la tête laser
- [Dépannage de Connexion](../troubleshooting/connection) - Résoudre les problèmes de connexion

## Ressources Externes

- [Configuration GRBL v1.1](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Configuration)
- [Commandes GRBL v1.1](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands)
- [Documentation Grbl_ESP32](https://github.com/bdring/Grbl_Esp32/wiki)
