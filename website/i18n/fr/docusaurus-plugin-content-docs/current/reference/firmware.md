# Compatibilité des firmwares

Cette page documente la compatibilité des firmwares pour les contrôleurs laser utilisés avec Rayforge.

## Aperçu

Rayforge est conçu principalement pour les **contrôleurs basés sur GRBL** mais a un support expérimental pour d'autres types de firmware.

### Matrice de compatibilité

| Firmware          | Version | Statut           | Pilote                | Notes                   |
| ----------------- | ------- | ---------------- | --------------------- | ----------------------- |
| **GRBL**          | 1.1+    | Entièrement pris en charge | GRBL Serial           | Recommandé              |
| **grblHAL**       | 2023+   | Compatible       | GRBL Serial           | Fork GRBL moderne       |
| **GRBL**          | 0.9     | Limité           | GRBL Serial           | Ancien, peut avoir des problèmes |
| **Smoothieware**  | Tous    | Expérimental     | Aucun (utiliser pilote GRBL) | Non testé           |
| **Marlin**        | 2.0+    | Expérimental     | Aucun (utiliser pilote GRBL) | Mode laser requis    |
| **Autre**         | -       | Non pris en charge | -                   | Demander le support     |

---

## Firmware GRBL

**Statut :** ✓ Entièrement pris en charge
**Versions :** 1.1+
**Pilote :** GRBL Serial

### GRBL 1.1 (Recommandé)

**Qu'est-ce que GRBL 1.1 ?**

GRBL 1.1 est le firmware le plus courant pour les machines CNC et laser de loisir. Publié en 2017, il est stable, bien documenté et largement pris en charge.

**Fonctionnalités prises en charge par Rayforge :**

- Communication série (USB)
- Rapport d'état en temps réel
- Mode laser (M4 puissance constante)
- Lecture/écriture des paramètres ($$, $X=valeur)
- Cycles de mise à l'origine ($H)
- Systèmes de coordonnées de travail (G54)
- Commandes de déplacement ($J=)
- Remplacement de vitesse d'avance
- Limites logicielles
- Limites matérielles (butoirs)

**Limitations connues :**

- Plage de puissance : 0-1000 (paramètre S)
- Pas de connectivité réseau (USB uniquement)
- Mémoire embarquée limitée (petit tampon G-code)

### Vérifier la version GRBL

**Interroger la version :**

Connectez-vous à votre contrôleur et envoyez :

```
$I
```

**Exemples de réponse :**

```
[VER:1.1h.20190825:]
[OPT:V,15,128]
```

- `1.1h` = version GRBL 1.1h
- La date indique la compilation

### GRBL 0.9 (Ancien)

**Statut :** Support limité

GRBL 0.9 est une ancienne version avec quelques problèmes de compatibilité :

**Différences :**

- Format de rapport d'état différent
- Pas de mode laser (M4) - utilise M3 uniquement
- Moins de paramètres
- Syntaxe de déplacement différente

**Si vous avez GRBL 0.9 :**

1. **Mettez à niveau vers GRBL 1.1** si possible (recommandé)
2. **Utilisez M3 au lieu de M4** (puissance moins prévisible)
3. **Testez minutieusement** - certaines fonctionnalités peuvent ne pas fonctionner

**Instructions de mise à niveau :** Voir [GRBL Wiki](https://github.com/gnea/grbl/wiki)

---

## grblHAL

**Statut :** Compatible
**Versions :** 2023+
**Pilote :** GRBL Serial

### Qu'est-ce que grblHAL ?

grblHAL est un fork moderne de GRBL avec des fonctionnalités améliorées :

- Support de multiple matériel de contrôleur (STM32, ESP32, etc.)
- Réseau Ethernet/WiFi
- Support carte SD
- Plus de broches E/S
- Support laser amélioré

**Compatibilité avec Rayforge :**

- **Entièrement compatible** - grblHAL maintient le protocole GRBL 1.1
- Toutes les fonctionnalités GRBL fonctionnent
- Les fonctionnalités supplémentaires (réseau, SD) ne sont pas encore prises en charge par Rayforge
- Rapport d'état identique à GRBL

**Utiliser grblHAL :**

1. Sélectionnez le pilote "GRBL Serial" dans Rayforge
2. Connectez-vous via série USB (comme GRBL)
3. Toutes les fonctionnalités fonctionnent comme documenté pour GRBL

**Futur :** Rayforge pourrait ajouter le support des fonctionnalités spécifiques à grblHAL (réseau, etc.)

---

## Smoothieware

**Versions :** Toutes
**Pilote :** GRBL Serial (mode compatibilité)

### Notes de compatibilité

Smoothieware utilise une syntaxe G-code différente :

**Différences clés :**

| Fonctionnalité  | GRBL           | Smoothieware     |
| --------------- | -------------- | ---------------- |
| **Laser allumé**| `M4 S<valeur>` | `M3 S<valeur>`   |
| **Plage puissance** | 0-1000      | 0.0-1.0 (flottant) |
| **État**        | format `<...>` | Format différent |

**Utiliser Smoothieware avec Rayforge :**

1. **Sélectionnez le dialecte Smoothieware** dans paramètres machine > G-code > Dialecte
2. **Testez avec faible puissance** d'abord
3. **Vérifiez que la plage de puissance** correspond à votre configuration
4. **Pas d'état en temps réel** - retour limité

**Limitations :**

- Rapport d'état pas entièrement compatible
- La mise à l'échelle de puissance peut différer
- Paramètres (commandes $$) non pris en charge
- Non testé sur matériel réel

**Recommandation :** Si possible, utilisez un firmware compatible GRBL à la place.

---

## Marlin

**Versions :** 2.0+ avec support laser
**Pilote :** GRBL Serial

### Marlin pour laser

Marlin 2.0+ peut contrôler des lasers lorsqu'il est correctement configuré.

**Exigences :**

1. **Firmware Marlin 2.0 ou ultérieur**
2. **Fonctionnalités laser activées :**
   ```cpp
   #define LASER_FEATURE
   #define LASER_POWER_INLINE
   ```
3. **Plage de puissance correcte** configurée :
   ```cpp
   #define SPEED_POWER_MAX 1000
   ```

**Compatibilité :**

- Mode laser M4 pris en charge
- G-code de base (G0, G1, G2, G3)
- Rapport d'état diffère
- Commandes de paramètres différentes
- Assistance air (M8/M9) peut ne pas fonctionner

**Utiliser Marlin avec Rayforge :**

1. **Sélectionnez le dialecte Marlin** dans paramètres machine > G-code > Dialecte
2. **Configurez Marlin** pour l'utilisation laser
3. **Testez que la plage de puissance** correspond (0-1000 ou 0-255)
4. **Test limité** - utilisez avec prudence

**Meilleure alternative :** Utilisez le firmware GRBL sur les machines laser.

---

## Guide de mise à niveau du firmware

### Mettre à niveau vers GRBL 1.1

**Pourquoi mettre à niveau ?**

- Mode laser (M4) pour puissance constante
- Meilleur rapport d'état
- Plus fiable
- Meilleur support Rayforge

**Comment mettre à niveau :**

1. **Identifiez votre carte contrôleur :**
   - Arduino Nano/Uno (ATmega328P)
   - Arduino Mega (ATmega2560)
   - Carte personnalisée

2. **Téléchargez GRBL 1.1 :**
   - [Versions GRBL](https://github.com/gnea/grbl/releases)
   - Obtenez la dernière version 1.1 (1.1h recommandée)

3. **Flashez le firmware :**

   **En utilisant Arduino IDE :**

   ```
   1. Installez Arduino IDE
   2. Ouvrez le croquis GRBL (grbl.ino)
   3. Sélectionnez la bonne carte et le bon port
   4. Téléversez
   ```

   **En utilisant avrdude :**

   ```bash
   avrdude -c arduino -p m328p -P /dev/ttyUSB0 \
           -U flash:w:grbl.hex:i
   ```

4. **Configurez GRBL :**
   - Connectez-vous via série
   - Envoyez `$$` pour voir les paramètres
   - Configurez pour votre machine

### Sauvegarde avant mise à niveau

**Sauvegardez vos paramètres :**

1. Connectez-vous au contrôleur
2. Envoyez la commande `$$`
3. Copiez toute la sortie des paramètres
4. Sauvegardez dans un fichier

**Après la mise à niveau :**

- Restaurez les paramètres un par un : `$0=10`, `$1=25`, etc.
- Ou utilisez les valeurs par défaut et reconfigurez

---

## Matériel de contrôleur

### Contrôleurs courants

| Carte                  | Firmware typique   | Support Rayforge |
| ---------------------- | ------------------ | ---------------- |
| **Arduino CNC Shield** | GRBL 1.1           | Excellent        |
| **MKS DLC32**          | grblHAL            | Excellent        |
| **Cohesion3D**         | Smoothieware       | Limité           |
| **Cartes SKR**         | Marlin/grblHAL     | Variable         |
| **Ruida**              | Propriétaire       | Non pris en charge |
| **Trocen**             | Propriétaire       | Non pris en charge |
| **TopWisdom**          | Propriétaire       | Non pris en charge |

### Contrôleurs recommandés

Pour la meilleure compatibilité Rayforge :

1. **Arduino Nano + CNC Shield** (GRBL 1.1)
   - Peu coûteux (~10-20€)
   - Facile à flasher
   - Bien documenté

2. **MKS DLC32** (grblHAL)
   - Moderne (basé ESP32)
   - Capacité WiFi
   - Développement actif

3. **Cartes GRBL personnalisées**
   - Beaucoup disponibles sur les marketplaces
   - Vérifiez le support GRBL 1.1+

---

## Configuration du firmware

### Paramètres GRBL pour laser

**Paramètres essentiels :**

```
$30=1000    ; Puissance max broche/laser (1000 = 100%)
$31=0       ; Puissance min broche/laser
$32=1       ; Mode laser activé (1 = on)
```

**Paramètres machine :**

```
$100=80     ; X pas/mm (calibrez pour votre machine)
$101=80     ; Y pas/mm
$110=3000   ; X vitesse max (mm/min)
$111=3000   ; Y vitesse max
$120=100    ; X accélération (mm/sec)
$121=100    ; Y accélération
$130=300    ; X course max (mm)
$131=200    ; Y course max (mm)
```

**Paramètres de sécurité :**

```
$20=1       ; Limites logicielles activées
$21=1       ; Limites matérielles activées (si vous avez des butoirs)
$22=1       ; Mise à l'origine activée
```

### Tester le firmware

**Séquence de test de base :**

1. **Test de connexion :**

   ```
   Envoyer : ?
   Attendre : <Idle|...>
   ```

2. **Vérification de version :**

   ```
   Envoyer : $I
   Attendre : [VER:1.1...]
   ```

3. **Vérification des paramètres :**

   ```
   Envoyer : $$
   Attendre : $0=..., $1=..., etc.
   ```

4. **Test de mouvement :**

   ```
   Envoyer : G91 G0 X10
   Attendre : La machine bouge de 10mm en X
   ```

5. **Test laser (très faible puissance) :**
   ```
   Envoyer : M4 S10
   Attendre : Le laser s'allume (faible)
   Envoyer : M5
   Attendre : Le laser s'éteint
   ```

---

## Résolution des problèmes de firmware

### Le firmware ne répond pas

**Symptômes :**

- Pas de réponse aux commandes
- Échec de connexion
- État non rapporté

**Diagnostic :**

1. **Vérifiez le débit en bauds :**
   - GRBL 1.1 par défaut : 115200
   - GRBL 0.9 : 9600
   - Essayez les deux

2. **Vérifiez le câble USB :**
   - Câble de données, pas charge uniquement
   - Remplacez par un câble connu bon

3. **Vérifiez le port :**
   - Linux : `/dev/ttyUSB0` ou `/dev/ttyACM0`
   - Windows : COM3, COM4, etc.
   - Le bon port est sélectionné dans Rayforge

4. **Testez avec un terminal :**
   - Utilisez screen, minicom, ou PuTTY
   - Envoyez `?` et voyez si vous obtenez une réponse

### Plantages du firmware

**Symptômes :**

- Le contrôleur se bloque pendant un travail
- Déconnexions aléatoires
- Comportement incohérent

**Causes possibles :**

1. **Dépassement de tampon** - Fichier G-code trop complexe
2. **Bruit électrique** - Mauvaise mise à la terre ou EMI
3. **Bug du firmware** - Mettez à niveau vers la dernière version
4. **Problème matériel** - Contrôleur défectueux

**Solutions :**

- Mettez à niveau le firmware vers la dernière version stable
- Simplifiez le G-code (réduisez la précision, moins de segments)
- Ajoutez des perles de ferrite au câble USB
- Améliorez la mise à la terre et le routage des câbles

### Mauvais firmware

**Symptômes :**

- Commandes rejetées
- Comportement inattendu
- Messages d'erreur

**Solution :**

1. Interrogez la version du firmware : `$I`
2. Comparez avec les attentes de Rayforge
3. Mettez à niveau ou sélectionnez le bon dialecte

---

## Support futur des firmwares

### Fonctionnalités demandées

Les utilisateurs ont demandé le support pour :

- **Contrôleurs Ruida** - Contrôleurs laser chinois
- **Trocen/AWC** - Contrôleurs laser commerciaux
- **ESP32 WiFi** - Connectivité réseau pour grblHAL
- **API Laser** - API machine directe (pas de G-code)

**Statut :** Actuellement non pris en charge. Les demandes de fonctionnalités sont les bienvenues sur GitHub.

### Contribuer

Pour ajouter le support d'un firmware :

1. Implémentez le pilote dans `rayforge/machine/driver/`
2. Définissez le dialecte G-code dans `rayforge/machine/models/dialect.py`
3. Testez minutieusement sur du matériel réel
4. Soumettez une pull request avec documentation

---

## Pages connexes

- [Dialectes G-code](gcode-dialects) - Détails des dialectes
- [Paramètres de l'appareil](../machine/device) - Configuration GRBL
- [Problèmes de connexion](../troubleshooting/connection) - Dépannage de connexion
- [Paramètres généraux](../machine/general) - Configuration machine
