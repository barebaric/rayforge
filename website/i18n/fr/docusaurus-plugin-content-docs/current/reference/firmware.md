# Compatibilité des firmwares

Cette page documente la compatibilité des firmwares pour les contrôleurs laser utilisés avec
Rayforge.

## Aperçu

Rayforge est conçu principalement pour les **contrôleurs basés sur GRBL** mais prend également en
charge Marlin, Smoothieware et d'autres types de firmware.

### Matrice de compatibilité

| Firmware         | Version | Statut                     | Pilote                                   | Notes                            |
| ---------------- | ------- | -------------------------- | ---------------------------------------- | -------------------------------- |
| **GRBL**         | 1.1+    | Entièrement pris en charge | GRBL (USB/Serial) (+ compatibility mode) | Recommandé                       |
| **GRBL**         | 1.1+    | Expérimental               | GRBL (Bluetooth)                         | Appareils SPP Bluetooth appairés |
| **grblHAL**      | 2023+   | Compatible                 | GRBL (USB/Serial) / grblHAL (Telnet)     | Fork GRBL moderne                |
| **GRBL**         | 0.9     | Limité                     | GRBL (USB/Serial)                        | Ancien, peut avoir des problèmes |
| **FluidNC**      | Tous    | Compatible                 | FluidNC / ESP3D (Web API)                | Basé sur ESP32                   |
| **Smoothieware** | Tous    | Compatible                 | SmoothieDriver (Telnet)                  | Basé sur le réseau               |
| **Marlin**       | 2.0+    | Compatible                 | Marlin Serial                            | Mode laser requis                |
| **ESP3D**        | Tous    | Compatible                 | grblHAL (Telnet) / Web API               | v2: Telnet, v3: Web API          |
| **OctoPrint**    | Tous    | Compatible                 | OctoPrint                                | Voir notes ci-dessous            |
| **Autre**        | -       | Non pris en charge         | -                                        | Demander le support              |

---

## Firmware GRBL

**Statut :** ✓ Entièrement pris en charge **Versions :** 1.1+ **Pilote :** GRBL (USB/Serial)

### GRBL 1.1 (Recommandé)

**Qu'est-ce que GRBL 1.1 ?**

GRBL 1.1 est le firmware le plus courant pour les machines CNC et laser de loisir. Publié en 2017,
il est stable, bien documenté et largement pris en charge.

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

### Pilote GRBL (USB/Serial — compatibility mode)

Rayforge inclut un second pilote série GRBL pour les appareils où le pilote standard à comptage de
tampon provoque de fausses alarmes ou des erreurs de communication.

**Fonctionnement :**

- Utilise un protocole ping-pong : envoyer une ligne, attendre « ok », envoyer la suivante
- Pas de gestion de tampon par comptage de caractères
- Pas de détection d'interblocage ni de récupération de blocage
- Plus simple et plus prévisible sur certains appareils

**Quand l'utiliser :**

- Votre appareil reçoit de fausses alarmes de tampon avec le pilote standard
- Des erreurs de communication surviennent par intermittence avec le pilote standard
- Vous avez un appareil avec un comportement de tampon inhabituel

**Quand ne pas l'utiliser :**

- Le pilote GRBL (USB/Serial) standard fonctionne de manière fiable sur la plupart des appareils
- Le pilote simple n'a pas de récupération d'interblocage, donc les travaux peuvent s'arrêter sur
  une réponse « ok » perdue sans récupération automatique

---

## Pilote GRBL (Bluetooth)

**Statut :** Expérimental **Firmware :** GRBL **Pilote :** GRBL (Bluetooth)

Le pilote GRBL (Bluetooth) connecte les contrôleurs via le profil de port série Bluetooth Classic
(SPP/RFCOMM), courant sur de nombreux contrôleurs laser basés sur ESP32. La connexion est établie
directement par le système d'exploitation ; aucun pont `/dev/rfcomm*` ni port COM sortant n'est
nécessaire.

**Utilisation du pilote GRBL (Bluetooth) :**

1. **Associez le contrôleur** dans les réglages Bluetooth du système d'exploitation
2. **Sélectionnez « GRBL (Bluetooth) »** dans les réglages de la machine
3. **Choisissez l'appareil** dans la liste ou saisissez son adresse manuellement
4. **Conservez le canal RFCOMM 1** sauf si l'appareil en utilise un autre

Linux, macOS et Windows sont pris en charge. Bluetooth Low Energy (BLE) n'est pas pris en charge.

## grblHAL

**Statut :** Compatible **Versions :** 2023+ **Pilote :** GRBL (USB/Serial), grblHAL (Telnet)

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
- La connexion réseau est prise en charge via le pilote grblHAL (Telnet)
- Rapport d'état identique à GRBL

**Utiliser grblHAL :**

1. Sélectionnez le pilote "GRBL (USB/Serial)" dans Rayforge
2. Connectez-vous via série USB (comme GRBL)
3. Toutes les fonctionnalités fonctionnent comme documenté pour GRBL

Pour une connexion réseau plutôt qu'USB, utilisez le pilote grblHAL (Telnet) décrit ci-dessous.

---

## Pilote grblHAL (Telnet)

**Statut :** Pris en charge **Firmware :** grblHAL, ESP3D et autres contrôleurs GRBL réseau **Pilote
:** grblHAL (Telnet)

### À propos du pilote grblHAL (Telnet)

Le pilote grblHAL (Telnet) se connecte aux contrôleurs basés sur GRBL via le réseau par une
interface Telnet. Idéal pour les cartes avec WiFi ou Ethernet intégré — pas de câble USB nécessaire.

Les builds grblHAL avec LWIP mDNS annoncent `_telnet._tcp`, donc l'assistant les trouve
immédiatement et confirme chaque carte avec la commande d'informations de compilation `$I` avant de
la lister. Les cartes dont le firmware ne s'annonce pas via mDNS sont trouvées par un scan de
sous-réseau restreint (port telnet uniquement, concurrence et délai par hôte limités) avec la même
confirmation.

**Fonctionnalités :**

- Connectivité réseau (Ethernet/WiFi)
- Compatible avec les cartes basées sur grblHAL et ESP3D
- Même protocole GRBL que le pilote série

**Utilisation du pilote grblHAL (Telnet) :**

1. **Configurez le réseau** sur votre contrôleur (WiFi ou Ethernet)
2. **Sélectionnez « grblHAL (Telnet) »** dans les paramètres machine
3. **Entrez le nom d'hôte ou l'adresse IP** et le port de votre contrôleur
4. **Connectez** — le pilote communique via Telnet

**Exigences :**

- Contrôleur GRBL compatible réseau (grblHAL, ESP3D, etc.)
- Contrôleur et ordinateur sur le même réseau
- Interface Telnet activée sur le contrôleur

---

## Pilote FluidNC / ESP3D (Web API)

**Statut :** Pris en charge **Firmware :** FluidNC, ESP3D v3 **Pilote :** FluidNC / ESP3D (Web API)

### À propos du pilote Web API

Le pilote FluidNC / ESP3D (Web API) utilise l'interface HTTP/WebSocket du contrôleur — la même que
son interface web. Par rapport au pilote telnet, il ajoute l'envoi de fichiers ainsi que la lecture
et l'écriture des réglages de l'appareil, et Rayforge découvre automatiquement les appareils FluidNC
et ESP3D v3 via mDNS.

**Utilisation du pilote Web API :**

1. **Configurez le réseau** sur votre contrôleur (WiFi ou Ethernet)
2. **Sélectionnez « FluidNC / ESP3D (Web API) »** dans les paramètres machine
3. **Saisissez le nom d'hôte ou l'adresse IP** et les ports HTTP/WebSocket (pour FluidNC, les deux
   valent généralement 80)
4. **Connectez-vous**

**Prérequis :**

- Firmware FluidNC ou ESP3D v3
- Contrôleur et ordinateur sur le même réseau

---

## Smoothieware

**Pilote :** GRBL (USB/Serial) (mode compatibilité)

### Notes de compatibilité

Smoothieware utilise une syntaxe G-code différente :

**Différences clés :**

| Fonctionnalité      | GRBL           | Smoothieware       |
| ------------------- | -------------- | ------------------ |
| **Laser allumé**    | `M4 S<valeur>` | `M3 S<valeur>`     |
| **Plage puissance** | 0-1000         | 0.0-1.0 (flottant) |
| **État**            | format `<...>` | Format différent   |

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

**Versions :** 2.0+ avec support laser **Pilote :** Marlin Serial

### Pilote Marlin Serial

Rayforge inclut un MarlinSerialDriver dédié qui se connecte au firmware Marlin via le port série
(USB). Marlin 2.0+ peut contrôler des lasers lorsqu'il est correctement configuré.

**Fonctionnalités :**

- Communication série (USB)
- Protocole de handshake Marlin (attend le message "start")
- Streaming G-code ligne par ligne avec accusé de réception `ok`
- Interrogation de position M114
- Exécution de travaux avec rapport de progression granulaire
- Prise d'origine (G28), jogging, déplacement, changement d'outil (T)
- Réglage des offsets WCS (G10 L2 P)
- Contrôle de la puissance laser via le dialecte Marlin
- Annulation via M410 (Quick Stop)
- Auto-configuration (interrogation M115, M211, M503)

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

**Utiliser Marlin avec Rayforge :**

1. **Sélectionnez le pilote "Marlin (Serial)"** dans les paramètres machine
2. **Définissez le port série** et le débit en bauds (généralement 115200)
3. **Sélectionnez le dialecte Marlin** dans paramètres machine > G-code > Dialecte
4. **Configurez Marlin** pour l'utilisation laser
5. **Testez la plage de puissance** (0-1000 ou 0-255)

**Limitations :**

- Lecture/écriture des paramètres (comme `$$` de GRBL) non prise en charge
- Pas de connectivité réseau (USB uniquement)

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

| Carte                  | Firmware typique | Support Rayforge |
| ---------------------- | ---------------- | ---------------- |
| **Arduino CNC Shield** | GRBL 1.1         | Excellent        |
| **MKS DLC32**          | grblHAL          | Excellent        |
| **Ruida**              | Propriétaire     | Expérimental     |
| **OctoPrint (Pi)**     | Divers           | Bon              |

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

## Contrôleurs Ruida

Rayforge se connecte aux contrôleurs basés sur Ruida (p.ex. RDC6442, RDC6445, Ruida R5) via le
pilote **Ruida**, qui parle le protocole natif du contrôleur en USB, UDP ou TCP. Les profils
d'appareil Generic Ruida et Monport MP-570 60W CO2 fournissent des points de départ pour de
nouvelles machines.

**Fonctionnalités :**

- Connectivité USB, UDP et TCP ; lorsque USB et réseau sont configurés, Rayforge bascule
  automatiquement entre eux au branchement et au débranchement des câbles
- Envoi de travaux avec rapport de position, jogging, homing et Déplacement vers une position
- Le cadrage trace le contour du travail avec des déplacements absolus faisceau éteint, il
  fonctionne donc sans tirer le laser
- Honore le réglage Power Mode de chaque étape : Dynamic active la mise à l'échelle de puissance du
  contrôleur, Constant la désactive
- Les ports série peuvent être liés depuis un menu déroulant de périphériques USB avec
  correspondance VID:PID, si bien que la reconnexion automatique suit la machine après que le
  système d'exploitation réénumère les périphériques USB

**Réglages de puissance :**

- **Power scaling** — relève le minimum émis lorsque la vitesse de coupe de la couche diminue
- **VECTOR power floor** — puissance minimale pour la compensation de coupe vectorielle pendant
  l'accélération et la décélération, réduisant la sur-combustion aux extrémités des lignes
- **IMAGE power bias** — puissance ajoutée aux lignes de balayage raster, car les tubes CO2 ne
  s'allument pas à très faible puissance

**Limitations :**

- Expérimental — testé jusqu'ici sur un contrôleur RDC6442S (Monport MP570 CO2)
- Pas de génération de G-code ; Ruida utilise son propre protocole propriétaire
- Le mode rotatif n'est pas encore pris en charge

---

## OctoPrint

Rayforge inclut un pilote OctoPrint qui soumet le G-code directement à un serveur OctoPrint via le
réseau. Ceci est utile si ton laser est connecté à un Raspberry Pi ou une autre machine exécutant
OctoPrint.

**Fonctionnalités :**

- Connexion WebSocket pour les mises à jour d'état en temps réel
- Repli sur l'interrogation REST lorsque WebSocket n'est pas disponible
- Reconnexion automatique en cas de perte de connexion
- Soumission de travaux avec démarrage automatique de l'impression
- Jogging, mise à l'origine et contrôles pause/reprise
- Flux « Demander l'accès » pour les clés d'application OctoPrint

**Utilisation du pilote OctoPrint :**

1. Sélectionne le pilote « OctoPrint » dans les paramètres machine
2. Entre le nom d'hôte ou l'adresse IP de ton serveur OctoPrint
3. Définis le port (par défaut : 80)
4. Clique sur « Demander l'accès » pour obtenir une clé API via le système de clés d'application
   d'OctoPrint
5. Connecte — Rayforge établira une connexion WebSocket

**Limitations :**

- Impossible de lire ou écrire les paramètres du firmware via OctoPrint
- Les résultats de palpage ne sont pas rapportés par OctoPrint
- La lecture des décalages WCS n'est pas prise en charge

---

### Autres fonctionnalités demandées

Les utilisateurs ont demandé le support pour :

- **Trocen/AWC** - Contrôleurs laser commerciaux
- **ESP32 WiFi** - Connectivité réseau pour grblHAL
- **API Laser** - API machine directe (pas de G-code)

**Statut :** Actuellement non pris en charge. Les demandes de fonctionnalités sont les bienvenues
sur GitHub.

## Contribuer

Pour ajouter le support d'un firmware :

1. Implémentez le pilote dans `rayforge/machine/driver/`
2. Définissez le dialecte G-code dans `rayforge/machine/models/dialect.py`
3. Testez minutieusement sur du matériel réel
4. Soumettez une pull request avec documentation

---

## Pages connexes

- [Dialectes G-code](gcode-dialects) - Détails des dialectes
- [Paramètres de l'appareil](../machine/device.md) - Configuration GRBL
- [Problèmes de connexion](../troubleshooting/connection.md) - Dépannage de connexion
- [Paramètres généraux](../machine/general.md) - Configuration machine
