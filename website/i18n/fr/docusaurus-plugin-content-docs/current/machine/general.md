---
description:
  "Configurez les paramètres généraux de la machine dans Rayforge — définissez le nom, sélectionnez
  un pilote et configurez les vitesses et l'accélération."
---

# Paramètres généraux

La page Général des Paramètres de la machine contient le nom de la machine, la sélection du pilote
et les paramètres de connexion, ainsi que les paramètres de vitesse.

![Paramètres généraux](/screenshots/machine-settings-general.webp)

## Nom de la machine

Donnez un nom descriptif à votre machine. Cela permet de l'identifier dans le menu déroulant de
sélection lorsque vous avez plusieurs machines configurées.

## Pilote

Sélectionnez le pilote correspondant au contrôleur de votre machine. Le pilote gère la communication
entre Rayforge et le matériel.

Les appareils GRBL peuvent se connecter en USB/série ou par réseau :

- **GRBL (USB/Serial)** — Pilote avec comptage de tampon, détection d'interblocage et récupération
  de blocage. Sa pile du protocole GRBL s'exécute en Rust. Recommandé pour la plupart des appareils
  GRBL
- **GRBL (USB/Serial — compatibility mode)** — Pilote à protocole ping-pong. Envoie une ligne,
  attend « ok », envoie la suivante. Sans gestion de tampon ni détection d'interblocage. Nécessaire
  uniquement lorsque le pilote standard déclenche de fausses alarmes ou des erreurs de communication
- **grblHAL (Telnet)** — Connexion TCP brute (telnet) pour les contrôleurs grblHAL en réseau et pour
  le pont telnet d'ESP3D
- **FluidNC / ESP3D (Web API)** — Pilote HTTP/WebSocket pour les contrôleurs FluidNC et ESP3D v3,
  avec envoi de fichiers et prise en charge des réglages de l'appareil

Les contrôleurs basés sur Ruida sont pris en charge par le pilote **Ruida RPA**, qui se connecte
directement en USB ou UDP, ou via TUI RPC par l'intermédiaire du Ruida Protocol Analyzer. Son champ
USB propose un menu déroulant des appareils connectés avec correspondance VID:PID, et il honore le
réglage Power Mode de chaque étape — Dynamic active la mise à l'échelle de puissance du contrôleur,
Constant la désactive. Le comportement de la puissance se règle avec les options du pilote **Power
Scaling**, **VECTOR power floor** et **IMAGE power bias** : la mise à l'échelle de puissance relève
le minimum émis lorsque la vitesse de coupe de la couche diminue, le plancher de puissance vector
compense la sur-combustion aux extrémités des lignes, et le biais d'image compense les tubes CO2 qui
ne s'allument pas à très faible puissance.

### Liaison du Port Série

Au lieu d'un chemin de périphérique (ex. `/dev/ttyUSB0` ou `COM3`), le champ du port série accepte
également un identifiant USB `VID:PID` tel que `0403:6001`. Lorsqu'une machine est liée par VID:PID,
la reconnexion automatique la suit vers son nouveau port après que le système d'exploitation
réénumère les périphériques USB — par exemple après un redémarrage ou un
débranchement/rebranchement. Vous pouvez trouver le VID:PID d'un périphérique dans la sortie de
`lsusb` (Linux) ou dans le Gestionnaire de périphériques → Identificateurs de matériel (Windows).

Après avoir sélectionné un pilote, des paramètres de connexion spécifiques apparaissent sous le
sélecteur (ex. : port série, baud rate). Ils varient selon le pilote choisi.

<!-- prettier-ignore-start -->
:::tip
Une bannière d'erreur en haut de la page vous avertit si le pilote n'est pas configuré ou
rencontre un problème.
:::
<!-- prettier-ignore-end -->

## Vitesses et accélération

Ces paramètres contrôlent les vitesses maximales et l'accélération. Ils sont utilisés pour
l'estimation du temps de travail et l'optimisation des trajectoires.

### Vitesse maximale de déplacement

La vitesse maximale pour les mouvements rapides (sans coupe) lorsque le laser est éteint et que la
tête se déplace vers une nouvelle position.

- **Plage typique** : 2000–5000 mm/min
- **Remarque** : La vitesse réelle est également limitée par les paramètres de votre firmware. Ce
  champ est désactivé si le dialecte G-code sélectionné ne prend pas en charge la spécification
  d'une vitesse de déplacement.

### Vitesse maximale de coupe

La vitesse maximale autorisée pendant les opérations de coupe ou de gravure.

- **Plage typique** : 500–2000 mm/min
- **Remarque** : Certaines opérations peuvent utiliser des vitesses inférieures

### Accélération

Le taux d'accélération et de décélération de la machine, utilisé pour les estimations de temps et le
calcul de la distance d'overscan par défaut.

- **Plage typique** : 500–2000 mm/s²
- **Remarque** : Doit correspondre ou être inférieure aux paramètres d'accélération du firmware

<!-- prettier-ignore-start -->
:::tip
Commencez avec des valeurs de vitesse conservatrices et augmentez-les progressivement.
Observez votre machine pour détecter tout saut de courroie, calage de moteur ou perte de précision
de positionnement.
:::
<!-- prettier-ignore-end -->

## Exporter un profil de machine

Cliquez sur l'icône de partage dans la barre d'en-tête de la boîte de dialogue des paramètres pour
exporter la configuration actuelle de la machine. Choisissez un dossier de destination. Un fichier
ZIP est créé, contenant les paramètres de la machine et son dialecte G-code, qui peut être partagé
avec d'autres utilisateurs ou importé sur un autre système.

## Voir aussi

- [Configuration Initiale](../getting-started/first-time-setup.md) - Créer une machine pas à pas
  avec l'assistant de configuration
- [Paramètres matériel](hardware) - Dimensions de la zone de travail et configuration des axes
- [Paramètres du périphérique](device) - Lire et écrire les paramètres du firmware sur le contrôleur
