---
description: "Utilisez l'assistant de configuration pour détecter et configurer automatiquement un appareil connecté en interrogeant ses paramètres de firmware."
---

# Assistant de Configuration

L'assistant de configuration détecte automatiquement votre appareil en s'y
connectant et en lisant ses paramètres de firmware. Il crée un profil de
machine entièrement configuré en quelques secondes, éliminant la
configuration manuelle.

## Démarrer l'Assistant

1. Ouvrez **Paramètres → Machines** et cliquez sur **Add Machine**
2. Dans le sélecteur de profil, cliquez sur **Other Device…** en bas

L'assistant s'ouvre alors. Il ne nécessite **pas** de profil d'appareil
existant — l'assistant en crée un de zéro en interrogeant le matériel
connecté.

## Connexion

La première page vous demande de sélectionner un pilote et de fournir les
paramètres de connexion.

![Assistant Connexion](/screenshots/app-settings-machines-wizard-connect.png)

### Sélection du Pilote

Choisissez le pilote correspondant au contrôleur de votre appareil. Seuls
les pilotes prenant en charge la détection sont listés. Typiquement :

- **GRBL (Série)** — Appareils GRBL connectés via USB
- **GRBL (Réseau)** — Appareils GRBL WiFi/Ethernet

### Paramètres de Connexion

Après avoir sélectionné un pilote, remplissez les détails de connexion
(port série, débit en bauds, hôte, etc.). Ce sont les mêmes paramètres que
dans les [Paramètres Généraux](general).

Cliquez sur **Next** pour commencer la détection.

## Détection

L'assistant se connecte à l'appareil et interroge son firmware pour
obtenir les données de configuration. Cela inclut :

- La version du firmware et les informations de build (`$I`)
- Tous les paramètres du firmware (`$$`)
- Les courses des axes, vitesses, accélération et plage de puissance laser

Cette étape se termine généralement en quelques secondes.

## Vérification

Après une détection réussie, la page de vérification affiche tous les
paramètres détectés. Tout est pré-rempli mais peut être ajusté avant la
création de la machine.

![Assistant Vérification](/screenshots/app-settings-machines-wizard-review.png)

### Informations sur l'Appareil

Informations en lecture seule détectées depuis le firmware :

- **Nom de l'appareil** — extrait des informations de build du firmware
- **Version du firmware** — par ex. `1.1h`
- **Taille du tampon RX** — taille du tampon de réception série
- **Tolérance d'arc** — tolérance d'interpolation d'arc du firmware

### Zone de Travail

- **Course X** / **Course Y** — course maximale des axes en unités machine,
  lue depuis les paramètres firmware `$130` et `$131`

### Vitesse

- **Vitesse de déplacement max.** — la plus petite valeur entre `$110` et `$111`
- **Vitesse de coupe max.** — par défaut identique à la vitesse de
  déplacement ; ajustez si nécessaire

### Accélération

- **Accélération** — la plus petite valeur entre `$120` et `$121`, en unités
  machine par seconde au carré

### Laser

- **Puissance max. (valeur S)** — depuis le paramètre firmware `$30`
  (spindle max)

### Comportement

- **Mise à l'origine au démarrage** — activé si le homing du firmware
  (`$22`) est activé
- **Mise à l'origine mono-axe** — activé si le firmware a été compilé avec
  le flag `H`

### Avertissements

L'assistant peut afficher des avertissements sur des problèmes potentiels,
comme :

- Le mode laser n'est pas activé (`$32=0`)
- L'appareil rapporte en pouces (`$13=1`)

## Créer la Machine

Cliquez sur **Create Machine** pour finaliser. L'assistant émet le profil
configuré et le processus normal de création de machine continue — le
[dialogue de paramètres de machine](general) s'ouvre pour des ajustements
supplémentaires.

## Limites

- L'assistant ne fonctionne qu'avec les pilotes prenant en charge la
  détection. Si votre pilote n'est pas listé, utilisez un profil
  prédéfini du sélecteur à la place.
- La détection nécessite que l'appareil soit allumé et connecté.
- Certains paramètres du firmware peuvent ne pas être lisibles sur tous
  les appareils.

## Voir aussi

- [Paramètres Généraux](general) — configuration manuelle de la machine
- [Paramètres du périphérique](device) — lire et écrire les paramètres du
  firmware sur une machine déjà configurée
- [Ajouter une Machine](../application-settings/machines) — aperçu du
  processus de création de machine
