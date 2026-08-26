---
description: "Configurez votre découpeuse ou graveuse laser pour la première fois. Utilisez l'assistant de configuration pour créer votre machine, puis connectez-vous et préparez-vous à découper avec Rayforge."
---

# Configuration Initiale

Après avoir installé Rayforge, vous devrez configurer votre découpeuse ou
graveuse laser. Ce guide vous accompagne dans la création de votre première
machine avec l'assistant de configuration et l'établissement d'une connexion.

## Étape 1 : Lancer Rayforge

Démarrez Rayforge depuis votre menu d'applications ou en exécutant `rayforge`
dans un terminal. Au premier lancement — lorsqu'aucune machine réelle n'a
encore été configurée — l'assistant de configuration s'ouvre automatiquement
pour que vous puissiez configurer votre machine sans chercher dans les menus.
(Vous pouvez toujours l'ouvrir plus tard depuis **Paramètres → Machines →
Add Machine**.)

## Étape 2 : Créer une Machine avec l'Assistant

Naviguez vers **Paramètres → Machines** ou appuyez sur <kbd>ctrl+comma</kbd>
pour ouvrir la boîte de dialogue des paramètres, puis sélectionnez la page
**Machines**.

![Paramètres Machine](/screenshots/app-settings-machines.png)

Cliquez sur **Add Machine** pour ouvrir le sélecteur de machine.

![Boîte de dialogue Add Machine](/screenshots/app-settings-machines-add.png)

### Vérification des permissions

Avant le démarrage de la découverte, l'assistant vérifie que Rayforge peut
effectivement accéder à vos ports série et caméras. Si un appareil est présent
mais l'accès manque, une **page de permissions** apparaît d'abord, expliquant
comment résoudre le problème sur votre plateforme :

- **Installations Snap** : accordez l'interface `serial-port` (et l'interface
  caméra si nécessaire) — les commandes exactes sont affichées avec un bouton
  de copie en un clic.
- **Linux non-Snap** : ajoutez votre utilisateur au groupe `dialout` pour que
  le nœud de périphérique série soit accessible.

Une fois l'accès en place, l'assistant continue automatiquement.

![Assistant — Vérification des permissions](/screenshots/config-wizard-permissions.png)

### Découverte automatique des appareils

L'assistant peut découvrir les appareils pour vous au lieu de vous demander de
choisir un point de départ et de tout remplir à la main :

- **Périphériques série USB** sont listés au fur et à mesure qu'ils apparaissent.
- **Appareils réseau** sont découverts via mDNS : les serveurs OctoPrint et
  les cartes ESP3D apparaissent aux côtés des périphériques série USB.
- Les appareils découverts sont **associés aux profils intégrés** lorsqu'une
  correspondance fiable est trouvée, vous pouvez donc souvent simplement
  confirmer les paramètres détectés au lieu de les saisir.
- GRBL sélectionne automatiquement le dialecte G-code correct à partir des
  drapeaux de compilation du firmware, et OctoPrint/Smoothieware sont
  sondés sur le réseau.
- Les appareils que vous avez déjà configurés sont affichés en **lecture seule**
  pour que vous ne créiez pas accidentellement des doublons.

Cliquez sur un appareil découvert pour préremplir l'assistant, ou choisissez
un point de départ manuellement comme décrit ci-dessous.

L'assistant de configuration adapte les étapes qu'il affiche à vos choix :

- Choisir un **profil intégré** préremplit le contrôleur, la zone de travail
  et la tête — l'assistant passe directement aux étapes rotatif, caméra et
  récapitulatif
- **Importer un profil** conserve les étapes matériel et tête afin de corriger
  ce que l'importation a mal configuré
- **Device Not Listed** vous guide à travers toutes les étapes, y compris les
  étapes contrôleur et recherche de spécifications IA

### Choisir un Point de Départ

Choisissez un profil d'appareil intégré pour préremplir les paramètres du
contrôleur, de la zone de travail et de la tête, ou cliquez sur **Device Not
Listed** pour tout configurer manuellement. Vous pouvez aussi **Import from
File…** un profil précédemment exporté ou un profil d'appareil LightBurn
(.lbdev) avec le calibrage de caméra et les paramètres laser.

![Assistant — Choisir un Point de Départ](/screenshots/config-wizard-profile.png)

### Choisir un Contrôleur

Choisissez la famille de firmware ou de protocole qui correspond à la carte
contrôleur de votre machine (GRBL, Marlin, Smoothie, Ruida, OctoPrint, …).
Choisissez **None — G-code export only** si vous souhaitez uniquement exporter
le G-code vers des fichiers et ne jamais piloter de machine physique. Cette
étape est ignorée lorsque vous partez d'un profil intégré ou d'une
importation.

![Assistant — Choisir un Contrôleur](/screenshots/config-wizard-controller.png)

### Connexion

Saisissez les paramètres de connexion requis par votre machine. Les champs
exacts dépendent du contrôleur que vous avez choisi :

- **Pilotes série** — chemin du périphérique USB (ex. `/dev/ttyUSB0` sur
  Linux, `COM3` sur Windows) et débit en bauds
- **Pilotes réseau** — adresse hôte et port (ex. `192.168.1.100`)
- **OctoPrint** — URL du serveur et clé API

![Assistant — Connexion](/screenshots/config-wizard-connect.png)

### Découvrir l'Appareil

Lorsque votre contrôleur le prend en charge, l'assistant propose de se
connecter à l'appareil et de lire automatiquement sa configuration — zone de
travail, vitesses, accélération et capacités du firmware. Cela fonctionne
via le port série USB **et via le réseau** (découverte mDNS pour OctoPrint
et ESP3D). Cliquez sur **Probe Now** pour détecter automatiquement ces
valeurs, ou utilisez **Next** pour les saisir manuellement dans les étapes
suivantes.

![Assistant — Découvrir l'Appareil](/screenshots/config-wizard-probe.png)

### Fournisseur IA

Affiché uniquement lorsqu'aucun fournisseur IA n'est encore configuré.
Saisissez un endpoint compatible OpenAI (URL de base et clé API) afin que
l'étape suivante puisse rechercher les spécifications des machines
commerciales connues. Ignorez cette étape pour saisir les valeurs
manuellement.

![Assistant — Fournisseur IA](/screenshots/config-wizard-ai-provider.png)

### Recherche de Spécifications IA

Si votre machine est un modèle commercial connu, l'IA peut préremplir ses
spécifications à partir de la documentation du fabricant. Saisissez le
fabricant et le modèle, puis cliquez sur **Look Up Specs**. Les valeurs
suggérées apparaissent sous forme de lignes à bascule et démarrent acceptées
— désactivez ce que vous ne souhaitez pas appliquer.

![Assistant — Recherche de Spécifications IA](/screenshots/config-wizard-ai-lookup.png)

### Matériel

Configurez la configuration physique de la machine :

- **Axes** — étendues X/Y de la zone de travail et coin d'origine des
  coordonnées (0,0)
- **Direction des axes** — inversez un axe si les coordonnées deviennent
  négatives
- **Axe Z** — si la machine a un axe Z (moteur de mise au point, lit
  mobile) ; lorsqu'il est absent, aucun mouvement Z n'est généré et le
  canevas 3D dispose le contenu sur le plan de gravure
- **Orientation du panneau** — fait pivoter l'espace de travail plat tel
  qu'il est présenté à l'écran (Natif, Pivoter à gauche, Pivoter à
  droite) ; les couches rotatives nécessitent Natif
- **Zone de Travail** — marges autour de l'espace inutilisable de la surface
  de travail
- **Limites Logicielles** — limites de sécurité facultatives pour le
  déplacement
- **Vitesses** — vitesse de déplacement max, vitesse de coupe max et
  accélération
- **Comportement** — retour à l'origine au démarrage et homing mono-axe

![Assistant — Matériel](/screenshots/config-wizard-hardware.png)

### Tête

Déclarez ce qui est fixé au portique — une tête laser ou une tête de broche —
et définissez ses paramètres. Pour un laser : puissance max (valeur S), taille
du spot, fréquence PWM et distance focale. Pour une broche : RPM max et min.

![Assistant — Tête](/screenshots/config-wizard-head.png)

### Module Rotatif

Configurez facultativement un accessoire rotatif : type (mandrins ou rouleaux),
axe (A/B/C), mode (vrai 4e axe vs. remplacement d'axe), géométrie et indicateur
d'inversion de direction. Ignorez cette étape pour ajouter un module rotatif
plus tard depuis les paramètres de la machine.

![Assistant — Module Rotatif](/screenshots/config-wizard-rotary.png)

### Caméras

Activez facultativement les caméras que vous souhaitez utiliser pour la
prévisualisation et l'alignement. Lorsque vous activez une caméra et
continuez, l'[assistant de
caméra](../machine/camera.md#étape-2--assistant-de-caméra) s'ouvre pour vous
guider à travers les paramètres d'image, la calibration d'objectif et
l'alignement d'image. Vous pouvez ignorer cette étape et configurer les
caméras plus tard depuis les paramètres caméra de la machine.

![Assistant — Caméras](/screenshots/config-wizard-camera.png)

### Récapitulatif et Nom

Donnez un nom à la machine et consultez un récapitulatif de tout ce que vous
avez configuré — pilote, connexion, zone de travail, vitesses, têtes, modules
rotatifs et caméras. L'assistant fait également remonter tout avertissement,
comme un pilote manquant ou une zone de travail non définie.

![Assistant — Récapitulatif et Nom](/screenshots/config-wizard-review.png)

Cliquez sur **Create Machine** pour finaliser. La boîte de dialogue Paramètres
Machine s'ouvre pour votre nouvelle machine, où vous pouvez ajuster tous les
paramètres préremplis par l'assistant. Consultez les pages [Configuration
Machine](../machine/general.md) pour plus de détails.

## Étape 3 : Connexion Automatique

Rayforge se connecte automatiquement à votre machine au démarrage de
l'application (si la machine est allumée et connectée). Vous n'avez pas besoin
de cliquer manuellement sur un bouton de connexion.

Le statut de connexion est affiché dans le coin inférieur gauche de la fenêtre
principale avec une icône de statut et une étiquette montrant l'état actuel
(Connecté, Connexion, Déconnecté, Erreur, etc.).

:::success Connecté !
Si votre machine affiche le statut "Connecté", vous êtes prêt à utiliser
Rayforge !
:::

---

## Dépannage des Problèmes de Connexion

### Appareil Non Trouvé

- **Linux (Série)** : Ajoutez votre utilisateur au groupe `dialout`. Ceci est
  requis pour **les installations Snap et non-Snap** sur les distributions
  basées sur Debian pour éviter les messages AppArmor DENIED :

  ```bash
  sudo usermod -a -G dialout $USER
  ```

  Déconnectez-vous et reconnectez-vous pour que les changements prennent effet.

- **Paquet Snap** : En plus du groupe `dialout` ci-dessus, assurez-vous d'avoir
  accordé les permissions de port série :

  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows** : Vérifiez le Gestionnaire de Périphériques pour confirmer que
  l'appareil est reconnu et notez le numéro de port COM.

### Connexion Refusée

- Vérifiez que l'adresse IP et le numéro de port sont corrects
- Assurez-vous que votre machine est allumée et connectée au réseau
- Vérifiez les paramètres du pare-feu si vous utilisez une connexion réseau

### Machine Ne Répond Pas

- Essayez un débit différent (certains appareils utilisent `9600` ou `57600`)
- Vérifiez les câbles desserrés ou les mauvaises connexions
- Éteignez et rallumez votre découpeuse laser et réessayez

Pour plus d'aide, voir [Problèmes de Connexion](../troubleshooting/connection.md).

---

**Suivant :** [Guide de Démarrage Rapide →](quick-start)
