# Configuration Initiale

Après avoir installé Rayforge, vous devrez configurer votre découpeuse ou graveuse laser. Ce guide vous accompagnera dans la création de votre première machine et l'établissement d'une connexion.

## Étape 1 : Lancer Rayforge

Démarrez Rayforge depuis votre menu d'applications ou en exécutant `rayforge` dans un terminal. Vous devriez voir l'interface principale avec un canevas vide.

## Étape 2 : Créer une Machine

Naviguez vers **Paramètres → Machines** ou appuyez sur <kbd>ctrl+comma</kbd> pour ouvrir la boîte de dialogue des paramètres, puis sélectionnez la page **Machines**.

Cliquez sur **Ajouter une Machine** pour créer une nouvelle machine. Vous pouvez soit :

1. **Choisir un profil intégré** - Sélectionnez parmi les modèles de machine prédéfinis
2. **Sélectionner "Personnalisé"** - Commencez avec une configuration vierge

Après la sélection, la boîte de dialogue Paramètres Machine s'ouvre pour votre nouvelle machine.

![Paramètres Machine](/screenshots/application-machines.png)

## Étape 3 : Configurer les Paramètres Généraux

La page **Général** contient les informations de base de la machine, la sélection du pilote et les paramètres de connexion.

![Paramètres Généraux](/screenshots/machine-general.png)

### Informations Machine

1. **Nom de la Machine** : Donnez un nom descriptif à votre machine (ex : "K40 Laser", "Ortur LM2")

### Sélection du Pilote

Sélectionnez le pilote approprié pour votre appareil dans le menu déroulant :

- **GRBL Série** - Pour les appareils GRBL connectés via port USB/série
- **GRBL Réseau** - Pour les appareils GRBL avec connectivité WiFi/Ethernet
- **Smoothie** - Pour les appareils basés sur Smoothieware

### Paramètres du Pilote

Selon le pilote sélectionné, configurez les paramètres de connexion :

#### GRBL Série (USB)

1. **Port** : Choisissez votre appareil dans le menu déroulant (ex : `/dev/ttyUSB0` sur Linux, `COM3` sur Windows)
2. **Débit** : Sélectionnez `115200` (standard pour la plupart des appareils GRBL)

:::info
Si votre appareil n'apparaît pas dans la liste, vérifiez qu'il est connecté et que vous avez les permissions nécessaires. Sur Linux, vous devrez peut-être ajouter votre utilisateur au groupe `dialout`.
:::

#### GRBL Réseau / Smoothie (WiFi/Ethernet)

1. **Hôte** : Entrez l'adresse IP de votre appareil (ex : `192.168.1.100`)
2. **Port** : Entrez le numéro de port (généralement `23` ou `8080`)

### Vitesses & Accélération

Ces paramètres sont utilisés pour l'estimation du temps de travail et l'optimisation du parcours :

- **Vitesse de Déplacement Max** : Vitesse de mouvement rapide maximale
- **Vitesse de Coupe Max** : Vitesse de coupe maximale
- **Accélération** : Utilisée pour les estimations de temps et les calculs d'overscan

## Étape 4 : Configurer les Paramètres Matériels

Passez à l'onglet **Matériel** pour configurer les dimensions physiques de votre machine.

![Paramètres Matériels](/screenshots/machine-hardware.png)

### Dimensions

- **Largeur** : Entrez la largeur maximale de votre zone de travail en millimètres
- **Hauteur** : Entrez la hauteur maximale de votre zone de travail en millimètres

### Axes

- **Origine des Coordonnées (0,0)** : Sélectionnez où l'origine de votre machine est située :
  - Bas Gauche (le plus courant pour GRBL)
  - Haut Gauche
  - Haut Droit
  - Bas Droit

### Décalages d'Axe (Optionnel)

Configurez les décalages X et Y si votre machine les nécessite pour un positionnement précis.

## Étape 5 : Connexion Automatique

Rayforge se connecte automatiquement à votre machine au démarrage de l'application (si la machine est allumée et connectée). Vous n'avez pas besoin de cliquer manuellement sur un bouton de connexion.

Le statut de connexion est affiché dans le coin inférieur gauche de la fenêtre principale avec une icône de statut et une étiquette montrant l'état actuel (Connecté, Connexion, Déconnecté, Erreur, etc.).

:::success Connecté !
Si votre machine affiche le statut "Connecté", vous êtes prêt à utiliser Rayforge !
:::

## Optionnel : Configuration Avancée

### Lasers Multiples

Si votre machine a plusieurs modules laser (ex : diode et CO2), vous pouvez les configurer dans la page **Laser**.

![Paramètres Laser](/screenshots/machine-laser.png)

Voir [Configuration Laser](../machine/laser) pour plus de détails.

### Configuration Caméra

Si vous avez une caméra USB pour l'alignement et le positionnement, configurez-la dans la page **Caméra**.

![Paramètres Caméra](/screenshots/machine-camera.png)

Voir [Intégration Caméra](../machine/camera) pour plus de détails.

### Paramètres de l'Appareil

La page **Appareil** vous permet de lire et modifier les paramètres du firmware directement sur votre appareil connecté (tels que les paramètres GRBL). C'est une fonctionnalité avancée qui doit être utilisée avec précaution.

:::warning
Modifier les paramètres de l'appareil peut être dangereux et peut rendre votre machine inopérable si des valeurs incorrectes sont appliquées !
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

Pour plus d'aide, voir [Problèmes de Connexion](../troubleshooting/connection).

---

**Suivant :** [Guide de Démarrage Rapide →](quick-start)
