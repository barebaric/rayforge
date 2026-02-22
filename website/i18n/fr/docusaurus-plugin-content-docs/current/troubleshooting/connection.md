# Problèmes de connexion

Cette page vous aide à diagnostiquer et résoudre les problèmes de connexion de Rayforge à votre machine laser via connexion série.

## Diagnostic rapide

### Symptômes

Les problèmes de connexion courants incluent :

- Erreur "Le port doit être configuré" lors de la tentative de connexion
- La connexion échoue et se reconnecte repeatedly
- Le port série n'apparaît pas dans la liste des ports
- Erreurs "Permission refusée" lors de la tentative d'ouverture du port série
- L'appareil semble connecté mais ne répond pas aux commandes

---

## Problèmes courants et solutions

### Aucun port série détecté

**Problème :** La liste déroulante des ports série est vide ou n'affiche pas votre appareil.

**Diagnostic :**

1. Vérifiez si votre appareil est allumé et connecté via USB
2. Essayez de débrancher et rebrancher le câble USB
3. Testez le câble USB avec un autre appareil (les câbles peuvent défaillir)
4. Essayez un port USB différent sur votre ordinateur

**Solutions :**

**Linux :**
Si vous utilisez la version Snap, vous devez accorder les permissions de port série :

```bash
sudo snap connect rayforge:serial-port
```

Voir [Permissions Snap](snap-permissions) pour la configuration détaillée de Linux.

Pour les installations non-Snap, ajoutez votre utilisateur au groupe `dialout` :

```bash
sudo usermod -a -G dialout $USER
```

Ensuite, déconnectez-vous et reconnectez-vous pour que le changement prenne effet.

**Windows :**
1. Ouvrez le Gestionnaire de périphériques (Win+X, puis sélectionnez Gestionnaire de périphériques)
2. Regardez sous "Ports (COM et LPT)" pour votre appareil
3. Si vous voyez une icône d'avertissement jaune, mettez à jour ou réinstallez le pilote
4. Notez le numéro de port COM (par exemple, COM3)
5. Si l'appareil n'est pas listé du tout, le câble USB ou le pilote peut être défectueux

**macOS :**
1. Vérifiez Informations système → USB pour vérifier que l'appareil est reconnu
2. Installez les pilotes CH340/CH341 si votre contrôleur utilise cette puce
3. Recherchez les appareils `/dev/tty.usbserial*` ou `/dev/cu.usbserial*`

### Erreurs de permission refusée

**Problème :** Vous obtenez des erreurs "Permission refusée" ou similaires lors de la tentative de connexion.

**Sur Linux (non-Snap) :**

Votre utilisateur doit être dans le groupe `dialout` (ou `uucp` sur certaines distributions) :

```bash
# Ajoutez-vous au groupe dialout
sudo usermod -a -G dialout $USER

# Vérifiez que vous êtes dans le groupe (après déconnexion/reconnexion)
groups | grep dialout
```

**Important :** Vous devez vous déconnecter et vous reconnecter (ou redémarrer) pour que les changements de groupe prennent effet.

**Sur Linux (Snap) :**

Accordez l'accès au port série au snap :

```bash
sudo snap connect rayforge:serial-port
```

Voir le guide [Permissions Snap](snap-permissions) pour plus de détails.

**Sur Windows :**

Fermez toutes les autres applications qui pourraient utiliser le port série, incluant :
- Instances précédentes de Rayforge
- Outils de surveillance série
- Autres logiciels laser
- Arduino IDE ou outils similaires

### Mauvais port série sélectionné

**Problème :** Rayforge se connecte mais la machine ne répond pas.

**Diagnostic :**

Vous avez peut-être sélectionné le mauvais port, surtout si vous avez plusieurs appareils USB connectés.

**Solution :**

1. Déconnectez tous les autres appareils série USB
2. Notez quels ports sont disponibles dans Rayforge
3. Branchez votre contrôleur laser
4. Rafraîchissez la liste des ports - le nouveau port est votre laser
5. Sur Linux, les contrôleurs laser apparaissent généralement comme :
   - `/dev/ttyUSB0` (courant pour les puces CH340)
   - `/dev/ttyACM0` (courant pour les contrôleurs USB natifs)
6. Sur Windows, notez le port COM depuis le Gestionnaire de périphériques
7. Évitez les ports nommés `/dev/ttyS*` sur Linux - ce sont des ports série matériels, pas USB

:::warning Ports série matériels
Rayforge vous avertira si vous sélectionnez des ports `/dev/ttyS*` sur Linux, car ce ne sont généralement pas des appareils GRBL basés sur USB. Les ports série USB utilisent `/dev/ttyUSB*` ou `/dev/ttyACM*`.
:::


### Débit en bauds incorrect

**Problème :** La connexion s'établit mais les commandes ne fonctionnent pas ou produisent des réponses illisibles.

**Solution :**

Les contrôleurs GRBL utilisent généralement un de ces débits en bauds :

- **115200** (le plus courant, GRBL 1.1+)
- **9600** (anciennes versions GRBL)
- **250000** (moins courant, certains firmwares personnalisés)

Essayez différents débits en bauds dans les paramètres d'appareil de Rayforge. Le plus courant est **115200**.

### La connexion se coupe constamment

**Problème :** Rayforge se connecte avec succès mais continue à se déconnecter et se reconnecter.

**Causes possibles :**

1. **Câble USB défectueux** - Remplacez par un câble connu bon (de préférence court, <2m)
2. **Problèmes d'alimentation USB** - Essayez un port USB différent, de préférence sur l'ordinateur lui-même plutôt qu'un hub
3. **EMI/Interférences** - Gardez les câbles USB éloignés des fils de moteur et des alimentations haute tension
4. **Problèmes de firmware** - Mettez à jour votre firmware GRBL si possible
5. **Conflits de port USB** - Sur Windows, essayez différents ports USB

**Étapes de dépannage :**

```bash
# Sur Linux, surveillez les journaux système pendant la connexion :
sudo dmesg -w
```

Recherchez des messages comme :
- "USB disconnect" - indique des problèmes physiques/câble
- "device descriptor read error" - souvent un problème d'alimentation ou de câble

### L'appareil ne répond pas après la connexion

**Problème :** L'état de connexion affiche "Connecté" mais la machine ne répond pas aux commandes.

**Diagnostic :**

1. Vérifiez que le bon type de firmware est sélectionné (GRBL vs autre)
2. Vérifiez que la machine est allumée (contrôleur et alimentation)
3. Vérifiez si la machine est dans un état d'alarme (nécessite une mise à l'origine ou effacement d'alarme)

**Solution :**

Essayez d'envoyer une commande manuelle dans la Console :

- `?` - Demander un rapport d'état
- `$X` - Effacer l'alarme
- `$H` - Mettre la machine à l'origine

S'il n'y a pas de réponse, vérifiez à nouveau le débit en bauds et la sélection du port.

---

## Messages d'état de connexion

Rayforge affiche différents états de connexion :

| État | Signification | Action |
|------|---------------|--------|
| **Déconnecté** | Non connecté à aucun appareil | Configurer le port et connecter |
| **Connexion** | Tentative d'établissement de connexion | Attendre, ou vérifier la configuration si bloqué |
| **Connecté** | Connecté avec succès et réception d'état | Prêt à utiliser |
| **Erreur** | La connexion a échoué avec une erreur | Vérifier le message d'erreur pour les détails |
| **Veille** | Attente avant tentative de reconnexion | La connexion précédente a échoué, nouvelle tentative dans 5s |

---

## Tester votre connexion

### Test de connexion étape par étape

1. **Configurez la machine :**
   - Ouvrez Paramètres → Machine
   - Sélectionnez ou créez un profil machine
   - Choisissez le bon pilote (GRBL Serial)
   - Sélectionnez le port série
   - Définissez le débit en bauds (généralement 115200)

2. **Tentative de connexion :**
   - Cliquez sur "Connecter" dans le panneau de contrôle machine
   - Surveillez l'indicateur d'état de connexion

3. **Vérifiez la communication :**
   - Si connecté, essayez d'envoyer une requête d'état
   - La machine devrait rapporter sa position et son état

4. **Testez les commandes de base :**
   - Essayez la mise à l'origine (`$H`) si votre machine a des butoirs
   - Ou effacez les alarmes (`$X`) si nécessaire

### Utiliser les journaux de débogage

Rayforge inclut une journalisation détaillée pour les problèmes de connexion. Pour activer la journalisation de débogage :

```bash
# Exécuter Rayforge depuis le terminal avec journalisation de débogage
rayforge --loglevel DEBUG
```

Vérifiez les journaux pour :
- Tentatives de connexion et échecs
- Données série transmises (TX) et reçues (RX)
- Messages d'erreur avec traces de pile

---

## Dépannage avancé

### Vérifier manuellement la disponibilité du port

**Linux :**
```bash
# Lister tous les appareils série USB
ls -l /dev/ttyUSB* /dev/ttyACM*

# Vérifier les permissions
ls -l /dev/ttyUSB0  # Remplacez par votre port

# Devrait afficher : crw-rw---- 1 root dialout
# Vous devez être dans le groupe 'dialout'

# Tester le port manuellement
sudo minicom -D /dev/ttyUSB0 -b 115200
```

**Windows :**
```powershell
# Lister les ports COM en PowerShell
[System.IO.Ports.SerialPort]::getportnames()

# Ou utiliser le Gestionnaire de périphériques :
# Win + X → Gestionnaire de périphériques → Ports (COM et LPT)
```

### Compatibilité du firmware

Rayforge est conçu pour les firmwares compatibles GRBL. Assurez-vous que votre contrôleur exécute :

- **GRBL 1.1** (le plus courant, recommandé)
- **GRBL 0.9** (plus ancien, peut avoir des fonctionnalités limitées)
- **grblHAL** (fork GRBL moderne, pris en charge)

D'autres types de firmware (Marlin, Smoothieware) ne sont pas actuellement pris en charge via le pilote GRBL.

### Chipsets USB-vers-série

Chipsets courants et leurs pilotes :

| Chipset | Linux | Windows | macOS |
|---------|-------|---------|-------|
| **CH340/CH341** | Pilote noyau intégré | [Pilote CH341SER](http://www.wch.cn/downloads/) | Nécessite un pilote |
| **FTDI FT232** | Pilote noyau intégré | Intégré (Windows 10+) | Intégré |
| **CP2102 (SiLabs)** | Pilote noyau intégré | Intégré (Windows 10+) | Intégré |

---

### Vous avez encore des problèmes ?

Si vous avez tout essayé ci-dessus et que vous ne pouvez toujours pas vous connecter :

1. **Vérifiez les tickets GitHub** - Quelqu'un a peut-être signalé le même problème
2. **Créez un rapport de problème détaillé** avec :
   - Système d'exploitation et version
   - Version Rayforge (Snap/Flatpak/AppImage/source)
   - Modèle de carte contrôleur et version du firmware
   - Chipset USB (vérifiez le Gestionnaire de périphériques sur Windows ou `lsusb` sur Linux)
   - Messages d'erreur complets et journaux de débogage
3. **Testez avec une autre application** - Essayez de vous connecter avec un terminal série (minicom, PuTTY, Moniteur série Arduino) pour vérifier que le matériel fonctionne

---

## Pages connexes

- [Permissions Snap](snap-permissions) - Configuration des permissions Snap Linux
- [Mode débogage](debug) - Activer la journalisation de diagnostic
- [Paramètres généraux](../machine/general) - Guide de configuration machine
- [Paramètres de l'appareil](../machine/device) - Référence de configuration GRBL
