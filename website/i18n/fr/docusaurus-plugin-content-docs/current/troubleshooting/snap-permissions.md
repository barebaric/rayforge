# Permissions Snap (Linux)

Cette page explique comment configurer les permissions pour Rayforge lorsqu'il est installé comme un paquet Snap sur Linux.

## Que sont les permissions Snap ?

Les Snaps sont des applications conteneurisées qui s'exécutent dans un bac à sable pour la sécurité. Par défaut, elles ont un accès limité aux ressources système. Pour utiliser certaines fonctionnalités (comme les ports série pour les contrôleurs laser), vous devez accorder explicitement des permissions.

## Permissions requises

Rayforge a besoin que ces interfaces Snap soient connectées pour une fonctionnalité complète :

| Interface | Objectif | Requis ? |
|-----------|----------|----------|
| `serial-port` | Accès aux appareils série USB (contrôleurs laser) | **Oui** (pour le contrôle machine) |
| `home` | Lire/écrire des fichiers dans votre répertoire personnel | Auto-connecté |
| `removable-media` | Accès aux lecteurs externes et stockage USB | Optionnel |
| `network` | Connectivité réseau (pour les mises à jour, etc.) | Auto-connecté |

---

## Accorder l'accès au port série

**C'est la permission la plus importante pour Rayforge.**

### Vérifier les permissions actuelles

```bash
# Voir toutes les connexions pour Rayforge
snap connections rayforge
```

Recherchez l'interface `serial-port`. Si elle affiche "disconnected" ou "-", vous devez la connecter.

### Connecter l'interface du port série

```bash
# Accorder l'accès au port série
sudo snap connect rayforge:serial-port
```

**Vous n'avez besoin de faire cela qu'une seule fois.** La permission persiste à travers les mises à jour de l'application et les redémarrages.

### Vérifier la connexion

```bash
# Vérifier si serial-port est maintenant connecté
snap connections rayforge | grep serial-port
```

Sortie attendue :
```
serial-port     rayforge:serial-port     :serial-port     -
```

Si vous voyez un indicateur plug/slot, la connexion est active.

---

## Accorder l'accès aux médias amovibles

Si vous voulez importer/exporter des fichiers depuis des clés USB ou des stockages externes :

```bash
# Accorder l'accès aux médias amovibles
sudo snap connect rayforge:removable-media
```

Maintenant vous pouvez accéder aux fichiers dans `/media` et `/mnt`.

---

## Dépannage des permissions Snap

### Le port série ne fonctionne toujours pas

**Après avoir connecté l'interface :**

1. **Rebranchez l'appareil USB :**
   - Débranchez votre contrôleur laser
   - Attendez 5 secondes
   - Rebranchez-le

2. **Redémarrez Rayforge :**
   - Fermez Rayforge complètement
   - Relancez depuis le menu d'application ou :
     ```bash
     snap run rayforge
     ```

3. **Vérifiez que le port apparaît :**
   - Ouvrez Rayforge → Paramètres → Machine
   - Recherchez les ports série dans la liste déroulante
   - Devrait voir `/dev/ttyUSB0`, `/dev/ttyACM0`, ou similaire

4. **Vérifiez que l'appareil existe :**
   ```bash
   # Lister les appareils série USB
   ls -l /dev/ttyUSB* /dev/ttyACM*
   ```

### "Permission refusée" malgré l'interface connectée

C'est rare mais peut arriver si :

1. **L'installation Snap est corrompue :**
   ```bash
   # Réinstaller le snap
   sudo snap refresh rayforge --devmode
   # Ou si cela échoue :
   sudo snap remove rayforge
   sudo snap install rayforge
   # Reconnecter les interfaces
   sudo snap connect rayforge:serial-port
   ```

2. **Règles udev conflictuelles :**
   - Vérifiez `/etc/udev/rules.d/` pour les règles de port série personnalisées
   - Elles pourraient être en conflit avec l'accès aux appareils Snap

3. **Refus AppArmor :**
   ```bash
   # Vérifier les refus AppArmor
   sudo journalctl -xe | grep DENIED | grep rayforge
   ```

   Si vous voyez des refus pour les ports série, il peut y avoir un conflit de profil AppArmor.

### Impossible d'accéder aux fichiers en dehors du répertoire personnel

**Par conception**, les Snaps ne peuvent pas accéder aux fichiers en dehors de votre répertoire personnel à moins que vous n'accordiez `removable-media`.

**Solutions de contournement :**

1. **Déplacez les fichiers vers votre répertoire personnel :**
   ```bash
   # Copier les fichiers SVG vers ~/Documents
   cp /une/autre/localisation/*.svg ~/Documents/
   ```

2. **Accordez l'accès removable-media :**
   ```bash
   sudo snap connect rayforge:removable-media
   ```

3. **Utilisez le sélecteur de fichiers Snap :**
   - Le sélecteur de fichiers intégré a un accès plus large
   - Ouvrez les fichiers via Fichier → Ouvrir plutôt que par arguments de ligne de commande

---

## Gestion manuelle des interfaces

### Lister toutes les interfaces disponibles

```bash
# Voir toutes les interfaces Snap sur votre système
snap interface
```

### Déconnecter une interface

```bash
# Déconnecter serial-port (si nécessaire)
sudo snap disconnect rayforge:serial-port
```

### Reconnecter après déconnexion

```bash
sudo snap connect rayforge:serial-port
```

---

## Alternative : Installer depuis les sources

Si les permissions Snap sont trop restrictives pour votre flux de travail :

**Option 1 : Compiler depuis les sources**

```bash
# Cloner le dépôt
git clone https://github.com/kylemartin57/rayforge.git
cd rayforge

# Installer les dépendances avec pixi
pixi install

# Exécuter Rayforge
pixi run rayforge
```

**Avantages :**
- Pas de restrictions de permissions
- Accès complet au système
- Débogage plus facile
- Dernière version de développement

**Inconvénients :**
- Mises à jour manuelles (git pull)
- Plus de dépendances à gérer
- Pas de mises à jour automatiques

**Option 2 : Utiliser Flatpak (si disponible)**

Flatpak a un bac à sable similaire mais parfois avec des modèles de permissions différents. Vérifiez si Rayforge propose un paquet Flatpak.

---

## Bonnes pratiques pour les permissions Snap

### Connectez uniquement ce dont vous avez besoin

Ne connectez pas les interfaces que vous n'utilisez pas :

- ✓ Connectez `serial-port` si vous utilisez un contrôleur laser
- ✓ Connectez `removable-media` si vous importez depuis des clés USB
- ✗ Ne connectez pas tout "au cas où" - cela va à l'encontre de l'objectif de sécurité

### Vérifiez la source du Snap

Installez toujours depuis le Snap Store officiel :

```bash
# Vérifier l'éditeur
snap info rayforge
```

Recherchez :
- Éditeur vérifié
- Source du dépôt officiel
- Mises à jour régulières

---

## Comprendre le bac à sable Snap

### Que peuvent accéder les Snaps par défaut ?

**Autorisé :**
- Fichiers dans votre répertoire personnel
- Connexions réseau
- Affichage/audio

**Non autorisé sans permission explicite :**
- Ports série (appareils USB)
- Médias amovibles
- Fichiers système
- Répertoires personnels d'autres utilisateurs

### Pourquoi c'est important pour Rayforge

Rayforge a besoin de :

1. **Accès au répertoire personnel** (auto-accordé)
   - Pour sauvegarder les fichiers de projet
   - Pour lire les fichiers SVG/DXF importés
   - Pour stocker les préférences

2. **Accès au port série** (doit être accordé)
   - Pour communiquer avec les contrôleurs laser
   - **C'est la permission critique**

3. **Médias amovibles** (optionnel)
   - Pour importer des fichiers depuis des clés USB
   - Pour exporter le G-code vers un stockage externe

---

## Débogage des problèmes Snap

### Activer la journalisation verbeuse Snap

```bash
# Exécuter Snap avec une sortie de débogage
snap run --shell rayforge
# Dans le shell snap :
export RAYFORGE_LOG_LEVEL=DEBUG
exec rayforge
```

### Vérifier les journaux Snap

```bash
# Voir les journaux Rayforge
snap logs rayforge

# Suivre les journaux en temps réel
snap logs -f rayforge
```

### Vérifier le journal système pour les refus

```bash
# Rechercher les refus AppArmor
sudo journalctl -xe | grep DENIED | grep rayforge

# Rechercher les événements d'appareils USB
sudo journalctl -f -u snapd
# Puis branchez votre contrôleur laser
```

---

## Obtenir de l'aide

Si vous avez toujours des problèmes liés à Snap :

1. **Vérifiez d'abord les permissions :**
   ```bash
   snap connections rayforge
   ```

2. **Essayez un test de port série :**
   ```bash
   # Si vous avez screen ou minicom installé
   sudo snap connect rayforge:serial-port
   # Puis testez dans Rayforge
   ```

3. **Signalez le problème avec :**
   - Sortie de `snap connections rayforge`
   - Sortie de `snap version`
   - Sortie de `snap info rayforge`
   - Votre version de distribution Ubuntu/Linux
   - Messages d'erreur exacts

4. **Envisagez des alternatives :**
   - Installez depuis les sources (voir ci-dessus)
   - Utilisez un format de paquet différent (AppImage, Flatpak)

---

## Commandes de référence rapide

```bash
# Accorder l'accès au port série (le plus important)
sudo snap connect rayforge:serial-port

# Accorder l'accès aux médias amovibles
sudo snap connect rayforge:removable-media

# Vérifier les connexions actuelles
snap connections rayforge

# Voir les journaux Rayforge
snap logs rayforge

# Rafraîchir/mettre à jour Rayforge
sudo snap refresh rayforge

# Supprimer et réinstaller (dernier recours)
sudo snap remove rayforge
sudo snap install rayforge
sudo snap connect rayforge:serial-port
```

---

## Pages connexes

- [Problèmes de connexion](connection) - Dépannage de la connexion série
- [Mode débogage](debug) - Activer la journalisation de diagnostic
- [Installation](../getting-started/installation) - Guide d'installation
- [Paramètres généraux](../machine/general) - Configuration machine
