---
description: "Configurez le calibrage de la caméra dans Rayforge pour un alignement précis de la pièce. Utilisez votre caméra pour prévisualiser et positionner les designs sur les matériaux."
---

# Intégration Caméra

Rayforge supporte l'intégration de caméra USB pour l'alignement et le
positionnement précis des matériaux. La fonctionnalité de superposition caméra
te permet de voir exactement où ton laser va couper ou graver sur le matériau,
éliminant les suppositions et réduisant le gaspillage de matériau.

![Paramètres Caméra](/screenshots/machine-camera.png)

## Flux de travail de configuration

La configuration d'une caméra peut se faire soit via l'
[Assistant de caméra](#étape-2--assistant-de-caméra) guidé — un flux unique
couvrant les paramètres d'image, la calibration d'objectif et l'alignement —
soit en configurant chaque zone manuellement depuis le panneau des propriétés
de la caméra. Dans tous les cas, la configuration couvre quatre zones :

1. **Ajouter une caméra** — Connecte ta caméra et ajoute-la à la configuration
   de la machine
2. **Ajuster les paramètres d'image** — Règle la luminosité, le contraste, la
   balance des blancs et la réduction de bruit
3. **Calibrer l'objectif** — Corrige la distorsion avec la calibration
   automatique de l'assistant de caméra ou des coefficients manuels
4. **Aligner la caméra** — Mappe les pixels de la caméra aux coordonnées
   machine pour un positionnement précis

Le panneau des propriétés de la caméra affiche des icônes d'état pour la
calibration et l'alignement en un coup d'œil :

- ✓ **Calibration d'objectif** — La calibration a été effectuée
- ⚠ **Alignement d'image** — Avertissement lorsque l'alignement doit être
  refait (p. ex., après la calibration d'objectif)
- ✓ **Alignement d'image** — L'alignement est à jour et valide

---

## Étape 1 : Ajouter une caméra

### Prérequis Matériels

**Caméras compatibles :**

- Webcams USB (les plus courantes)
- Caméras intégrées d'ordinateur portable (si tu exécutes Rayforge sur
  un ordinateur portable près de la machine)
- Toute caméra supportée par Video4Linux2 (V4L2) sur Linux ou DirectShow
  sur Windows

**Configuration recommandée :**

- Caméra montée au-dessus de la zone de travail avec une vue claire du matériau
- Conditions d'éclairage constantes
- Caméra positionnée pour capturer la zone de travail laser
- Montage sécurisé pour éviter le mouvement de la caméra

### Ajouter une Caméra

1. **Connecte ta caméra** à ton ordinateur via USB

2. **Ouvre les Paramètres Caméra :**
   - Navigue vers **Paramètres → Préférences → Caméra**
   - Ou utilise le bouton de la barre d'outils caméra

3. **Ajoute une nouvelle caméra :**
   - Clique sur le bouton "+" pour ajouter une caméra
   - Entre un nom descriptif (ex : "Caméra Supérieure",
     "Caméra Zone de Travail")
   - Sélectionne l'appareil dans le menu déroulant
     - Sur Linux : `/dev/video0`, `/dev/video1`, etc.
     - Sur Windows : Caméra 0, Caméra 1, etc.

4. **Active la caméra :**
   - Bascule l'interrupteur d'activation de la caméra
   - Le flux en direct devrait apparaître sur ton canevas

---

## Étape 2 : Assistant de caméra

L'**Assistant de caméra** exécute toute la configuration de la caméra dans un
flux unique et guidé, couvrant les trois zones dans l'ordre — paramètres
d'image, calibration d'objectif et alignement d'image. Il est lancé depuis :

- La ligne **Assistant de caméra** du panneau des propriétés de la caméra —
  clique sur **Start**
- Automatiquement depuis l'[assistant de
  configuration](../getting-started/first-time-setup.md) lorsque tu actives
  une caméra à son étape caméra et que tu continues

### Étape 2.1 : Ajuster les paramètres d'image

![Boîte de dialogue Paramètres d'image](/screenshots/camera-image-settings.png)

Les paramètres d'image constituent la première étape de l'assistant de caméra
— il s'y ouvre et te permet de définir la résolution, la balance des blancs,
la luminosité, le contraste et la réduction de bruit. Si tu n'as pas lancé
l'assistant, ou si tu souhaites ajuster les valeurs qu'il a définies, clique
sur **Configure** à côté de **Paramètres d'image** dans les propriétés de la
caméra pour ouvrir la boîte de dialogue des paramètres d'image. Ajuste ces
paramètres pour obtenir une vue caméra claire :

| Paramètre              | Description                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| **Luminosité**         | Luminosité globale de l'image (-100 à +100)                                                     |
| **Contraste**          | Définition des bords et contraste (0 à 100)                                                     |
| **Préférer YUYV**      | Utilise le YUYV non compressé au lieu de MJPEG. Plus lent mais peut corriger certains problèmes |
| **Transparence**       | Opacité de la superposition sur le canevas (0% opaque à 100% transparent)                       |
| **Balance des Blancs** | Correction de la température de couleur (Auto ou 2500-10000K)                                   |
| **Réduction de Bruit** | Réduction de bruit temporelle (0.0 à 0.95)                                                      |

L'option YUYV est utile si ta caméra produit des images teintées de vert avec
le format MJPEG par défaut. Note que le YUYV est non compressé et peut réduire
la résolution disponible ou le taux d'images sur les connexions USB 2.0.

### Étape 2.2 : Calibration d'objectif

Si ta caméra a un objectif grand angle ou est montée en angle, l'image
peut montrer une courbure visible — les lignes droites apparaissent
courbées, surtout près des bords de l'image. C'est ce qu'on appelle la
distorsion d'objectif, et elle peut fausser l'alignement même si tes
points d'alignement sont mesurés avec soin.

La calibration d'objectif est la deuxième étape de l'assistant de caméra.
Elle te permet de choisir comment corriger la distorsion :

- **Automatic** — capture des images d'une carte de calibration imprimée ;
  l'assistant calcule le modèle de distorsion pour toi
- **Manual** — saisir à la main les coefficients radiaux (k1–k3) et
  tangentiels (p1–p2)
- **Skip** — laisser la distorsion non corrigée ; tu pourras calibrer plus
  tard

#### Calibration automatique

Pour la calibration **Automatic**, l'assistant te guide pour capturer
plusieurs images d'une carte de calibration imprimée depuis différentes
positions sur le lit, puis calcule un modèle de distorsion automatiquement.

![Assistant — Paramètres de la carte](/screenshots/camera-lens-calibration-wizard-card.png)

1. Définis la **Largeur** et la **Hauteur** de ta carte imprimée. L'aperçu
   se met à jour en temps réel — la carte doit couvrir environ 70% de la vue
   caméra.
2. Clique sur **Save to PDF** pour exporter la carte à imprimer, puis
   imprime-la et place-la sur le lit laser.

![Assistant — Capture](/screenshots/camera-lens-calibration-wizard-capture.png)

3. Déplace la carte à différentes positions et angles dans la vue caméra et
   clique sur **Capture Frame** pour chaque position. Vise au moins 8 captures
   couvrant tout le cadre, y compris les coins et les bords. La barre de
   progression et les indicateurs d'état montrent la qualité de la capture.
4. Une fois suffisamment d'images capturées, l'assistant calcule le modèle de
   distorsion et l'applique — la superposition caméra affiche maintenant une
   image corrigée et droite.

#### Calibration manuelle

![Boîte de dialogue Calibration d'objectif](/screenshots/camera-lens-calibration.png)

Pour les coefficients manuels ou pour ajuster finement le résultat après une
calibration automatique, ouvre la boîte de dialogue de calibration d'objectif
en cliquant sur **Configure** à côté de **Calibration d'objectif** dans les
propriétés de la caméra. À partir de là, tu peux ajuster manuellement les
coefficients de distorsion — ajuste finement les paramètres de distorsion
radiale (k1–k3) et tangentielle (p1–p2).

### Étape 2.3 : Alignement d'image

![Boîte de dialogue Alignement d'image](/screenshots/camera-image-alignment.png)

L'alignement d'image est l'étape finale de l'assistant de caméra. L'alignement
de la caméra calibre la relation entre les pixels de la caméra et les
coordonnées du monde réel, permettant un positionnement précis. L'assistant
utilise la même procédure décrite ici, et appliquer l'alignement termine
l'assistant.

#### Pourquoi l'Alignement est Nécessaire

La caméra voit la zone de travail d'en haut, mais l'image peut être :

- Tournée par rapport aux axes de la machine
- Mise à l'échelle différemment dans les directions X et Y
- Déformée par la perspective de l'objectif

L'alignement crée une matrice de transformation qui mappe les pixels de la
caméra aux coordonnées machine.

#### Procédure d'Alignement

1. **Ouvre la Boîte de Dialogue d'Alignement :**
   - Clique sur le bouton **Configure** à côté de **Alignement d'image** dans
     les propriétés de la caméra
   - La boîte de dialogue affiche le flux caméra avec la superposition
     d'alignement actuelle

2. **Place des marqueurs d'alignement :**
   - Tu as besoin d'au moins 3 points de référence (4 recommandés pour une
     meilleure précision)
   - Les points d'alignement doivent être répartis sur la zone de travail
   - Utilise des positions connues comme :
     - Position d'origine de la machine
     - Marques de règle
     - Trous d'alignement pré-découpés
     - Grille de calibration

3. **Marque les points d'image :**
   - Clique sur l'image de la caméra pour placer un point à un emplacement
     connu
   - Le widget bulle apparaît montrant les coordonnées du point
   - Répète pour chaque point de référence

4. **Entre les coordonnées monde :**
   - Pour chaque point d'image, entre les coordonnées X/Y réelles en mm
   - Ce sont les coordonnées machine réelles où chaque point est situé
   - Mesure avec précision avec une règle ou utilise des positions machine
     connues

5. **Applique l'alignement :**
   - Clique sur **Appliquer** pour calculer la transformation
   - La superposition caméra sera maintenant correctement alignée

6. **Vérifie l'alignement :**
   - Déplace la tête laser à une position connue
   - Vérifie que le point laser s'aligne avec la position attendue dans la vue
     caméra
   - Ajuste finement en ré-alignant si nécessaire

#### Statut d'Alignement

Le panneau des propriétés de la caméra affiche le statut d'alignement avec une
icône :

- **Coche** — L'alignement est à jour et valide
- **Avertissement** — L'alignement doit être refait. Cela se produit lorsque
  la calibration d'objectif est mise à jour, car la correction de distorsion
  modifie l'image de la caméra et invalide l'alignement existant. Tes points
  d'alignement sont conservés — il suffit d'ouvrir la boîte de dialogue et de
  cliquer à nouveau sur **Appliquer**.

#### Exemple de flux de travail

1. Déplace le laser à la position d'origine (0, 0) et marque dans la caméra
2. Déplace le laser à (100, 0) et marque dans la caméra
3. Déplace le laser à (100, 100) et marque dans la caméra
4. Déplace le laser à (0, 100) et marque dans la caméra
5. Entre les coordonnées exactes pour chaque point
6. Clique sur **Appliquer** et vérifie

:::tip Meilleures Pratiques

- Utilise des points aux coins de ta zone de travail pour une couverture
  maximale
- Évite de regrouper les points dans une zone
- Mesure les coordonnées monde avec soin — la précision ici détermine la
  qualité globale de l'alignement
- Ré-aligne si tu déplaces la caméra ou changes la distance de mise au point
- Ré-aligne après la mise à jour de la calibration d'objectif
- Sauvegarde ton alignement — il persiste entre les sessions
  :::

---

## Utiliser la Superposition Caméra

Une fois alignée, la superposition caméra aide à positionner les travaux
avec précision. Active-la en cliquant sur l'icône caméra dans la barre
d'outils de la fenêtre principale.

---

### Caméras Multiples

Rayforge supporte plusieurs caméras pour différentes vues ou machines :

- Ajoute plusieurs caméras dans les préférences
- Chaque caméra peut avoir un alignement indépendant
- Bascule entre les caméras en utilisant le sélecteur de caméra
- Cas d'utilisation :
  - Vue de dessus + vue latérale pour objets 3D
  - Différentes caméras pour différentes machines
  - Grand angle + caméra de détail

---

## Dépannage

### Caméra Non Détectée

**Problème :** La caméra n'apparaît pas dans la liste des appareils.

**Solutions :**

**Linux :**
Vérifie si la caméra est reconnue par le système :

```bash
# Lister les appareils vidéo
ls -l /dev/video*

# Vérifier la caméra avec v4l2
v4l2-ctl --list-devices

# Tester avec une autre application
cheese  # ou VLC, etc.
```

**Pour les utilisateurs Snap :**

```bash
# Accorder l'accès caméra
sudo snap connect rayforge:camera
```

**Windows :**

- Vérifie le Gestionnaire de Périphériques pour la caméra sous "Caméras"
  ou "Appareils d'imagerie"
- Assure-toi qu'aucune autre application n'utilise la caméra (ferme Zoom,
  Skype, etc.)
- Essaye un port USB différent
- Mets à jour les pilotes de la caméra

### La Caméra Affiche un Écran Noir

**Problème :** Caméra détectée mais n'affiche pas d'image.

**Causes possibles :**

1. **Caméra utilisée par une autre application** - Ferme les autres
   applications vidéo
2. **Mauvais appareil sélectionné** - Essaye différents IDs d'appareil
3. **Permissions caméra** - Sur Linux Snap, assure-toi que l'interface
   caméra est connectée
4. **Problème matériel** - Teste la caméra avec une autre application

**Solutions :**

```bash
# Linux : Libérer l'appareil caméra
sudo killall cheese  # ou autres applications caméra

# Vérifier quel processus utilise la caméra
sudo lsof /dev/video0
```

### Alignement Non Précis

**Problème :** La superposition caméra ne correspond pas à la position
réelle du laser.

**Diagnostic :**

1. **Points d'alignement insuffisants** - Utilise au moins 4 points
2. **Erreurs de mesure** - Vérifie les coordonnées monde
3. **Caméra déplacée** - Ré-aligne si la position de la caméra a changé
4. **Distorsion non-linéaire** - Peut nécessiter une calibration d'objectif

**Améliorer la précision :**

- Utilise plus de points d'alignement (6-8 pour de très grandes zones)
- Répartis les points sur toute la zone de travail
- Mesure les coordonnées monde très soigneusement
- Utilise les commandes de mouvement machine pour positionner
  précisément le laser aux coordonnées connues
- Ré-aligne après tout ajustement de la caméra

### Mauvaise Qualité d'Image

**Problème :** L'image caméra est floue, sombre ou délavée.

**Solutions :**

1. **Ajuste la luminosité/contraste** dans les paramètres caméra
2. **Améliore l'éclairage** - Ajoute un éclairage de zone de travail constant
3. **Nettoie l'objectif de la caméra** - La poussière et les débris
   réduisent la clarté
4. **Vérifie la mise au point** - L'autofocus peut ne pas bien
   fonctionner ; utilise le manuel si possible
5. **Réduis temporairement la transparence** pour voir l'image caméra
   plus clairement
6. **Essaye différents paramètres de balance des blancs**
7. **Ajuste la réduction de bruit** si l'image apparaît granuleuse

### Lag ou Saccades de la Caméra

**Problème :** Le flux caméra en direct est saccadé ou retardé.

**Solutions :**

- Réduis la résolution caméra dans les paramètres de l'appareil (si accessible)
- Ferme les autres applications utilisant CPU/GPU
- Mets à jour les pilotes graphiques

---

## Pages Connexes

- [Vue 3D](../ui/3d-preview.md) — Visualiser l'exécution avec superposition
  caméra
- [Cadrer les Travaux](../features/framing-your-job.md) — Vérifier la position du
  travail
- [Paramètres Généraux](general) — Configuration de la machine
