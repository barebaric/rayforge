# Intégration Caméra

Rayforge supporte l'intégration de caméra USB pour l'alignement et le positionnement précis des matériaux. La fonctionnalité de superposition caméra te permet de voir exactement où ton laser va couper ou graver sur le matériau, éliminant les suppositions et réduisant le gaspillage de matériau.

![Paramètres Caméra](/screenshots/machine-camera.png)

## Aperçu

L'intégration caméra fournit :

- **Superposition vidéo en direct** sur le canevas montrant ton matériau en temps réel
- **Alignement d'image** pour calibrer la position de la caméra par rapport au laser
- **Positionnement visuel** pour placer avec précision les travaux sur des matériaux irréguliers ou pré-marqués
- **Aperçu du matériau** avant d'exécuter les travaux
- **Support multi-caméras** pour différentes configurations de machine

:::tip Cas d'Utilisation

- Aligner les coupes sur des matériaux pré-imprimés
- Travailler avec des matériaux de forme irrégulière
- Placement précis de gravures sur des objets existants
- Réduire les tests de coupe et le gaspillage de matériau
:::

---

## Configuration de la Caméra

### Prérequis Matériels

**Caméras compatibles :**

- Webcams USB (les plus courantes)
- Caméras intégrées d'ordinateur portable (si tu exécutes Rayforge sur un ordinateur portable près de la machine)
- Toute caméra supportée par Video4Linux2 (V4L2) sur Linux ou DirectShow sur Windows

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
   - Entre un nom descriptif (ex : "Caméra Supérieure", "Caméra Zone de Travail")
   - Sélectionne l'appareil dans le menu déroulant
     - Sur Linux : `/dev/video0`, `/dev/video1`, etc.
     - Sur Windows : Caméra 0, Caméra 1, etc.

4. **Active la caméra :**
   - Bascule l'interrupteur d'activation de la caméra
   - Le flux en direct devrait apparaître sur ton canevas

5. **Ajuste les paramètres de la caméra :**
   - **Luminosité :** Ajuste si le matériau est trop sombre/lumineux
   - **Contraste :** Améliore la visibilité des bords
   - **Transparence :** Contrôle l'opacité de la superposition (20-50% recommandé)
   - **Balance des Blancs :** Auto ou température Kelvin manuelle

---

## Alignement de la Caméra

L'alignement de la caméra calibre la relation entre les pixels de la caméra et les coordonnées du monde réel, permettant un positionnement précis.

### Pourquoi l'Alignement est Nécessaire

La caméra voit la zone de travail d'en haut, mais l'image peut être :

- Tournée par rapport aux axes de la machine
- Mise à l'échelle différemment dans les directions X et Y
- Déformée par la perspective de l'objectif

L'alignement crée une matrice de transformation qui mappe les pixels de la caméra aux coordonnées machine.

### Procédure d'Alignement

1. **Ouvre la Boîte de Dialogue d'Alignement :**
   - Clique sur le bouton d'alignement caméra dans la barre d'outils
   - Ou va à **Caméra → Aligner Caméra**

2. **Place des marqueurs d'alignement :**
   - Tu as besoin d'au moins 3 points de référence (4 recommandés pour une meilleure précision)
   - Les points d'alignement doivent être répartis sur la zone de travail
   - Utilise des positions connues comme :
     - Position d'origine de la machine
     - Marques de règle
     - Trous d'alignement pré-découpés
     - Grille de calibration

3. **Marque les points d'image :**
   - Clique sur l'image de la caméra pour placer un point à un emplacement connu
   - Le widget bulle apparaît montrant les coordonnées du point
   - Répète pour chaque point de référence

4. **Entre les coordonnées monde :**
   - Pour chaque point d'image, entre les coordonnées X/Y réelles en mm
   - Ce sont les coordonnées machine réelles où chaque point est situé
   - Mesure avec précision avec une règle ou utilise des positions machine connues

5. **Applique l'alignement :**
   - Clique sur "Appliquer" pour calculer la transformation
   - La superposition caméra sera maintenant correctement alignée

6. **Vérifie l'alignement :**
   - Déplace la tête laser à une position connue
   - Vérifie que le point laser s'aligne avec la position attendue dans la vue caméra
   - Ajuste finement en ré-alignant si nécessaire

### Conseils d'Alignement

:::tip Meilleures Pratiques

- Utilise des points aux coins de ta zone de travail pour une couverture maximale
- Évite de regrouper les points dans une zone
- Mesure les coordonnées monde avec soin - la précision ici détermine la qualité globale de l'alignement
- Ré-aligne si tu déplaces la caméra ou changes la distance de mise au point
- Sauvegarde ton alignement - il persiste entre les sessions
:::

**Exemple de flux de travail d'alignement :**

1. Déplace le laser à la position d'origine (0, 0) et marque dans la caméra
2. Déplace le laser à (100, 0) et marque dans la caméra
3. Déplace le laser à (100, 100) et marque dans la caméra
4. Déplace le laser à (0, 100) et marque dans la caméra
5. Entre les coordonnées exactes pour chaque point
6. Applique et vérifie

---

## Utiliser la Superposition Caméra

Une fois alignée, la superposition caméra aide à positionner les travaux avec précision.

### Activer/Désactiver la Superposition

- **Basculer caméra :** Clique sur l'icône caméra dans la barre d'outils
- **Ajuster la transparence :** Utilise le curseur dans les paramètres caméra (20-50% fonctionne bien)
- **Rafraîchir l'image :** La caméra se met à jour continuellement lorsqu'elle est activée

### Positionner les Travaux avec la Caméra

**Flux de travail pour un placement précis :**

1. **Active la superposition caméra** pour voir ton matériau

2. **Importe ton design** (SVG, DXF, etc.)

3. **Positionne le design** sur le canevas :
   - Glisse le design pour l'aligner avec les caractéristiques visibles dans la caméra
   - Utilise le zoom pour voir les détails fins
   - Fais pivoter/mets à l'échelle si nécessaire

4. **Prévisualise l'alignement :**
   - Utilise le [Mode Simulation](../features/simulation-mode) pour visualiser
   - Vérifie que les coupes/gravures seront où tu les attends

5. **Cadre le travail** pour vérifier le positionnement avant d'exécuter

6. **Exécute le travail** en confiance

### Exemple : Gravure sur une Carte Pré-Imprimée

1. Place la carte imprimée sur le lit laser
2. Active la superposition caméra
3. Importe ton design de gravure
4. Glisse et positionne le design pour l'aligner avec les caractéristiques imprimées
5. Ajuste finement la position avec les touches fléchées
6. Cadre pour vérifier
7. Exécute le travail

---

## Référence des Paramètres Caméra

### Paramètres de l'Appareil

| Paramètre | Description | Valeurs |
| --------- | ----------- | ------- |
| **Nom** | Nom descriptif pour la caméra | Tout texte |
| **ID Appareil** | Identifiant d'appareil système | `/dev/video0` (Linux), `0` (Windows) |
| **Activé** | État actif de la caméra | On/Off |

### Ajustement d'Image

| Paramètre | Description |
| --------- | ----------- |
| **Luminosité** | Luminosité globale de l'image (-100 à +100) |
| **Contraste** | Définition des bords et contraste (0 à 100) |
| **Préférer YUYV** | Utilise le YUYV non compressé au lieu de MJPEG. Plus lent mais peut corriger certains problèmes |
| **Transparence** | Opacité de la superposition sur le canevas (0% opaque à 100% transparent) |
| **Balance des Blancs** | Correction de la température de couleur (Auto ou 2500-10000K) |
| **Réduction de Bruit** | Réduction de bruit temporelle (0.0 à 0.95) |

L'option YUYV est utile si ta caméra produit des images teintées de vert avec le
format MJPEG par défaut. Note que le YUYV est non compressé et peut réduire la
résolution disponible ou le taux d'images sur les connexions USB 2.0.

### Données d'Alignement

| Propriété | Description |
| --------- | ----------- |
| **Points d'Image** | Coordonnées pixel dans l'image caméra |
| **Points Monde** | Coordonnées machine monde réel (mm) |
| **Matrice de Transformation** | Mappage calculé (interne) |

---

## Fonctionnalités Avancées

### Calibration de Caméra (Correction de Distorsion d'Objectif)

Si ta caméra a un objectif grand angle ou est montée en angle, l'image peut montrer
une courbure visible — les lignes droites apparaissent courbées, surtout près des bords
de l'image. C'est ce qu'on appelle la distorsion d'objectif, et elle peut fausser
l'alignement même si tes points d'alignement sont mesurés avec soin.

Rayforge inclut un assistant de calibration guidé qui corrige cette distorsion
automatiquement. Voici comment ça fonctionne :

1. **Imprime la carte de calibration** — Rayforge fournit un motif imprimable (une grille de
   marqueurs) que tu places sur ton lit laser
2. **Suis l'assistant** — L'assistant de calibration te guide pour capturer plusieurs
   images de la carte depuis différentes positions sur le lit
3. **Applique la correction** — Rayforge calcule un modèle de distorsion à partir des images
   capturées et l'utilise pour redresser la superposition caméra

Une fois calibrée, la superposition caméra affichera une représentation nettement plus précise
de ce qui se trouve sur le lit. C'est particulièrement utile pour les objectifs grand angle,
les caméras montées hors centre, ou les travaux nécessitant des tolérances d'alignement serrées.

:::note Quand Calibrer
La calibration est surtout utile lorsque tu remarques que la superposition caméra ne s'aligne pas bien
avec le lit réel, même après un alignement soigné. Si ton alignement actuel semble bon, tu n'en as
peut-être pas besoin. Mais si les choses semblent légèrement décalées — surtout vers les bords de
l'image — passer par l'assistant de calibration aide généralement.
:::

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

- Vérifie le Gestionnaire de Périphériques pour la caméra sous "Caméras" ou "Appareils d'imagerie"
- Assure-toi qu'aucune autre application n'utilise la caméra (ferme Zoom, Skype, etc.)
- Essaye un port USB différent
- Mets à jour les pilotes de la caméra

### La Caméra Affiche un Écran Noir

**Problème :** Caméra détectée mais n'affiche pas d'image.

**Causes possibles :**

1. **Caméra utilisée par une autre application** - Ferme les autres applications vidéo
2. **Mauvais appareil sélectionné** - Essaye différents IDs d'appareil
3. **Permissions caméra** - Sur Linux Snap, assure-toi que l'interface caméra est connectée
4. **Problème matériel** - Teste la caméra avec une autre application

**Solutions :**

```bash
# Linux : Libérer l'appareil caméra
sudo killall cheese  # ou autres applications caméra

# Vérifier quel processus utilise la caméra
sudo lsof /dev/video0
```

### Alignement Non Précis

**Problème :** La superposition caméra ne correspond pas à la position réelle du laser.

**Diagnostic :**

1. **Points d'alignement insuffisants** - Utilise au moins 4 points
2. **Erreurs de mesure** - Vérifie les coordonnées monde
3. **Caméra déplacée** - Ré-aligne si la position de la caméra a changé
4. **Distorsion non-linéaire** - Peut nécessiter une calibration d'objectif

**Améliorer la précision :**

- Utilise plus de points d'alignement (6-8 pour de très grandes zones)
- Répartis les points sur toute la zone de travail
- Mesure les coordonnées monde très soigneusement
- Utilise les commandes de mouvement machine pour positionner précisément le laser aux coordonnées connues
- Ré-aligne après tout ajustement de la caméra

### Mauvaise Qualité d'Image

**Problème :** L'image caméra est floue, sombre ou délavée.

**Solutions :**

1. **Ajuste la luminosité/contraste** dans les paramètres caméra
2. **Améliore l'éclairage** - Ajoute un éclairage de zone de travail constant
3. **Nettoie l'objectif de la caméra** - La poussière et les débris réduisent la clarté
4. **Vérifie la mise au point** - L'autofocus peut ne pas bien fonctionner ; utilise le manuel si possible
5. **Réduis temporairement la transparence** pour voir l'image caméra plus clairement
6. **Essaye différents paramètres de balance des blancs**
7. **Ajuste la réduction de bruit** si l'image apparaît granuleuse

### Lag ou Saccades de la Caméra

**Problème :** Le flux caméra en direct est saccadé ou retardé.

**Solutions :**

- Réduis la résolution caméra dans les paramètres de l'appareil (si accessible)
- Ferme les autres applications utilisant CPU/GPU
- Mets à jour les pilotes graphiques
- Sur Linux, assure-toi d'utiliser le backend V4L2 (automatique dans Rayforge)
- Désactive la caméra lorsqu'elle n'est pas nécessaire pour économiser des ressources

---

## Pages Connexes

- [Mode Simulation](../features/simulation-mode) - Prévisualiser l'exécution avec superposition caméra
- [Vue 3D](../ui/3d-preview) - Visualiser les travaux en 3D
- [Cadrer les Travaux](../features/framing-your-job) - Vérifier la position du travail
- [Paramètres Généraux](general) - Configuration de la machine
