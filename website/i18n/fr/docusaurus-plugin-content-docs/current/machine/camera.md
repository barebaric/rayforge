# Intégration Caméra

Rayforge supporte l'intégration de caméra USB pour l'alignement et le positionnement précis des matériaux. La fonctionnalité de superposition caméra vous permet de voir exactement où votre laser va couper ou graver sur le matériau, éliminant les suppositions et réduisant le gaspillage de matériau.

![Paramètres Caméra](/screenshots/machine-camera.png)

## Aperçu

L'intégration caméra fournit :

- **Superposition vidéo en direct** sur le canevas montrant votre matériau en temps réel
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
- Caméras intégrées d'ordinateur portable (si vous exécutez Rayforge sur un ordinateur portable près de la machine)
- Toute caméra supportée par Video4Linux2 (V4L2) sur Linux ou DirectShow sur Windows

**Configuration recommandée :**

- Caméra montée au-dessus de la zone de travail avec une vue claire du matériau
- Conditions d'éclairage constantes
- Caméra positionnée pour capturer la zone de travail laser
- Montage sécurisé pour éviter le mouvement de la caméra

### Ajouter une Caméra

1. **Connectez votre caméra** à votre ordinateur via USB

2. **Ouvrez les Paramètres Caméra :**
   - Naviguez vers **Paramètres Préférences Caméra**
   - Ou utilisez le bouton de la barre d'outils caméra

3. **Ajoutez une nouvelle caméra :**
   - Cliquez sur le bouton "+" pour ajouter une caméra
   - Entrez un nom descriptif (ex : "Caméra Supérieure", "Caméra Zone de Travail")
   - Sélectionnez l'appareil dans le menu déroulant
     - Sur Linux : `/dev/video0`, `/dev/video1`, etc.
     - Sur Windows : Caméra 0, Caméra 1, etc.

4. **Activez la caméra :**
   - Basculez l'interrupteur d'activation de la caméra
   - Le flux en direct devrait apparaître sur votre canevas

5. **Ajustez les paramètres de la caméra :**
   - **Luminosité :** Ajustez si le matériau est trop sombre/lumineux
   - **Contraste :** Améliorez la visibilité des bords
   - **Transparence :** Contrôlez l'opacité de la superposition (20-50% recommandé)
   - **Balance des Blancs :** Auto ou température Kelvin manuelle

---

## Alignement de la Caméra

L'alignement de la caméra calibre la relation entre les pixels de la caméra et les coordonnées du monde réel, permettant un positionnement précis.

### Pourquoi l'Alignement est Nécessaire

La caméra voit la zone de travail d'en haut, mais l'image peut être :

- Rotationnée par rapport aux axes de la machine
- Mise à l'échelle différemment dans les directions X et Y
- Déformée par la perspective de l'objectif

L'alignement crée une matrice de transformation qui mappe les pixels de la caméra aux coordonnées machine.

### Procédure d'Alignement

1. **Ouvrez la Boîte de Dialogue d'Alignement :**
   - Cliquez sur le bouton d'alignement caméra dans la barre d'outils
   - Ou allez à **Caméra Aligner Caméra**

2. **Placez des marqueurs d'alignement :**
   - Vous avez besoin d'au moins 3 points de référence (4 recommandés pour une meilleure précision)
   - Les points d'alignement doivent être répartis sur la zone de travail
   - Utilisez des positions connues comme :
     - Position d'origine de la machine
     - Marques de règle
     - Trous d'alignement pré-découpés
     - Grille de calibration

3. **Marquez les points d'image :**
   - Cliquez sur l'image de la caméra pour placer un point à un emplacement connu
   - Le widget bulle apparaît montrant les coordonnées du point
   - Répétez pour chaque point de référence

4. **Entrez les coordonnées monde :**
   - Pour chaque point d'image, entrez les coordonnées X/Y réelles en mm
   - Ce sont les coordonnées machine réelles où chaque point est situé
   - Mesurez avec précision avec une règle ou utilisez des positions machine connues

5. **Appliquez l'alignement :**
   - Cliquez sur "Appliquer" pour calculer la transformation
   - La superposition caméra sera maintenant correctement alignée

6. **Vérifiez l'alignement :**
   - Déplacez la tête laser à une position connue
   - Vérifiez que le point laser s'aligne avec la position attendue dans la vue caméra
   - Ajustez finement en ré-alignant si nécessaire

### Conseils d'Alignement

:::tip Meilleures Pratiques
- Utilisez des points aux coins de votre zone de travail pour une couverture maximale
- Évitez de regrouper les points dans une zone
- Mesurez les coordonnées monde avec soin - la précision ici détermine la qualité globale de l'alignement
- Ré-alignez si vous déplacez la caméra ou changez la distance de mise au point
- Sauvegardez votre alignement - il persiste entre les sessions
:::

**Exemple de flux de travail d'alignement :**

1. Déplacez le laser à la position d'origine (0, 0) et marquez dans la caméra
2. Déplacez le laser à (100, 0) et marquez dans la caméra
3. Déplacez le laser à (100, 100) et marquez dans la caméra
4. Déplacez le laser à (0, 100) et marquez dans la caméra
5. Entrez les coordonnées exactes pour chaque point
6. Appliquez et vérifiez

---

## Utiliser la Superposition Caméra

Une fois alignée, la superposition caméra aide à positionner les travaux avec précision.

### Activer/Désactiver la Superposition

- **Basculer caméra :** Cliquez sur l'icône caméra dans la barre d'outils
- **Ajuster la transparence :** Utilisez le curseur dans les paramètres caméra (20-50% fonctionne bien)
- **Rafraîchir l'image :** La caméra se met à jour continuellement lorsqu'elle est activée

### Positionner les Travaux avec la Caméra

**Flux de travail pour un placement précis :**

1. **Activez la superposition caméra** pour voir votre matériau

2. **Importez votre design** (SVG, DXF, etc.)

3. **Positionnez le design** sur le canevas :
   - Glissez le design pour l'aligner avec les caractéristiques visibles dans la caméra
   - Utilisez le zoom pour voir les détails fins
   - Faites pivoter/mettez à l'échelle si nécessaire

4. **Prévisualisez l'alignement :**
   - Utilisez le [Mode Simulation](../features/simulation-mode) pour visualiser
   - Vérifiez que les coupes/gravures seront où vous les attendez

5. **Cadrez le travail** pour vérifier le positionnement avant d'exécuter

6. **Exécutez le travail** en confiance

### Exemple : Gravure sur une Carte Pré-Imprimée

1. Placez la carte imprimée sur le lit laser
2. Activez la superposition caméra
3. Importez votre design de gravure
4. Glissez et positionnez le design pour l'aligner avec les caractéristiques imprimées
5. Ajustez finement la position avec les touches fléchées
6. Cadrez pour vérifier
7. Exécutez le travail

---

## Référence des Paramètres Caméra

### Paramètres de l'Appareil

| Paramètre | Description | Valeurs |
| --------- | ----------- | ------- |
| **Nom** | Nom descriptif pour la caméra | Tout texte |
| **ID Appareil** | Identifiant d'appareil système | `/dev/video0` (Linux), `0` (Windows) |
| **Activé** | État actif de la caméra | On/Off |

### Ajustement d'Image

| Paramètre | Description | Plage |
| --------- | ----------- | ----- |
| **Luminosité** | Luminosité globale de l'image | -100 à +100 |
| **Contraste** | Définition des bords et contraste | 0 à 100 |
| **Transparence** | Opacité de la superposition sur le canevas | 0% (opaque) à 100% (transparent) |
| **Balance des Blancs** | Correction de la température de couleur | Auto ou 2000-10000K |

### Données d'Alignement

| Propriété | Description |
| --------- | ----------- |
| **Points d'Image** | Coordonnées pixel dans l'image caméra |
| **Points Monde** | Coordonnées machine monde réel (mm) |
| **Matrice de Transformation** | Mappage calculé (interne) |

---

## Fonctionnalités Avancées

### Calibration de Caméra (Correction de Distorsion d'Objectif)

Pour un travail précis, vous pouvez calibrer la caméra pour corriger la distorsion en barillet/en coussinet :

1. **Imprimez un motif de damier** (ex : grille 8×6 avec des carrés de 25mm)
2. **Capturez 10+ images** du motif depuis différents angles/positions
3. **Utilisez les outils de calibration OpenCV** pour calculer la matrice caméra et les coefficients de distorsion
4. **Appliquez la calibration** dans Rayforge (paramètres avancés)

:::note Quand Calibrer
La correction de distorsion d'objectif n'est nécessaire que pour :

- Objectifs grand angle avec distorsion en barillet noticeable
- Travail de précision nécessitant une précision <1mm
- Grandes zones de travail où la distorsion s'accumule

La plupart des webcams standard fonctionnent bien sans calibration pour le travail laser typique.
:::

### Caméras Multiples

Rayforge supporte plusieurs caméras pour différentes vues ou machines :

- Ajoutez plusieurs caméras dans les préférences
- Chaque caméra peut avoir un alignement indépendant
- Basculez entre les caméras en utilisant le sélecteur de caméra
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
Vérifiez si la caméra est reconnue par le système :

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

- Vérifiez le Gestionnaire de Périphériques pour la caméra sous "Caméras" ou "Appareils d'imagerie"
- Assurez-vous qu'aucune autre application n'utilise la caméra (fermez Zoom, Skype, etc.)
- Essayez un port USB différent
- Mettez à jour les pilotes de la caméra

### La Caméra Affiche un Écran Noir

**Problème :** Caméra détectée mais n'affiche pas d'image.

**Causes possibles :**

1. **Caméra utilisée par une autre application** - Fermez les autres applications vidéo
2. **Mauvais appareil sélectionné** - Essayez différents IDs d'appareil
3. **Permissions caméra** - Sur Linux Snap, assurez-vous que l'interface caméra est connectée
4. **Problème matériel** - Testez la caméra avec une autre application

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

1. **Points d'alignement insuffisants** - Utilisez au moins 4 points
2. **Erreurs de mesure** - Vérifiez les coordonnées monde
3. **Caméra déplacée** - Ré-alignez si la position de la caméra a changé
4. **Distorsion non-linéaire** - Peut nécessiter une calibration d'objectif

**Améliorer la précision :**

- Utilisez plus de points d'alignement (6-8 pour de très grandes zones)
- Répartissez les points sur toute la zone de travail
- Mesurez les coordonnées monde très soigneusement
- Utilisez les commandes de mouvement machine pour positionner précisément le laser aux coordonnées connues
- Ré-alignez après tout ajustement de la caméra

### Mauvaise Qualité d'Image

**Problème :** L'image caméra est floue, sombre ou délavée.

**Solutions :**

1. **Ajustez la luminosité/contraste** dans les paramètres caméra
2. **Améliorez l'éclairage** - Ajoutez un éclairage de zone de travail constant
3. **Nettoyez l'objectif de la caméra** - La poussière et les débris réduisent la clarté
4. **Vérifiez la mise au point** - L'autofocus peut ne pas bien fonctionner ; utilisez le manuel si possible
5. **Réduisez temporairement la transparence** pour voir l'image caméra plus clairement
6. **Essayez différents paramètres de balance des blancs**

### Lag ou Saccades de la Caméra

**Problème :** Le flux caméra en direct est saccadé ou retardé.

**Solutions :**

- Réduisez la résolution caméra dans les paramètres de l'appareil (si accessible)
- Fermez les autres applications utilisant CPU/GPU
- Mettez à jour les pilotes graphiques
- Sur Linux, assurez-vous d'utiliser le backend V4L2 (automatique dans Rayforge)
- Désactivez la caméra lorsqu'elle n'est pas nécessaire pour économiser des ressources

---

## Pages Connexes

- [Mode Simulation](../features/simulation-mode) - Prévisualiser l'exécution avec superposition caméra
- [Aperçu 3D](../ui/3d-preview) - Visualiser les travaux en 3D
- [Cadrer les Travaux](../features/framing-your-job) - Vérifier la position du travail
- [Paramètres Généraux](general) - Configuration de la machine
