# Importer des fichiers

Rayforge prend en charge l'importation de divers formats de fichiers, tant
vectoriels que matriciels. Cette page explique comment importer des fichiers et
les optimiser pour obtenir les meilleurs résultats.

## Formats de fichiers pris en charge

### Formats vectoriels

| Format    | Extension | Méthode d'importation             | Idéal pour                            |
| --------- | --------- | --------------------------------- | ------------------------------------- |
| **SVG**   | `.svg`    | Vecteurs directs ou vectorisation | Graphiques vectoriels, logos, dessins |
| **DXF**   | `.dxf`    | Vecteurs directs                  | Dessins CAO, plans techniques         |
| **PDF**   | `.pdf`    | Vecteurs directs ou vectorisation | Documents avec contenu vectoriel      |
| **Ruida** | `.rd`     | Vecteurs directs                  | Fichiers de tâches contrôleur Ruida   |

### Formats matriciels

| Format   | Extension       | Méthode d'importation | Idéal pour                           |
| -------- | --------------- | --------------------- | ------------------------------------ |
| **PNG**  | `.png`          | Vectorisation         | Photos, images avec transparence     |
| **JPEG** | `.jpg`, `.jpeg` | Vectorisation         | Photos, images à tons continus       |
| **BMP**  | `.bmp`          | Vectorisation         | Graphiques simples, captures d'écran |

:::note Importation d'images matricielles
:::

Toutes les images matricielles sont **vectorisées** pour créer des tracés
vectoriels utilisables pour les opérations laser. La qualité dépend de la
configuration de la vectorisation.

---

## Importer des fichiers

### La boîte de dialogue d'importation

Rayforge dispose d'une boîte de dialogue d'importation unifiée offrant un
aperçu en direct et des options de configuration pour tous les types de fichiers
pris en charge. La boîte de dialogue permet de :

- **Prévisualiser l'importation** avant de l'ajouter au document
- **Configurer les paramètres de vectorisation** pour les images matricielles
- **Choisir la méthode d'importation** pour les fichiers SVG (vecteurs directs
  ou vectorisation)
- **Ajuster les paramètres** comme le seuil, l'inversion et le seuil automatique

![Boîte de dialogue d'importation](/screenshots/import-dialog.png)

### Méthode 1 : Menu Fichier

1. **Importer un fichier** (ou Ctrl+I)
2. **Sélectionner votre fichier** dans le sélecteur de fichiers
3. **Configurer les paramètres d'importation** dans la boîte de dialogue
4. **Prévisualiser** le résultat avant l'importation
5. **Cliquer sur Importer** pour ajouter au canevas et à l'arborescence du
   document

### Méthode 2 : Glisser-déposer

1. **Faire glisser le fichier** depuis le gestionnaire de fichiers
2. **Déposer sur** le canevas de Rayforge
3. **Configurer les paramètres d'importation** dans la boîte de dialogue
4. **Prévisualiser** le résultat avant l'importation
5. **Cliquer sur Importer** pour ajouter au canevas et à l'arborescence du
   document

### Méthode 3 : Ligne de commande

```bash
# Ouvrir Rayforge avec un fichier
rayforge myfile.svg

# Plusieurs fichiers
rayforge file1.svg file2.dxf
```

### Redimensionnement automatique à l'importation

Lors de l'importation de fichiers plus grands que la zone de travail de votre
machine, Rayforge effectuera automatiquement les actions suivantes :

1. **Réduire** le contenu importé pour qu'il tienne dans les limites de la
   machine
2. **Préserver les proportions** lors de la mise à l'échelle
3. **Centrer** le contenu mis à l'échelle dans l'espace de travail
4. **Afficher une notification** avec la possibilité d'annuler le
   redimensionnement

La notification de redimensionnement apparaît sous forme de message toast :

- ⚠️ « L'élément importé était plus grand que la zone de travail et a été
  réduit pour s'ajuster. »
- Inclut un bouton **« Réinitialiser »** pour annuler le redimensionnement
  automatique
- Le toast reste visible jusqu'à ce qu'il soit ignoré ou que l'action de
  réinitialisation soit effectuée

Cela garantit que vos dessins s'adaptent toujours aux capacités de votre machine
tout en vous offrant la flexibilité de restaurer la taille d'origine si
nécessaire.

---

## Importation SVG

SVG (Scalable Vector Graphics) est le **format recommandé** pour les dessins
vectoriels.

### Options d'importation dans la boîte de dialogue

Lors de l'importation SVG, la boîte de dialogue propose un commutateur pour
choisir entre deux méthodes :

#### 1. Utiliser les vecteurs originaux (Recommandé)

Cette option est activée par défaut dans la boîte de dialogue d'importation.

**Fonctionnement :**

- Analyse le SVG et convertit les tracés directement en géométrie Rayforge
- Préservation haute fidélité des courbes et formes
- Conservation exacte des données vectorielles

**Avantages :**

- Meilleure qualité et précision
- Tracés modifiables
- Taille de fichier réduite

**Inconvénients :**

- Certaines fonctionnalités SVG avancées ne sont pas prises en charge
- Les SVG complexes peuvent poser problème

**À utiliser pour :**

- Dessins vectoriels propres depuis Inkscape, Illustrator
- Complexité simple à modérée
- Dessins sans fonctionnalités SVG avancées

#### 2. Vectoriser l'image

Désactivez « Utiliser les vecteurs originaux » pour utiliser cette méthode.

**Fonctionnement :**

- Rend d'abord le SVG en image matricielle
- Vectorise l'image rendue pour créer des vecteurs
- Plus compatible mais moins précis

**Avantages :**

- Gère les fonctionnalités SVG complexes
- Méthode de repli robuste
- Prend en charge les effets et filtres

**Inconvénients :**

- Perte de qualité due à la rastérisation
- Tailles de fichier plus importantes
- Moins précis

**À utiliser pour :**

- SVG dont l'importation directe échoue
- SVG avec effets, filtres, dégradés
- Quand l'importation directe produit des erreurs

### Aperçu en direct

La boîte de dialogue d'importation affiche un aperçu en direct de l'importation
de votre SVG :

- Les tracés vectoriels sont affichés en surbrillance bleue
- En mode vectorisation, l'image originale est montrée avec les tracés
  vectorisés
- L'aperçu se met à jour en temps réel lorsque vous modifiez les paramètres

### Bonnes pratiques SVG

**Préparez votre SVG pour obtenir les meilleurs résultats :**

1. **Convertir le texte en tracés :**
   - Inkscape : `Chemin → Objet en tracé`
   - Illustrator : `Texte → Vectoriser le texte`

2. **Simplifier les tracés complexes :**
   - Inkscape : `Chemin → Simplifier` (Ctrl+L)
   - Supprimer les nœuds inutiles

3. **Dégrouper les groupes imbriqués :**
   - Aplanir la hiérarchie lorsque c'est possible
   - `Objet → Dégrouper` (Ctrl+Shift+G)

4. **Supprimer les éléments masqués :**
   - Effacer les guides, grilles, lignes de construction
   - Supprimer les objets invisibles/transparentes

5. **Enregistrer en SVG simple :**
   - Inkscape : « SVG simple » ou « SVG optimisé »
   - Pas « SVG Inkscape » (contient des métadonnées supplémentaires)

6. **Vérifier les unités du document :**
   - Définir en mm ou pouces selon les besoins
   - Rayforge utilise le mm en interne

**Fonctionnalités SVG courantes pouvant ne pas être importées :**

- Dégradés (convertir en aplats ou matriciel)
- Filtres et effets (aplatir en tracés)
- Masques et tracés de détourage (étendre/aplatir)
- Images matricielles intégrées (exporter séparément)
- Texte (convertir en tracés au préalable)

---

## Importation DXF

DXF (Drawing Exchange Format) est courant dans les logiciels de CAO.

### Versions DXF

Rayforge prend en charge les formats DXF standard :

- **R12/LT2** (recommandé) - Meilleure compatibilité
- **R13, R14** - Bonne prise en charge
- **R2000+** - Fonctionne généralement, mais R12 est plus sûr

**Astuce :** Exportez en DXF R12/LT2 pour une compatibilité maximale.

### Conseils pour l'importation DXF

**Avant d'exporter depuis la CAO :**

1. **Simplifier le dessin :**
   - Supprimer les calques inutiles
   - Effacer les cotations et annotations
   - Supprimer les objets 3D (utiliser la projection 2D)

2. **Vérifier les unités :**
   - Confirmer les unités du dessin (mm vs pouces)
   - Rayforge utilise les mm par défaut

3. **Aplatir les calques :**
   - Envisager d'exporter uniquement les calques pertinents
   - Masquer ou supprimer les calques de construction

4. **Utiliser une précision appropriée :**
   - La précision laser est généralement de 0,1 mm
   - Ne pas sur-spécifier la précision

**Après l'importation :**

- Vérifier l'échelle (les unités DXF peuvent nécessiter un ajustement)
- Confirmer que tous les tracés ont été importés correctement
- Supprimer les éléments de construction indésirables

---

## Importation PDF

Les fichiers PDF peuvent contenir des graphiques vectoriels, des images
matricielles, ou les deux.

### Importation vectorielle directe

Lors de l'importation d'un PDF contenant des tracés vectoriels, Rayforge peut
les importer directement — tout comme les fichiers SVG ou DXF. Cela vous donne
une géométrie propre et redimensionnable sans perte de qualité due à la
rastérisation.

Si le PDF contient des calques, Rayforge les détecte et vous permet de choisir
lesquels importer. Chaque calque devient une pièce distincte dans votre
document. Cela fonctionne de la même manière que l'importation de calques SVG :
activez ou désactivez les calques individuels dans la boîte de dialogue
d'importation avant d'importer.

Ceci est particulièrement utile pour les PDF exportés depuis des logiciels de
conception comme Illustrator ou Inkscape, où les tracés vectoriels sont propres
et bien organisés.

### Repli : Rendu et vectorisation

Pour les PDF qui ne contiennent pas de données vectorielles exploitables —
documents numérisés, photos intégrées, ou PDF dont le texte n'a pas été
converti en contours — Rayforge peut recourir au rendu du PDF en image suivi
d'une vectorisation. Cela fonctionne comme l'importation d'images matricielles.

### Conseils pour l'importation PDF

**Pour de meilleurs résultats :**

1. **Utiliser des PDF vectoriels** : Les PDF créés depuis des logiciels
   vectoriels (Illustrator, Inkscape) donnent les résultats les plus propres
   avec l'importation directe.

2. **Vérifier les calques** : Si votre PDF comporte des calques, ils seront
   listés dans la boîte de dialogue d'importation. Sélectionnez uniquement les
   calques dont vous avez besoin.

3. **Pour les documents avec du texte** : Exportez en SVG avec les polices
   converties en tracés pour une qualité optimale, ou utilisez le repli rendu et
   vectorisation.

4. **Utiliser l'aperçu de la boîte de dialogue** : Ajustez les paramètres de
   seuil et d'inversion en mode vectorisation. L'aperçu montre exactement
   comment le PDF sera vectorisé.

---

## Importation Ruida

Les fichiers Ruida (.rd) sont des fichiers de tâches binaires propriétaires
utilisés par les contrôleurs Ruida dans de nombreuses machines de découpe laser.
Ces fichiers contiennent à la fois la géométrie vectorielle et les paramètres
laser organisés en calques (couleurs).

**Après l'importation :**

- **Vérifier l'échelle** - Confirmer que les dimensions correspondent à la
  taille attendue
- **Examiner les calques** - S'assurer que tous les calques ont été importés
  correctement
- **Valider les tracés** - Confirmer la présence de tous les tracés de découpe

### Limitations

- **Importation en lecture seule** - Les fichiers Ruida ne peuvent être
  qu'importés, pas exportés
- **Format binaire** - L'édition directe des fichiers .rd originaux n'est pas
  prise en charge
- **Fonctionnalités propriétaires** - Certaines fonctionnalités Ruida avancées
  peuvent ne pas être entièrement prises en charge

---

## Importation d'images matricielles (PNG, JPG, BMP)

Les images matricielles sont **vectorisées** pour créer des tracés vectoriels via
la boîte de dialogue d'importation.

### Processus de vectorisation dans la boîte de dialogue

**Fonctionnement :**

1. **Image chargée** dans la boîte de dialogue d'importation
2. **Aperçu en direct** affiche le résultat de la vectorisation
3. **Paramètres de vectorisation** ajustables en temps réel
4. **Tracés vectoriels créés** à partir des contours vectorisés
5. **Tracés ajoutés** au document comme pièces lors de l'importation

### Configuration de la vectorisation dans la boîte de dialogue

La boîte de dialogue d'importation fournit les paramètres ajustables suivants :

| Paramètre             | Description           | Effet                                               |
| --------------------- | --------------------- | --------------------------------------------------- |
| **Seuil automatique** | Détection automatique | Activé, trouve automatiquement le seuil optimal     |
| **Seuil**             | Coupure noir/blanc    | Plus bas = plus de détails, plus haut = plus simple |
| **Inverser**          | Inverser les couleurs | Vectorise les objets clairs sur fond sombre         |

**Les paramètres par défaut** fonctionnent bien pour la plupart des images. La
boîte de dialogue affiche un aperçu en direct qui se met à jour lorsque vous
ajustez ces paramètres, vous permettant d'affiner la vectorisation avant
l'importation.

### Préparer les images pour la vectorisation

**Pour de meilleurs résultats :**

1. **Contraste élevé :**
   - Ajuster la luminosité/le contraste dans un éditeur d'images
   - Distinction nette entre le premier plan et l'arrière-plan

2. **Arrière-plan propre :**
   - Supprimer le bruit et les artefacts
   - Arrière-plan blanc uni ou transparent

3. **Résolution appropriée :**
   - 300-500 PPP pour les photos
   - Trop élevée = vectorisation lente, trop basse = mauvaise qualité

4. **Recadrer au contenu :**
   - Supprimer les bordures inutiles
   - Se concentrer sur la zone à graver/découper

5. **Convertir en noir et blanc :**
   - Pour la découpe : noir et blanc pur
   - Pour la gravure : les niveaux de gris sont acceptables

**Outils d'édition d'images :**

- GIMP (gratuit)
- Photoshop
- Krita (gratuit)
- Paint.NET (gratuit, Windows)

### Qualité de la vectorisation

**Bons candidats pour la vectorisation :**

- Logos avec des bords nets
- Images à contraste élevé
- Dessins au trait et illustrations
- Texte (bien que le vectoriel soit préférable)

**Mauvais candidats pour la vectorisation :**

- Images basse résolution
- Photos aux bords doux
- Images avec des dégradés
- Photos très détaillées ou complexes

---

## Pages connexes

- [Formats pris en charge](formats) - Spécifications détaillées des formats
- [Exporter du G-code](exporting) - Options de sortie
- [Démarrage rapide](../getting-started/quick-start) - Tutoriel de première importation
