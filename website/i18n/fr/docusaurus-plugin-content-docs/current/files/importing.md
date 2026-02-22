# Importer des fichiers

Rayforge prend en charge l'importation de divers formats de fichiers, vectoriels et raster. Cette page explique
comment importer des fichiers et les optimiser pour obtenir les meilleurs résultats.

## Formats de fichiers pris en charge

### Formats vectoriels

| Format    | Extension | Méthode d'importation   | Meilleur pour                         |
| --------- | --------- | ----------------------- | ------------------------------------- |
| **SVG**   | `.svg`    | Vecteurs directs ou tracé | Graphiques vectoriels, logos, conceptions |
| **DXF**   | `.dxf`    | Vecteurs directs        | Dessins CAO, conceptions techniques   |
| **PDF**   | `.pdf`    | Rendu et tracé          | Documents avec contenu vectoriel      |
| **Ruida** | `.rd`     | Vecteurs directs        | Fichiers de travail contrôleur Ruida  |

### Formats raster

| Format   | Extension       | Méthode d'importation | Meilleur pour                          |
| -------- | --------------- | --------------------- | -------------------------------------- |
| **PNG**  | `.png`          | Tracer en vecteurs    | Photos, images avec transparence       |
| **JPEG** | `.jpg`, `.jpeg` | Tracer en vecteurs    | Photos, images à tons continus         |
| **BMP**  | `.bmp`          | Tracer en vecteurs    | Graphiques simples, captures d'écran   |

:::note Importation raster
:::

Toutes les images raster sont **tracées** pour créer des chemins vectoriels utilisables pour les opérations laser. La qualité dépend de la configuration du traçage.

---

## Importer des fichiers

### La boîte de dialogue d'importation

Rayforge dispose d'une boîte de dialogue d'importation unifiée qui offre un aperçu en direct et
des options de configuration pour tous les types de fichiers pris en charge. La boîte de dialogue vous permet de :

- **Prévisualiser votre importation** avant de l'ajouter au document
- **Configurer les paramètres de traçage** pour les images raster
- **Choisir la méthode d'importation** pour les fichiers SVG (vecteurs directs ou tracé)
- **Ajuster les paramètres** comme le seuil, l'inversion et le seuil automatique

![Boîte de dialogue d'importation](/screenshots/import-dialog.png)

### Méthode 1 : Menu Fichier

1. **Fichier Importer** (ou Ctrl+I)
2. **Sélectionnez votre fichier** dans le sélecteur de fichiers
3. **Configurez les paramètres d'importation** dans la boîte de dialogue d'importation
4. **Prévisualisez** le résultat avant d'importer
5. **Cliquez sur Importer** pour ajouter au canevas et à l'arborescence du document

### Méthode 2 : Glisser-déposer

1. **Faites glisser le fichier** depuis votre gestionnaire de fichiers
2. **Déposez sur** le canevas Rayforge
3. **Configurez les paramètres d'importation** dans la boîte de dialogue d'importation
4. **Prévisualisez** le résultat avant d'importer
5. **Cliquez sur Importer** pour ajouter au canevas et à l'arborescence du document

### Méthode 3 : Ligne de commande

```bash
# Ouvrir Rayforge avec un fichier
rayforge monfichier.svg

# Fichiers multiples
rayforge fichier1.svg fichier2.dxf
```

### Redimensionnement automatique à l'importation

Lors de l'importation de fichiers plus grands que la zone de travail de votre machine, Rayforge
fera automatiquement :

1. **Mise à l'échelle** du contenu importé pour s'adapter aux limites de la machine
2. **Préservation du rapport d'aspect** pendant la mise à l'échelle
3. **Centrage** du contenu mis à l'échelle dans l'espace de travail
4. **Affichage d'une notification** avec l'option d'annuler le redimensionnement

La notification de redimensionnement apparaît comme un message toast :

- ⚠️ "L'élément importé était plus grand que la zone de travail et a été mis à l'échelle pour s'adapter."
- Inclut un bouton **"Réinitialiser"** pour annuler le redimensionnement automatique
- Le toast reste visible jusqu'à ce qu'il soit ignoré ou que l'action de réinitialisation soit effectuée

Cela garantit que vos conceptions s'adaptent toujours aux capacités de votre machine tout en vous donnant
la flexibilité de restaurer la taille originale si nécessaire.

---

## Importation SVG

SVG (Scalable Vector Graphics) est le **format recommandé** pour les conceptions vectorielles.

### Options d'importation dans la boîte de dialogue

Lors de l'importation SVG, la boîte de dialogue d'importation fournit un commutateur pour choisir
entre deux méthodes :

#### 1. Utiliser les vecteurs originaux (Recommandé)

Cette option est activée par défaut dans la boîte de dialogue d'importation.

**Comment cela fonctionne :**

- Analyse le SVG et convertit les chemins directement en géométrie Rayforge
- Préservation haute fidélité des courbes et formes
- Maintient les données vectorielles exactes

**Avantages :**

- Meilleure qualité et précision
- Chemins modifiables
- Taille de fichier réduite

**Inconvénients :**

- Certaines fonctionnalités SVG avancées non prises en charge
- Les SVG complexes peuvent avoir des problèmes

**Utiliser pour :**

- Conceptions vectorielles propres depuis Inkscape, Illustrator
- Complexité simple à modérée
- Conceptions sans fonctionnalités SVG avancées

#### 2. Tracer bitmap

Désactivez "Utiliser les vecteurs originaux" pour utiliser cette méthode.

**Comment cela fonctionne :**

- Rend d'abord le SVG en une image raster
- Trace l'image rendue pour créer des vecteurs
- Plus compatible mais moins précis

**Avantages :**

- Gère les fonctionnalités SVG complexes
- Méthode de secours robuste
- Supporte les effets et filtres

**Inconvénients :**

- Perte de qualité due à la rastérisation
- Tailles de fichiers plus importantes
- Moins précis

**Utiliser pour :**

- SVG qui échouent l'importation directe
- SVG avec effets, filtres, dégradés
- Quand l'importation directe produit des erreurs

### Aperçu en direct

La boîte de dialogue d'importation affiche un aperçu en direct de la façon dont votre SVG sera importé :

- Les chemins vectoriels sont affichés en superposition bleue
- Pour le mode tracé, l'image originale est affichée avec les chemins tracés
- L'aperçu se met à jour en temps réel lorsque vous modifiez les paramètres

### Meilleures pratiques SVG

**Préparez votre SVG pour obtenir les meilleurs résultats :**

1. **Convertissez le texte en chemins :**

   - Inkscape : `Chemin → Objet en chemin`
   - Illustrator : `Type → Créer des contours`

2. **Simplifiez les chemins complexes :**

   - Inkscape : `Chemin → Simplifier` (Ctrl+L)
   - Supprimez les nœuds inutiles

3. **Dissociez les groupes imbriqués :**

   - Aplatissez la hiérarchie lorsque possible
   - `Objet → Dissocier` (Ctrl+Shift+G)

4. **Supprimez les éléments cachés :**

   - Supprimez guides, grilles, lignes de construction
   - Retirez les objets invisibles/transparents

5. **Enregistrez en SVG simple :**

   - Inkscape : "SVG simple" ou "SVG optimisé"
   - Pas "SVG Inkscape" (a des métadonnées supplémentaires)

6. **Vérifiez les unités du document :**
   - Définissez en mm ou pouces selon le cas
   - Rayforge utilise mm en interne

**Fonctionnalités SVG courantes qui peuvent ne pas s'importer :**

- Dégradés (convertissez en remplissages solides ou raster)
- Filtres et effets (aplatissez en chemins)
- Masques et chemins de détourage (étendez/applatissez)
- Images raster intégrées (exportez séparément)
- Texte (convertissez en chemins d'abord)

---

## Importation DXF

DXF (Drawing Exchange Format) est courant pour les logiciels CAO.

### Versions DXF

Rayforge prend en charge les formats DXF standard :

- **R12/LT2** (recommandé) - Meilleure compatibilité
- **R13, R14** - Bon support
- **R2000+** - Fonctionne généralement, mais R12 est plus sûr

**Astuce :** Exportez en DXF R12/LT2 pour une compatibilité maximale.

### Conseils d'importation DXF

**Avant d'exporter depuis la CAO :**

1. **Simplifiez le dessin :**

   - Supprimez les calques inutiles
   - Effacez les dimensions et annotations
   - Retirez les objets 3D (utilisez une projection 2D)

2. **Vérifiez les unités :**

   - Vérifiez les unités de dessin (mm vs pouces)
   - Rayforge suppose mm par défaut

3. **Aplatissez les calques :**

   - Envisagez d'exporter uniquement les calques pertinents
   - Masquez ou supprimez les calques de construction

4. **Utilisez une précision appropriée :**
   - La précision laser est typiquement de 0.1mm
   - Ne sur-spécifiez pas la précision

**Après importation :**

- Vérifiez l'échelle (les unités DXF peuvent nécessiter un ajustement)
- Vérifiez que tous les chemins ont été importés correctement
- Supprimez tous les éléments de construction indésirables

---

## Importation PDF

Les fichiers PDF peuvent contenir des graphiques vectoriels, des images raster, ou les deux.

### Comment fonctionne l'importation PDF

Lors de l'importation de fichiers PDF via la boîte de dialogue d'importation, Rayforge **rend le PDF**
en image, puis le **trace** pour créer des vecteurs.

**Processus :**

1. PDF rendu et affiché dans l'aperçu de la boîte de dialogue d'importation
2. Vous pouvez ajuster les paramètres de traçage en temps réel
3. Image rendue tracée en utilisant la vectorisation avec vos paramètres
4. Chemins résultants ajoutés au document lorsque vous cliquez sur Importer

**Limitations :**

- Le texte est rastérisé (non modifiable en tant que chemins)
- La qualité vectorielle dépend du DPI de rendu
- PDF multipages : seule la première page est importée

### Conseils d'importation PDF

**Meilleurs résultats :**

1. **Utilisez des PDF vectoriels :**

   - PDF créés depuis un logiciel vectoriel (Illustrator, Inkscape)
   - Pas des documents numérisés ou images intégrées

2. **Exportez en SVG à la place si possible :**

   - La plupart des logiciels de conception peuvent exporter en SVG directement
   - Le SVG aura une meilleure qualité que l'importation PDF

3. **Pour les documents avec du texte :**

   - Exportez en SVG avec les polices converties en chemins
   - Ou rendez le PDF à haute résolution (600+) et tracez

4. **Utilisez l'aperçu de la boîte de dialogue d'importation :**
   - Ajustez les paramètres de seuil et d'inversion pour les meilleurs résultats
   - L'aperçu montre exactement comment le PDF sera tracé

---

## Importation Ruida

Les fichiers Ruida (.rd) sont des fichiers de travail binaires propriétaires utilisés par les contrôleurs Ruida dans de nombreuses
machines de découpe laser. Ces fichiers contiennent à la fois la géométrie vectorielle et les paramètres laser
organisés en calques (couleurs).

**Après importation :**

- **Vérifiez l'échelle** - Vérifiez que les dimensions correspondent à la taille attendue
- **Passez en revue les calques** - Assurez-vous que tous les calques ont été importés correctement
- **Validez les chemins** - Confirmez que tous les chemins de coupe sont présents

### Limitations

- **Importation en lecture seule** - Les fichiers Ruida ne peuvent être qu'importés, pas exportés
- **Format binaire** - L'édition directe des fichiers .rd originaux non prise en charge
- **Fonctionnalités propriétaires** - Certaines fonctionnalités Ruida avancées peuvent ne pas être entièrement prises en charge

---

## Importation d'images raster (PNG, JPG, BMP)

Les images raster sont **tracées** pour créer des chemins vectoriels en utilisant la boîte de dialogue d'importation.

### Processus de traçage dans la boîte de dialogue

**Comment cela fonctionne :**

1. **Image chargée** dans la boîte de dialogue d'importation
2. **Aperçu en direct** affiche le résultat tracé
3. **Paramètres de traçage** peuvent être ajustés en temps réel
4. **Chemins vectoriels créés** à partir des bords tracés
5. **Chemins ajoutés** au document en tant que pièces lors de l'importation

### Configuration du traçage dans la boîte de dialogue

La boîte de dialogue d'importation fournit ces paramètres ajustables :

| Paramètre           | Description         | Effet                                              |
| ------------------- | ------------------- | -------------------------------------------------- |
| **Seuil automatique**| Détection automatique | Quand activé, trouve automatiquement le seuil optimal |
| **Seuil**           | Coupure noir/blanc  | Plus bas = plus de détail, plus haut = plus simple  |
| **Inverser**        | Inverser les couleurs | Tracer les objets clairs sur fond sombre           |

Les **paramètres par défaut** fonctionnent bien pour la plupart des images. La boîte de dialogue affiche un aperçu en direct
qui se met à jour lorsque vous ajustez ces paramètres, vous permettant d'affiner le tracé
avant l'importation.

### Préparer les images pour le traçage

**Pour les meilleurs résultats :**

1. **Haut contraste :**

   - Ajustez la luminosité/contraste dans un éditeur d'images
   - Distinction claire entre le premier plan et l'arrière-plan

2. **Arrière-plan propre :**

   - Supprimez le bruit et les artefacts
   - Arrière-plan blanc uni ou transparent

3. **Résolution appropriée :**

   - 300-500 DPI pour les photos
   - Trop haute = traçage lent, trop basse = mauvaise qualité

4. **Recadrez au contenu :**

   - Supprimez les bordures inutiles
   - Concentrez-vous sur la zone à graver/découper

5. **Convertissez en noir et blanc :**
   - Pour la découpe : B&W pur
   - Pour la gravure : niveaux de gris acceptables

**Outils d'édition d'images :**

- GIMP (gratuit)
- Photoshop
- Krita (gratuit)
- Paint.NET (gratuit, Windows)

### Qualité du tracé

**Bons candidats pour le tracé :**

- Logos avec bords nets
- Images à fort contraste
- Art linéaire et dessins
- Texte (bien que mieux en vecteur)

**Mauvais candidats pour le tracé :**

- Images basse résolution
- Photos avec bords doux
- Images avec dégradés
- Photos très détaillées ou complexes

---

## Pages connexes

- [Formats pris en charge](formats) - Spécifications détaillées des formats
- [Exporter du G-code](exporting) - Options de sortie
- [Démarrage rapide](../getting-started/quick-start) - Tutoriel de première importation
