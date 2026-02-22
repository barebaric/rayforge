# Formats de fichiers pris en charge

Cette page fournit des informations détaillées sur tous les formats de fichiers pris en charge par Rayforge, y compris les capacités, les limitations et les recommandations.

## Aperçu des formats

### Référence rapide

| Format                 | Type     | Importation | Exportation         | Utilisation recommandée       |
| ---------------------- | -------- | ----------- | ------------------- | ----------------------------- |
| **SVG**                | Vecteur  | ✓ Direct    | ✓ Export d'objet    | Format de conception principal|
| **DXF**                | Vecteur  | ✓ Direct    | ✓ Export d'objet    | Échange CAD                   |
| **PDF**                | Mixte    | ✓ Traçage   | –                   | Export de documents (limité)  |
| **PNG**                | Raster   | ✓ Traçage   | –                   | Photos, images                |
| **JPEG**               | Raster   | ✓ Traçage   | –                   | Photos                        |
| **BMP**                | Raster   | ✓ Traçage   | –                   | Graphiques simples            |
| **RFS**                | Croquis  | ✓ Direct    | ✓ Export d'objet    | Croquis paramétriques         |
| **G-code**             | Contrôle | –           | ✓ Principal         | Sortie machine                |
| **Projet Rayforge**    | Projet   | ✓           | ✓                   | Enregistrer/charger des projets|

---

## Formats vectoriels

### SVG (Scalable Vector Graphics)

**Extension :** `.svg`
**Type MIME :** `image/svg+xml`
**Importation :** Analyse vectorielle directe ou traçage bitmap
**Exportation :** Export d'objet (géométrie uniquement)

**Qu'est-ce que le SVG ?**

SVG est un format d'image vectorielle basé sur XML. C'est le **format préféré** pour importer des conceptions dans Rayforge.

**Fonctionnalités prises en charge :**

- ✓ Chemins (lignes, courbes, arcs)
- ✓ Formes de base (rectangles, cercles, ellipses, polygones)
- ✓ Groupes et transformations
- ✓ Couleurs de contour et de remplissage
- ✓ Plusieurs calques
- ✓ Transformations de coordonnées (translation, rotation, échelle)

**Fonctionnalités non prises en charge/limitées :**

- ✗ Texte (doit être converti en chemins d'abord)
- ✗ Dégradés (simplifiés ou ignorés)
- ✗ Filtres et effets (ignorés)
- ✗ Masques et chemins de détourage (peuvent ne pas fonctionner correctement)
- ✗ Images raster intégrées (importées séparément si possible)
- ✗ Styles de contour complexes (les tirets peuvent être simplifiés)
- ✗ Symboles et éléments use (les instances peuvent ne pas se mettre à jour)

**Notes d'exportation :**

Lors de l'exportation d'une pièce vers SVG, Rayforge exporte la géométrie sous forme de chemins vectoriels avec :

- Rendu contour uniquement (pas de remplissage)
- Unités millimétriques
- Couleur de contour noire

**Meilleures pratiques :**

1. **Utilisez le format SVG simple** (pas SVG Inkscape ou autres variantes spécifiques à un outil)
2. **Convertissez le texte en chemins** avant d'exporter
3. **Simplifiez les chemins complexes** pour réduire le nombre de nœuds
4. **Aplatissez les groupes** lorsque possible
5. **Supprimez les éléments inutilisés** (guides, grilles, calques cachés)
6. **Définissez les unités du document** en mm (unité native de Rayforge)

**Recommandations logicielles :**

- **Inkscape** (gratuit) - Excellent support SVG, format natif

---

### DXF (Drawing Exchange Format)

**Extension :** `.dxf`
**Type MIME :** `application/dxf`, `image/vnd.dxf`
**Importation :** Analyse vectorielle directe
**Exportation :** Export d'objet (géométrie uniquement)

**Qu'est-ce que le DXF ?**

DXF est un format de dessin AutoCAD, largement utilisé pour l'échange CAD.

**Versions prises en charge :**

- ✓ **R12/LT2** (recommandé - meilleure compatibilité)
- ✓ R13, R14
- ✓ R2000 et versions ultérieures (fonctionne généralement, mais R12 est plus sûr)

**Entités prises en charge :**

- ✓ Lignes (LINE)
- ✓ Polylignes (LWPOLYLINE, POLYLINE)
- ✓ Arcs (ARC)
- ✓ Cercles (CIRCLE)
- ✓ Splines (SPLINE) - converties en polylignes
- ✓ Ellipses (ELLIPSE)
- ✓ Calques

**Fonctionnalités non prises en charge/limitées :**

- ✗ Entités 3D (utilisez une projection 2D)
- ✗ Dimensions et annotations (ignorées)
- ✗ Blocs/insertions (peuvent ne pas s'instancier correctement)
- ✗ Types de ligne complexes (simplifiés en solide)
- ✗ Texte (ignoré, convertissez en contours d'abord)
- ✗ Hachures (peuvent être simplifiées ou ignorées)

**Notes d'exportation :**

Lors de l'exportation d'une pièce vers DXF, Rayforge exporte :

- Lignes en tant qu'entités LWPOLYLINE
- Arcs en tant qu'entités ARC
- Courbes de Bézier en tant qu'entités SPLINE
- Unités millimétriques (INSUNITS = 4)

---

### RFS (Croquis Rayforge)

**Extension :** `.rfs`
**Type MIME :** `application/x-rayforge-sketch`
**Importation :** Direct (pièces basées sur croquis)
**Exportation :** Export d'objet (pièces basées sur croquis)

**Qu'est-ce que le RFS ?**

RFS est le format de croquis paramétrique natif de Rayforge. Il préserve tous les éléments
géométriques et contraintes paramétriques, vous permettant d'enregistrer et partager des
croquis entièrement modifiables.

**Fonctionnalités :**

- ✓ Tous les éléments géométriques (lignes, arcs, cercles, rectangles, etc.)
- ✓ Toutes les contraintes paramétriques
- ✓ Valeurs dimensionnelles et expressions
- ✓ Zones de remplissage

**Quand l'utiliser :**

- Enregistrer des conceptions paramétriques réutilisables
- Partager des croquis modifiables avec d'autres utilisateurs Rayforge
- Archiver un travail en cours

---

### PDF (Portable Document Format)

**Extension :** `.pdf`
**Type MIME :** `application/pdf`
**Importation :** Rendu en bitmap, puis tracé
**Exportation :** Non prise en charge

**Qu'est-ce que l'importation PDF ?**

Rayforge peut importer des fichiers PDF en les rastérisant d'abord, puis en les traçant en vecteurs.

**Processus :**

1. PDF rendu en image raster (300 DPI par défaut)
2. Raster tracé pour créer des chemins vectoriels
3. Chemins ajoutés au document

**Limitations :**

- **Pas de véritable importation vectorielle** - Même les PDF vectoriels sont rastérisés
- **Perte de qualité** due à la rastérisation
- **Première page uniquement** - Les PDF multipages n'importent que la page 1
- **Lent pour les PDF complexes** - Le rendu et le traçage prennent du temps

**Quand l'utiliser :**

- Dernier recours lorsque SVG/DXF n'est pas disponible
- Importation rapide de conceptions simples
- Documents avec contenu mixte

**Meilleures alternatives :**

- **Exportez en SVG depuis la source** au lieu de PDF
- **Utilisez des formats vectoriels** (SVG, DXF) lorsque possible
- **Pour le texte :** Exportez avec le texte converti en contours

---

## Formats raster

Tous les formats raster sont **importés par traçage** - convertis automatiquement en chemins vectoriels.

### PNG (Portable Network Graphics)

**Extension :** `.png`
**Type MIME :** `image/png`
**Importation :** Tracer en vecteurs
**Exportation :** Non prise en charge

**Caractéristiques :**

- **Compression sans perte** - Pas de perte de qualité
- **Support de la transparence** - Canal alpha préservé
- **Bon pour :** Logos, art linéaire, captures d'écran, tout ce qui nécessite de la transparence

**Qualité de traçage :** ★★★★★ (Excellent pour les images à fort contraste)

**Meilleures pratiques :**

- Utilisez PNG pour les logos et graphiques avec des bords nets
- Assurez un fort contraste entre le premier plan et l'arrière-plan
- L'arrière-plan transparent fonctionne mieux que blanc

---

### JPEG (Joint Photographic Experts Group)

**Extension :** `.jpg`, `.jpeg`
**Type MIME :** `image/jpeg`
**Importation :** Tracer en vecteurs
**Exportation :** Non prise en charge

**Caractéristiques :**

- **Compression avec perte** - Quelque perte de qualité
- **Pas de transparence** - A toujours un arrière-plan
- **Bon pour :** Photos, images à tons continus

**Qualité de traçage :** ★★★☆☆ (Bon pour les photos, mais complexe)

**Meilleures pratiques :**

- Utilisez un JPEG de haute qualité (faible compression)
- Augmentez le contraste avant d'importer
- Envisagez un pré-traitement dans un éditeur d'images
- Mieux vaut convertir en PNG d'abord si possible

---

### BMP (Bitmap)

**Extension :** `.bmp`
**Type MIME :** `image/bmp`
**Importation :** Tracer en vecteurs
**Exportation :** Non prise en charge

**Caractéristiques :**

- **Non compressé** - Fichiers volumineux
- **Format simple** - Largement compatible
- **Bon pour :** Graphiques simples, sortie de logiciels anciens

**Qualité de traçage :** ★★★★☆ (Bon, mais pas mieux que PNG)

**Meilleures pratiques :**

- Convertissez en PNG pour une taille de fichier plus petite (pas de différence de qualité)
- Utilisez uniquement si le logiciel source ne peut pas exporter en PNG/SVG

---

## Pages connexes

- [Importer des fichiers](importing) - Comment importer chaque format
- [Exporter](exporting) - Options d'exportation G-code
