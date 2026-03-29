# Formats de fichier pris en charge

Cette page fournit des informations détaillées sur tous les formats de fichier
pris en charge par Rayforge, y compris les capacités, les limitations et les
recommandations.

## Aperçu des formats

### Référence rapide

| Format              | Type      | Importation      | Exportation   | Utilisation recommandée          |
| ------------------- | --------- | ---------------- | ------------- | -------------------------------- |
| **SVG**             | Vecteur   | ✓ Direct / Trace | ✓ Export obj. | Format de conception principal   |
| **DXF**             | Vecteur   | ✓ Direct         | ✓ Export obj. | Échange CAO                      |
| **PDF**             | Mixte     | ✓ Direct / Trace | –             | Documents avec contenu vectoriel |
| **PNG**             | Matriciel | ✓ Trace          | –             | Photos, images                   |
| **JPEG**            | Matriciel | ✓ Trace          | –             | Photos                           |
| **BMP**             | Matriciel | ✓ Trace          | –             | Graphiques simples               |
| **RFS**             | Croquis   | ✓ Direct         | ✓ Export obj. | Croquis paramétriques            |
| **G-code**          | Commande  | –                | ✓ Principal   | Sortie machine                   |
| **Projet Rayforge** | Projet    | ✓                | ✓             | Enregistrer/charger projets      |

---

## Formats vectoriels

### SVG (Scalable Vector Graphics)

**Extension :** `.svg`
**Type MIME :** `image/svg+xml`
**Importation :** Analyse vectorielle directe ou tracé bitmap
**Exportation :** Export d'objet (géométrie uniquement)

**Qu'est-ce que le SVG ?**

Le SVG est un format d'image vectorielle basé sur XML. C'est le **format
préféré** pour importer des conceptions dans Rayforge.

**Fonctionnalités prises en charge :**

- ✓ Chemins (lignes, courbes, arcs)
- ✓ Formes de base (rectangles, cercles, ellipses, polygones)
- ✓ Groupes et transformations
- ✓ Couleurs de trait et de remplissage
- ✓ Plusieurs calques
- ✓ Transformations de coordonnées (translation, rotation, mise à l'échelle)

**Fonctionnalités non prises en charge/limitées :**

- ✗ Texte (doit être converti en chemins au préalable)
- ✗ Dégradés (simplifiés ou ignorés)
- ✗ Filtres et effets (ignorés)
- ✗ Masques et chemins de détourage (peuvent ne pas fonctionner correctement)
- ✗ Images matricielles intégrées (importées séparément si possible)
- ✗ Styles de trait complexes (les tirets peuvent être simplifiés)
- ✗ Symboles et éléments use (les instances peuvent ne pas se mettre à jour)

**Notes d'exportation :**

Lors de l'exportation d'une pièce vers SVG, Rayforge exporte la géométrie sous
forme de chemins vectoriels avec :

- Rendu trait uniquement (sans remplissage)
- Unités en millimètres
- Couleur de trait noire

**Bonnes pratiques :**

1. **Utilisez le format SVG simple** (pas Inkscape SVG ni d'autres variantes
   spécifiques à un outil)
2. **Convertissez le texte en chemins** avant l'exportation
3. **Simplifiez les chemins complexes** pour réduire le nombre de nœuds
4. **Aplatissez les groupes** lorsque c'est possible
5. **Supprimez les éléments inutilisés** (guides, grilles, calques masqués)
6. **Définissez les unités du document** en mm (unité native de Rayforge)

**Recommandations logicielles :**

- **Inkscape** (gratuit) - Excellente prise en charge SVG, format natif

---

### DXF (Drawing Exchange Format)

**Extension :** `.dxf`
**Type MIME :** `application/dxf`, `image/vnd.dxf`
**Importation :** Analyse vectorielle directe
**Exportation :** Export d'objet (géométrie uniquement)

**Qu'est-ce que le DXF ?**

Le DXF est un format de dessin AutoCAD, largement utilisé pour l'échange de
données CAO.

**Versions prises en charge :**

- ✓ **R12/LT2** (recommandée - meilleure compatibilité)
- ✓ R13, R14
- ✓ R2000 et ultérieures (fonctionne généralement, mais R12 est plus sûre)

**Entités prises en charge :**

- ✓ Lignes (LINE)
- ✓ Polylignes (LWPOLYLINE, POLYLINE)
- ✓ Arcs (ARC)
- ✓ Cercles (CIRCLE)
- ✓ Splines (SPLINE) - converties en polylignes
- ✓ Ellipses (ELLIPSE)
- ✓ Calques

**Fonctionnalités non prises en charge/limitées :**

- ✗ Entités 3D (utilisez la projection 2D)
- ✗ Cotations et annotations (ignorées)
- ✗ Blocs/insertions (l'instanciation peut ne pas fonctionner correctement)
- ✗ Types de ligne complexes (simplifiés en ligne continue)
- ✗ Texte (ignoré, convertissez en contours au préalable)
- ✗ Hachures (peuvent être simplifiées ou ignorées)

**Notes d'exportation :**

Lors de l'exportation d'une pièce vers DXF, Rayforge exporte :

- Les lignes en tant qu'entités LWPOLYLINE
- Les arcs en tant qu'entités ARC
- Les courbes de Bézier en tant qu'entités SPLINE
- Les unités en millimètres (INSUNITS = 4)

---

### RFS (Croquis Rayforge)

**Extension :** `.rfs`
**Type MIME :** `application/x-rayforge-sketch`
**Importation :** Directe (pièces basées sur des croquis)
**Exportation :** Export d'objet (pièces basées sur des croquis)

**Qu'est-ce que le RFS ?**

Le RFS est le format natif de croquis paramétrique de Rayforge. Il préserve
tous les éléments géométriques et contraintes paramétriques, vous permettant
d'enregistrer et de partager des croquis entièrement modifiables.

**Fonctionnalités :**

- ✓ Tous les éléments géométriques (lignes, arcs, cercles, rectangles, etc.)
- ✓ Toutes les contraintes paramétriques
- ✓ Valeurs dimensionnelles et expressions
- ✓ Zones de remplissage

**Quand l'utiliser :**

- Enregistrer des conceptions paramétriques réutilisables
- Partager des croquis modifiables avec d'autres utilisateurs de Rayforge
- Archiver des travaux en cours

---

### PDF (Portable Document Format)

**Extension :** `.pdf`
**Type MIME :** `application/pdf`
**Importation :** Vecteurs directs (avec prise en charge des calques) ou
rendu et tracé
**Exportation :** Non pris en charge

**Qu'est-ce que l'importation PDF ?**

Les fichiers PDF peuvent contenir des chemins vectoriels réels, et Rayforge les
importe directement lorsqu'ils sont disponibles — vous offrant la même
géométrie propre que celle obtenue à partir d'un SVG. Si le PDF contient des
calques, chaque calque peut être importé comme une pièce distincte.

Pour les PDF sans contenu vectoriel exploitable (documents numérisés, photos),
Rayforge a recours au rendu et au tracé.

**Capacités :**

- ✓ **Importation vectorielle directe** pour les PDF vectoriels
- ✓ **Détection et sélection des calques** — choisissez les calques à importer
- ✓ Rendu et tracé de secours pour le contenu matriciel

**Limitations :**

- Première page uniquement — les PDF multipages importent la page 1
- Le texte peut nécessiter une conversion en contours dans l'application source

**Quand l'utiliser :**

- PDF reçus de designers contenant des illustrations vectorielles
- Tout PDF avec des calques bien organisés
- Lorsque SVG ou DXF n'est pas disponible depuis la source

---

## Formats matriciels

Tous les formats matriciels sont **importés par tracé** — convertis
automatiquement en chemins vectoriels.

### PNG (Portable Network Graphics)

**Extension :** `.png`
**Type MIME :** `image/png`
**Importation :** Tracer en vecteurs
**Exportation :** Non pris en charge

**Caractéristiques :**

- **Compression sans perte** - Aucune perte de qualité
- **Prise en charge de la transparence** - Canal alpha préservé
- **Idéal pour :** Logos, dessins au trait, captures d'écran, tout élément
  nécessitant de la transparence

**Qualité de tracé :** (Excellente pour les images à contraste élevé)

**Bonnes pratiques :**

- Utilisez PNG pour les logos et graphiques aux bords nets
- Assurez un contraste élevé entre le premier plan et l'arrière-plan
- L'arrière-plan transparent fonctionne mieux que le blanc

---

### JPEG (Joint Photographic Experts Group)

**Extension :** `.jpg`, `.jpeg`
**Type MIME :** `image/jpeg`
**Importation :** Tracer en vecteurs
**Exportation :** Non pris en charge

**Caractéristiques :**

- **Compression avec perte** - Certaine perte de qualité
- **Pas de transparence** - A toujours un arrière-plan
- **Idéal pour :** Photos, images à tons continus

**Qualité de tracé :** (Bonne pour les photos, mais complexe)

**Bonnes pratiques :**

- Utilisez des JPEG de haute qualité (faible compression)
- Augmentez le contraste avant l'importation
- Envisagez un prétraitement dans un éditeur d'images
- Mieux vaut convertir en PNG d'abord si possible

---

### BMP (Bitmap)

**Extension :** `.bmp`
**Type MIME :** `image/bmp`
**Importation :** Tracer en vecteurs
**Exportation :** Non pris en charge

**Caractéristiques :**

- **Non compressé** - Tailles de fichier importantes
- **Format simple** - Largement compatible
- **Idéal pour :** Graphiques simples, sorties de logiciels anciens

**Qualité de tracé :** (Bonne, mais pas meilleure que PNG)

**Bonnes pratiques :**

- Convertissez en PNG pour une taille de fichier réduite (aucune différence
  de qualité)
- À utiliser uniquement si le logiciel source ne peut pas exporter en PNG/SVG

---

## Pages associées

- [Importer des fichiers](importing) - Comment importer chaque format
- [Exporter](exporting) - Options d'exportation G-code
