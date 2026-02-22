# Exporter depuis Rayforge

Rayforge prend en charge plusieurs options d'exportation pour différents usages :

- **G-code** - Sortie de contrôle machine pour exécuter des travaux
- **Export d'objet** - Exporter des pièces individuelles vers des formats vectoriels
- **Export de document** - Exporter toutes les pièces dans un seul fichier

---

## Exporter des objets

Vous pouvez exporter n'importe quelle pièce vers des formats vectoriels pour une utilisation dans des
logiciels de conception, applications CAO, ou pour l'archivage.

### Comment exporter

1. **Sélectionnez une pièce** sur le canevas
2. **Choisissez Objet → Exporter l'objet...** (ou clic droit → Exporter l'objet...)
3. **Sélectionnez le format** et l'emplacement de sauvegarde

### Formats disponibles

| Format  | Extension | Description                                                                                                    |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| **RFS** | `.rfs`    | Format de croquis paramétrique natif de Rayforge. Préserve toutes les contraintes et peut être réimporté pour édition. |
| **SVG** | `.svg`    | Scalable Vector Graphics. Largement compatible avec les logiciels de conception comme Inkscape, Illustrator et les navigateurs web. |
| **DXF** | `.dxf`    | Drawing Exchange Format. Compatible avec la plupart des applications CAO comme AutoCAD, FreeCAD et LibreCAD.   |

### Notes d'exportation

- **SVG et DXF** exportent la géométrie résolue (pas les contraintes paramétriques)
- Les exportations utilisent les **unités millimétriques**
- La géométrie est mise à l'échelle aux dimensions réelles (espace monde)
- Les sous-chemins multiples (formes déconnectées) sont préservés comme éléments séparés

### Cas d'utilisation

**Partage de conceptions :**

- Exporter en SVG pour partager avec des utilisateurs Inkscape
- Exporter en DXF pour les utilisateurs de logiciels CAO

**Édition ultérieure :**

- Exporter en SVG/DXF, modifier dans un logiciel externe, réimporter

**Archivage :**

- Utiliser RFS pour les conceptions basées sur croquis pour préserver l'éditabilité
- Utiliser SVG/DXF pour le stockage à long terme ou les non-utilisateurs Rayforge

---

## Exporter des documents

Vous pouvez exporter toutes les pièces d'un document dans un seul fichier vectoriel. C'est
utile pour partager des projets complets ou créer des sauvegardes dans des formats standard.

### Comment exporter

1. **Choisissez Fichier → Exporter le document...**
2. **Sélectionnez le format** (SVG ou DXF)
3. **Choisissez l'emplacement de sauvegarde**

### Formats disponibles

| Format  | Extension | Description                                                                                                    |
| ------- | --------- | -------------------------------------------------------------------------------------------------------------- |
| **SVG** | `.svg`    | Scalable Vector Graphics. Largement compatible avec les logiciels de conception comme Inkscape, Illustrator et les navigateurs web. |
| **DXF** | `.dxf`    | Drawing Exchange Format. Compatible avec la plupart des applications CAO comme AutoCAD, FreeCAD et LibreCAD.   |

### Notes d'exportation

- Toutes les pièces de tous les calques sont combinées dans un seul fichier
- Les positions des pièces sont préservées
- Les pièces vides sont ignorées
- La boîte englobante englobe toute la géométrie

### Cas d'utilisation

- **Partage de projet** : Exporter le projet entier pour collaboration
- **Sauvegarde** : Créer une archive visuelle de votre travail
- **Édition ultérieure** : Importer toute la conception dans Inkscape ou un logiciel CAO

---

## Exporter du G-code

Le G-code généré contient tout exactement comme il serait envoyé à la machine.
Le format exact, les commandes, la précision numérique, etc. dépendent des paramètres de
la machine actuellement sélectionnée et de son dialecte G-code.

---

### Méthodes d'exportation

### Méthode 1 : Menu Fichier

**Fichier Exporter G-code** (Ctrl+E)

- Ouvre la boîte de dialogue de sauvegarde de fichier
- Choisissez l'emplacement et le nom de fichier
- Le G-code est généré et enregistré

### Méthode 2 : Ligne de commande

```bash
# Exporter depuis la ligne de commande (si pris en charge)
rayforge --export output.gcode input.svg
```

---

### Sortie G-code

Le G-code généré contient tout exactement comme il serait envoyé à la machine.
Le format exact, les commandes, la précision numérique, etc. dépendent des paramètres de
la machine actuellement sélectionnée et de son dialecte G-code.

---

## Pages connexes

- [Importer des fichiers](importing) - Importer des conceptions dans Rayforge
- [Formats pris en charge](formats) - Détails des formats de fichiers
- [Dialectes G-code](../reference/gcode-dialects) - Différences entre dialectes
- [Hooks et Macros](../machine/hooks-macros) - Personnaliser la sortie
- [Mode simulation](../features/simulation-mode) - Prévisualiser avant exportation
