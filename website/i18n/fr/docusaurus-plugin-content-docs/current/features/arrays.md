---
description: "Créez des copies en grille, en rotation ponctuelle ou circulaire. Chaque mode propose un aperçu en direct et un placement interactif."
---

# Tableaux

La fonction Tableau permet de créer plusieurs copies de pièces de travail
sélectionnées selon trois modes de disposition différents. Chaque mode
ouvre un dialogue non modal, vous permettant de continuer à interagir avec
le canevas pendant que vous ajustez les paramètres — l'aperçu se met à jour
en temps réel.

Pour ouvrir un dialogue de tableau, sélectionnez une ou plusieurs pièces de
travail sur le canevas, puis choisissez le mode de tableau dans la barre
d'outils ou le menu contextuel.

:::tip
Tous les modes de tableau sont non modaux. Vous pouvez glisser des pièces
de travail sur le canevas pendant que le dialogue est ouvert, et l'aperçu
se mettra à jour en direct pour refléter les nouvelles positions.
:::

---

## Grille

Le mode Grille organise les copies dans une matrice rectangulaire de lignes
et de colonnes, avec un espacement horizontal et vertical configurable.

![Tableau Grille](/screenshots/main-array-grid.png)

### Paramètres

| Paramètre | Description |
|-----------|-------------|
| **Lignes** | Nombre de lignes (1–360) |
| **Colonnes** | Nombre de colonnes (1–360) |
| **Mode d'espacement** | Choisissez entre *Écart* (espace entre les copies) ou *Pas* (distance de bord à bord de chaque copie) |
| **Espacement des colonnes** | Espacement horizontal entre les colonnes |
| **Espacement des lignes** | Espacement vertical entre les lignes |

---

## Rotation Ponctuelle

Le mode Rotation Ponctuelle crée des copies en les tournant sur place autour
du propre centre de la sélection. Ceci est utile pour créer des motifs
circulaires où chaque copie reste à son emplacement d'origine mais est
tournée d'une fraction de l'angle total.

![Tableau Rotation Ponctuelle](/screenshots/main-array-point-rotation.png)

### Paramètres

| Paramètre | Description |
|-----------|-------------|
| **Nombre** | Nombre de copies (1–360) |
| **Angle total (deg)** | Étendue angulaire totale de toutes les copies (−360° à 360°) |

:::info
Puisque la rotation est autour du propre centre de la sélection, glisser
la pièce de travail sur le canevas déplace toutes les copies ensemble
tandis que le dialogue reste ouvert.
:::

---

## Circulaire

Le mode Circulaire place les copies le long d'un arc circulaire autour d'un
point central. Un marqueur en croix sur le canevas indique le centre, et
vous pouvez le glisser vers une nouvelle position pendant que le dialogue
est ouvert.

![Tableau Circulaire](/screenshots/main-array-circular.png)

### Paramètres

| Paramètre | Description |
|-----------|-------------|
| **Nombre** | Nombre de copies (1–360) |
| **Angle total (deg)** | Étendue angulaire de l'arc (−360° à 360°) |
| **Centre X** | Coordonnée X du centre du cercle |
| **Centre Y** | Coordonnée Y du centre du cercle |
| **Rayon** | Rayon de la trajectoire circulaire |
| **Tourner les copies** | Lorsque activé, chaque copie est tournée pour suivre la tangente de l'arc |

:::tip Glisser le centre
La croix sur le canevas représente le centre du cercle. Glissez-la pour
repositionner le tableau de manière interactive — les champs Centre X et
Centre Y dans le dialogue seront mis à jour automatiquement.
:::

:::tip Glisser les pièces de travail
Vous pouvez également glisser la pièce de travail d'origine sur le canevas.
Le rayon sera mis à jour automatiquement pour maintenir les copies à leur
distance actuelle du centre.
:::
