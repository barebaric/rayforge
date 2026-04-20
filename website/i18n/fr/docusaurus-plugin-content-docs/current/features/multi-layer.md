---
description: "Organisez les travaux laser en calques avec différents réglages. Gérez l'ordre de coupe, les opérations et les matériaux avec le système multi-calques de Rayforge."
---

# Flux de travail multi-calques

![Panneau des calques](/screenshots/bottom-panel-layers.png)

Le système multi-calques de Rayforge vous permet d'organiser les travaux
en étapes de traitement séparées. Chaque calque est un conteneur pour
les pièces et possède son propre flux de travail — une séquence
d'étapes, chacune avec des réglages laser indépendants.

:::tip Quand vous n'avez pas besoin de plusieurs calques
Dans de nombreux cas, un seul calque suffit. Chaque étape au sein d'un
calque dispose de ses propres réglages de laser, puissance, vitesse et
autres paramètres, ce qui vous permet de graver et contourner dans le
même calque. Des calques séparés ne sont nécessaires que lorsque vous
souhaitez contourner différentes parties d'une image avec des réglages
différents, ou lorsque vous avez besoin de configurations WCS ou
rotatives différentes.
:::

## Créer et gérer les calques

### Ajouter un calque

Cliquez sur le bouton **+** dans le panneau des calques. Les nouveaux
documents commencent avec trois calques vides.

### Réorganiser les calques

Glissez-déposez les calques dans le panneau pour modifier l'ordre
d'exécution. Les calques sont traités de gauche à droite. Vous pouvez
utiliser le **glisser avec le clic du milieu** pour faire défiler la
liste des calques.

### Réorganiser les pièces

Les pièces au sein d'un calque peuvent être réorganisées par
glisser-déposer pour contrôler leur ordre Z.

### Supprimer un calque

Sélectionnez le calque et cliquez sur le bouton de suppression. Toutes
les pièces du calque sont supprimées. Vous pouvez annuler la suppression
si nécessaire.

## Propriétés du calque

Chaque calque possède les réglages suivants, accessibles via l'icône
d'engrenage dans la colonne du calque :

- **Nom** — affiché dans l'en-tête du calque
- **Couleur** — utilisée pour le rendu des opérations du calque sur le
  canevas
- **Visibilité** — l'icône en forme d'œil bascule l'affichage du calque
  sur le canevas et dans les aperçus. Les calques masqués sont
  toujours inclus dans le G-code généré.
- **Système de coordonnées (WCS)** — assigne un système de coordonnées
  de travail à ce calque. Lorsqu'il est défini sur un WCS spécifique
  (par ex. G54, G55), la machine bascule vers ce système de
  coordonnées au début du calque. Sélectionnez **Par défaut** pour
  utiliser le WCS global à la place.
- **Mode rotatif** — active le mode d'accessoire rotatif pour ce calque,
  vous permettant de mélanger le travail sur lit plat et cylindrique
  dans le même projet. Configurez le module rotatif et le diamètre de
  l'objet dans les réglages du calque.

## Flux de travail des calques

Chaque calque possède un **flux de travail** — une séquence d'étapes
affichée comme un pipeline d'icônes dans la colonne du calque. Chaque
étape définit une seule opération (par ex. contour, gravure raster)
avec ses propres réglages de laser, puissance, vitesse et autres
paramètres.

Cliquez sur une étape pour la configurer. Utilisez le bouton **+** du
pipeline pour ajouter des étapes à un calque. Les étapes peuvent être
réorganisées par glisser-déposer.

## Import de fichiers vectoriels

Lors de l'importation de fichiers vectoriels (SVG, DXF, PDF), le
dialogue d'importation propose trois façons de gérer les calques du
fichier source :

- **Mapper vers les calques existants** — importe chaque calque source
  dans le calque de document correspondant par position
- **Nouveaux calques** — crée un nouveau calque de document pour chaque
  calque source
- **Aplatir** — importe tout dans le calque actif

Lors de l'utilisation de **Mapper vers les calques existants** ou
**Nouveaux calques**, le dialogue affiche une liste des calques du
fichier source avec des interrupteurs pour sélectionner lesquels
importer.

## Assigner des pièces aux calques

**Glisser-déposer :** Sélectionnez les pièce(s) sur le canevas ou dans
le panneau de document et glissez-les vers le calque cible.

**Couper-coller :** Coupez une pièce du calque actuel (Ctrl+X),
sélectionnez le calque cible et collez (Ctrl+V).

## Ordre d'exécution

Pendant un travail, les calques sont traités de gauche à droite. Au
sein de chaque calque, toutes les pièces sont traitées avant de passer
au calque suivant. Le flux de travail standard consiste à graver en
premier et couper en dernier, afin que les pièces restent en place
pendant la gravure.

## Pages associées

- [Opérations](./operations/contour) - Types d'opérations pour les
  flux de travail par calque
- [Mode simulation](./simulation-mode) - Aperçu de l'exécution
  multi-calques
- [Macros et Hooks](../machine/hooks-macros) - Hooks au niveau du
  calque pour l'automatisation
- [Aperçu 3D](../ui/3d-preview) - Visualiser la pile de calques
