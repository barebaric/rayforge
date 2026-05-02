---
description: "Avant d'exécuter ou d'exporter un travail, Rayforge vérifie automatiquement les problèmes courants comme les dépassements de limites, les violations de zone de travail et les collisions avec les zones interdites."
---

# Vérifications de cohérence du travail

Avant d'exécuter ou d'exporter un travail, Rayforge effectue automatiquement
un ensemble de vérifications de cohérence et présente les résultats dans une
boîte de dialogue structurée. Cela vous permet de détecter les problèmes tôt,
avant qu'ils ne se traduisent par du matériel gaspillé.

![Boîte de dialogue de vérification](/screenshots/sanity-check.png)

## Vérifications effectuées

- **Dépassements des limites de la machine** : Géométrie qui s'étend au-delà de
  ce que votre machine peut atteindre physiquement, signalée par axe et par
  direction
- **Violations de la zone de travail** : Pièces situées en dehors des limites
  de la zone de travail configurée
- **Collisions avec les zones interdites** : Trajets d'outil passant par des
  zones interdites activées

Chaque vérification produit au plus un problème par violation unique, gardant
la boîte de dialogue lisible même pour les projets complexes. La boîte de
dialogue fait la distinction entre les erreurs et les avertissements, et vous
pouvez tout examiner avant de décider de continuer ou non.

---

## Pages associées

- [Zones interdites](../machine/nogo-zones) - Définir des zones restreintes
  sur la surface de travail
- [Vue 3D](../ui/3d-preview) - Visualisation 3D des trajets d'outil
