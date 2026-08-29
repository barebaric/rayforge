---
description: "Apprenez à créer des lignes, courbes de Bézier, arcs, ellipses, rectangles et autres géométries 2D dans l'esquisseur Rayforge."
---

# Création de géométrie 2D

L'esquisseur prend en charge la création des éléments géométriques de base
suivants :

- **Tracés (lignes et courbes de Bézier)** : Dessinez des lignes droites et des
  courbes de Bézier lisses avec l'outil de tracé unifié. Cliquez pour placer
  des points, glissez pour créer des poignées de Bézier.
- **Arcs** : Dessinez des arcs en spécifiant un point central, un point de
  départ et un point d'arrivée
- **Ellipses** : Créez des ellipses (et des cercles) en deux clics : le
  premier définit le point central, le second le point du bord. Vous pouvez
  également appuyer au centre, glisser et relâcher au bord - les deux gestes
  fonctionnent de façon interchangeable. Maintenez `Ctrl` pour contraindre
  à un cercle parfait et `Shift` pour utiliser le point de départ comme
  centre de l'ellipse.
- **Rectangles** : Dessinez des rectangles en spécifiant deux coins opposés,
  ou appuyez au premier coin, glissez et relâchez au coin opposé. Chaque
  rectangle crée automatiquement un point central (contraint au centre
  géométrique) pour que vous puissiez dimensionner ou vous y accrocher.
  Maintenez `Shift` en dessinant pour placer le rectangle symétriquement
  autour du point de départ, et `Ctrl` pour le contraindre à un carré.
- **Rectangles arrondis** : Dessinez des rectangles avec des coins arrondis
  en utilisant les mêmes gestes et modificateurs que l'outil rectangle :
  deux clics ou clic-glisser, avec `Shift` pour centrer sur le point de
  départ et `Ctrl` pour contraindre à un carré. Le rayon des coins peut être
  défini en saisissant des dimensions (`0-9`, champs W, H et R).
- **Zones de texte** : Ajoutez des éléments textuels à votre esquisse. Le
  contenu du texte prend en charge les expressions de modèle paramétriques
  (voir [Modèles de texte](../text.md)).
- **Remplissages** : Remplissez des régions fermées pour créer des zones
  solides

Ces éléments constituent la base de vos conceptions 2D et peuvent être combinés
pour créer des formes complexes. Les remplissages sont particulièrement utiles
pour créer des régions solides qui seront gravées ou découpées en une seule
pièce.

## Deux clics ou glisser

Les outils de création de formes (ellipse, rectangle, rectangle arrondi)
acceptent deux gestes de façon interchangeable : cliquez au premier point,
déplacez, puis cliquez au second, ou appuyez au premier point, glissez et
relâchez au second. Un clic rapide sans mouvement arme simplement l'outil et
attend le second point ; les clics accidentels ne laissent donc jamais de
géométrie dégénérée. Pendant qu'un aperçu est actif, la barre d'état affiche
les touches modificatrices disponibles, et `Esc` annule l'aperçu.

## Travailler avec les courbes de Bézier

L'outil de tracé prend en charge les courbes de Bézier pour créer des formes
lisses et organiques :

### Dessiner des courbes de Bézier

1. Sélectionnez l'outil de tracé dans le menu radial ou utilisez le raccourci
   clavier
2. Cliquez pour placer des points ; chaque clic crée un nouveau point
3. Glissez après avoir cliqué pour créer des poignées de Bézier pour des
   courbes lisses
4. Continuez à ajouter des points pour construire votre tracé
5. Appuyez sur Échap ou double-cliquez pour terminer le tracé

### Modifier les courbes de Bézier

- **Déplacer des points** : Cliquez et glissez n'importe quel point pour le
  repositionner
- **Ajuster les poignées** : Glissez les extrémités des poignées pour modifier
  la forme de la courbe
- **Se connecter aux points existants** : Lors de la modification d'un tracé,
  vous pouvez vous accrocher aux points existants de votre esquisse
- **Rendre lisse/symétrique** : Les points connectés par une contrainte de
  coïncidence peuvent être rendus lisses (tangente continue) ou symétriques
  (poignées en miroir)

### Convertir des courbes en lignes

Utilisez l'**outil d'aplatissement** pour convertir les courbes de Bézier en
lignes droites. Ceci est utile lorsque vous avez besoin d'une géométrie propre
et simple. Sélectionnez les segments de Bézier que vous souhaitez convertir et
appliquez l'action d'aplatissement.
