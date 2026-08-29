---
description: "Utilisez l'esquisseur paramétrique 2D intégré de Rayforge pour créer des conceptions laser prêtes avec des lignes, cercles, courbes de Bézier et des contraintes."
---

# Esquisseur paramétrique 2D

L'Esquisseur paramétrique 2D est une fonctionnalité puissante de Rayforge qui
vous permet de créer et modifier des conceptions 2D précises basées sur des
contraintes directement dans l'application. Cette fonctionnalité vous permet de
concevoir des pièces personnalisées à partir de zéro sans avoir besoin d'un
logiciel de CAO externe.

## Aperçu

L'esquisseur fournit un ensemble complet d'outils pour créer des formes
géométriques et appliquer des contraintes paramétriques afin de définir des
relations précises entre les éléments. Cette approche garantit que vos
conceptions conservent leur géométrie prévue même lorsque les dimensions sont
modifiées.

## Création et modification d'esquisses

### Créer une nouvelle esquisse

1. Ouvrez le panneau inférieur et cliquez sur le bouton **Nouvelle esquisse**,
   ou faites un clic droit sur le canevas et sélectionnez **Nouvelle esquisse**
   dans le menu contextuel.
2. Un nouvel espace de travail vide s'ouvrira avec l'interface de l'éditeur
   d'esquisse
3. Commencez à créer de la géométrie avec les outils de dessin du menu radial
   ou les raccourcis clavier
4. Appliquez des contraintes pour définir les relations entre les éléments
5. Cliquez sur « Terminer l'esquisse » pour enregistrer votre travail et
   revenir à l'espace de travail principal

### Modifier une esquisse existante

1. Double-cliquez sur une pièce basée sur une esquisse dans l'espace de travail
   principal
2. Alternativement, sélectionnez une esquisse et choisissez « Modifier
   l'esquisse » dans le menu contextuel
3. Effectuez vos modifications avec les mêmes outils et contraintes
4. Cliquez sur « Terminer l'esquisse » pour enregistrer les modifications ou
   sur « Annuler l'esquisse » pour les annuler

## Conseils de flux de travail

1. **Commencez par une géométrie approximative** : Créez d'abord des formes de
   base, puis affinez avec des contraintes
2. **Utilisez les contraintes tôt** : Appliquez des contraintes au fur et à
   mesure pour maintenir l'intention de conception
3. **Vérifiez l'état des contraintes** : Le système indique quand les esquisses
   sont entièrement contraintes
4. **Surveillez les conflits** : Les contraintes en conflit sont mises en
   évidence en rouge et affichées dans le panneau des contraintes pour une
   identification facile
5. **Utilisez la symétrie** : Les contraintes de symétrie peuvent accélérer
   considérablement les conceptions complexes
6. **Utilisez la grille** : Activez la grille pour un alignement précis et
   utilisez Ctrl pour l'accrochage à la grille
7. **Itérez et affine** : N'hésitez pas à modifier les contraintes pour obtenir
   le résultat souhaité

## Fonctionnalités d'édition

- **Prise en charge complète d'annuler/rétablir** : L'état complet de
  l'esquisse est enregistré à chaque opération
- **Curseur dynamique** : Le curseur change pour refléter l'outil de dessin
  actif
- **Visualisation des contraintes** : Les contraintes appliquées sont clairement
  indiquées dans l'interface
- **Mises à jour en temps réel** : Les modifications des contraintes mettent à
  jour immédiatement la géométrie
- **Édition par double-clic** : Double-cliquer sur des contraintes
  dimensionnelles (Distance, Rayon, Diamètre, Angle, Rapport d'aspect) ouvre une
  boîte de dialogue pour modifier leurs valeurs
- **Expressions paramétriques** : Les contraintes dimensionnelles prennent en
  charge les expressions, permettant de calculer des valeurs à partir d'autres
  paramètres (par ex., `width/2` pour un rayon égal à la moitié de la largeur)
