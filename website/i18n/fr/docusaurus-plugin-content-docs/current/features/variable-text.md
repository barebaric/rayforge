---
description:
  "Numéros de série dans les modèles de texte : un compteur de projet qui avance après chaque
  travail, avec des numéros consécutifs pour les copies d'un réseau."
---

# Texte variable

Les zones de texte résolvent les modèles `{...}` (voir
[Paramètres et expressions](sketcher/expressions.md#template-expressions-in-text-boxes)). En plus
des paramètres de l'esquisse et des fonctions intégrées, le projet propose des variables qui
changent d'un travail à l'autre, pour graver des plaques, étiquettes et badges numérotés sans
modifier le texte à chaque fois.

## Numéros de série

Utilisez `{serial}` dans une zone de texte. C'est un nombre entier, donc les spécifications de
format contrôlent son apparence :

- `{serial}` — `7`
- `{serial:04d}` — `0007`, quatre chiffres avec des zéros en tête
- `No. {serial:03d}` — `No. 007`

Ouvrez **Outils → Texte variable…** pour configurer le compteur :

- **Valeur actuelle** — le numéro par lequel commence le prochain travail.
- **Valeur de départ** — là où **Réinitialiser** revient.
- **Avancer de** — l'incrément par copie et par travail.
- **Reboucler à la valeur de fin** / **Valeur de fin** — après la valeur de fin, le compteur reprend
  à la valeur de départ.
- **Avancer après chaque travail** — quand un travail envoyé à la machine est terminé, le compteur
  avance. Le cadrage, les essais à blanc avec le pointeur et les travaux annulés ne le modifient
  pas.
- **Avancer après l'export G-code** — avancer aussi lors de l'export du G-code, pour les travaux
  lancés depuis une carte mémoire.

Le groupe **Prochain travail** affiche un aperçu des numéros que le prochain travail utilisera.
**Précédent** et **Suivant** déplacent le compteur d'un travail, **Réinitialiser** revient à la
valeur de départ. Chaque modification, y compris l'avance automatique, fait partie de l'historique
d'annulation et est enregistrée dans le projet.

## Numéroter des copies

Chaque pièce a un **Décalage de variable** (dans le panneau des propriétés, sous **Texte
variable**). Une copie avec le décalage _n_ affiche le numéro situé _n_ copies après le numéro
actuel. Un [réseau en grille ou circulaire](arrays.md) donne automatiquement à ses copies des
décalages croissants : un réseau de 20 plaques grave ainsi 20 numéros consécutifs, et le travail
suivant continue avec le 21e.

Les réseaux créés dans une esquisse sont différents : leurs copies appartiennent à une seule pièce
et affichent toutes le même numéro. Créez plutôt le réseau sur le canevas pour numéroter les copies.

Pour des copies placées vous-même, sélectionnez-les toutes et choisissez **Organiser → Réseau →
Numéroter les copies** : les décalages sont attribués dans l'ordre de lecture, les lignes de haut en
bas et chaque ligne de gauche à droite.

Le compteur avance du nombre de copies du travail : le plus grand décalage de variable plus un.
