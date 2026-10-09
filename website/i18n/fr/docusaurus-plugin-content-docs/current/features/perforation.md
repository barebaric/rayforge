---
description:
  "Découpez des lignes en pointillés dans Rayforge avec le post-traitement Perforation : réglez une
  longueur de coupe et une longueur de saut pour plis, lignes prédécoupées et trous de couture."
---

# Perforation

Le post-traitement **Perforation** transforme une découpe continue en découpe en pointillés. Le
laser tire sur la **Longueur de coupe**, puis se déplace laser éteint sur la **Longueur de saut**,
et répète cela le long de chaque contour de la pièce.

## Quand l'utiliser

- Lignes de pliage dans le carton et le carton ondulé
- Lignes prédécoupées (billets, coupons, emballages)
- Trous de couture dans le cuir
- Pointillés décoratifs

## Réglages

La perforation est disponible dans les réglages de post-traitement des opérations **Contour**. Elle
est désactivée par défaut.

- **Longueur de coupe** : la distance sur laquelle le laser tire avant chaque interruption.
- **Longueur de saut** : la distance parcourue laser éteint entre deux coupes.

Le motif est mesuré le long de chaque contour de la pièce et recommence au début de chaque contour,
toujours par une coupe complète. Un contour plus court qu'une coupe plus un saut est découpé en
entier.

## Conseils

- Pour une ligne de pliage qui ne doit pas traverser, commencez avec une longueur de saut proche de
  la longueur de coupe et baissez la puissance.
- Des longueurs très courtes (quelques dixièmes de millimètre) font s'allumer et s'éteindre le laser
  rapidement, ce qui réduit la puissance effective.
- La perforation fonctionne avec les [Ponts de maintien](holding-tabs),
  l'[Entrée/Sortie](lead-in-out) et les [Passes multiples](multi-pass).

## Pages associées

- [Découpe de contour](operations/contour) - L'opération de découpe qui utilise la perforation
- [Ponts de maintien](holding-tabs) - Interruptions placées à la main pour garder les pièces
  attachées
