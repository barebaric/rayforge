---
description:
  "Gravez des QR codes, Code 128 et EAN-13 dans Rayforge. Les codes-barres sont des actifs
  réutilisables dont le contenu peut être un modèle, pour que chaque pièce porte son propre code."
---

# Codes-barres

L'extension Codes-barres ajoute des QR codes, Code 128 et EAN-13 comme actifs réutilisables. Un
code-barres est dessiné en contours vectoriels avec un remplissage : une opération de Gravure le
grave proprement et une opération de Contour peut toujours le suivre.

## Ajouter un code-barres

![Le groupe Code-barres avec un symbole Code 128](/screenshots/addon-barcodes.webp)

Choisissez **Objet - Ajouter un code-barres**. Un QR code apparaît au centre de la zone de travail
et est sélectionné. Modifiez-le dans le groupe **Code-barres** du panneau de propriétés :

- **Nom** — le nom de l'actif dans le navigateur d'actifs.
- **Contenu** — le texte à encoder. Appuyez sur `Entrée` ou sur le bouton d'application.
- **Type** — QR Code, Code 128 ou EAN-13.
- **Correction d'erreur** (QR) — la part du code qui peut être endommagée : Faible, Moyenne,
  Quartile ou Élevée. Les niveaux élevés agrandissent le code.
- **Taille du module** — la largeur de la barre la plus fine ou d'un carré QR. La taille du module
  et la hauteur des barres suivent votre unité de longueur.
- **Hauteur des barres** (Code 128, EAN-13) — la hauteur des barres.
- **Afficher le texte** (Code 128, EAN-13) — imprime le contenu sous les barres. EAN-13 utilise sa
  disposition habituelle, les barres de garde descendant entre les chiffres.
- **Inverser** — grave les parties claires, pour un matériau sombre. La **zone de silence** règle la
  bordure sombre en modules et n'est modifiable qu'avec Inverser activé.

## Le navigateur d'actifs

Chaque code-barres est aussi un actif. Ouvrez le navigateur d'actifs dans le panneau inférieur :

- **Ajouter** (`+`) crée un actif sans le placer. Utilisez **Créer une nouvelle pièce**, faites-le
  glisser sur le canevas ou double-cliquez sur la carte.
- Un double-clic sur une pièce code-barres la sélectionne et affiche ses propriétés.
- Dupliquer, copier, couper et supprimer fonctionnent comme pour les autres actifs.

Modifier l'actif met à jour toutes les copies placées. Utilisez des modèles lorsque les copies
doivent différer.

Laissez une marge autour d'un code non inversé : 4 modules pour les QR codes, 10 pour Code 128 et 9
pour EAN-13.

## Types

- **QR Code** accepte tout texte, y compris les liens.
- **Code 128** accepte l'ASCII imprimable. Les suites de chiffres sont regroupées par paires.
- **EAN-13** prend 12 chiffres et ajoute le chiffre de contrôle. Avec 13 chiffres, il est vérifié.
  Le champ devient rouge lorsque le contenu ne convient pas ; survolez-le pour voir pourquoi.

## Modèles

Le contenu peut contenir des expressions `{...}` comme une zone de texte, par exemple `{2 * 21}`.
Avec le [compteur de travaux](../features/job-counter.md), `SN-{counter:05d}` donne à chaque pièce
son propre numéro, et les copies de tableau continuent à compter.
