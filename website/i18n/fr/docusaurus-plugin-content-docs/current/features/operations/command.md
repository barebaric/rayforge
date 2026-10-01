---
description:
  "L'étape de commande injecte du code machine personnalisé, une ligne par commande, à une position
  exacte dans le flux de travail d'un calque. À utiliser pour le prépositionnement, la commande
  d'équipements et les commandes spécifiques à la machine."
---

# Commande

L'étape de commande injecte du code machine personnalisé dans le travail, exactement à la position
qu'elle occupe dans le flux de travail du calque. Utilisez-la pour envoyer des commandes que les
opérations de géométrie ne couvrent pas : positionnement de la tête, activation/désactivation de
l'assistance air ou d'autres équipements autour d'opérations spécifiques, ou envoi de codes
spécifiques à la machine.

![Paramètres de l'étape de commande](/screenshots/step-settings-command-general.webp)

## Aperçu

L'étape de commande :

- Contient un bloc de code machine multi-lignes, une commande par ligne
- Émet chaque ligne telle quelle à la position de l'étape lorsque le travail est encodé
- S'exécute une fois par calque, à sa position dans le flux de travail — elle n'agit pas sur les
  pièces
- Fonctionne sans aucune pièce dans le calque, de sorte qu'un calque ne contenant qu'une étape de
  commande peut quand même générer un travail
- Prend en charge les mêmes variables de chemin que les macros (voir ci-dessous)

Le texte est stocké non développé dans le projet, il voyage donc avec le fichier `.ryp` et documente
exactement ce qui est envoyé où.

## Quand utiliser l'étape de commande

Utilisez l'étape de commande pour :

- Prépositionner la tête (p. ex. remonter Z avant le début d'une découpe)
- Activer/désactiver l'assistance air, le liquide de refroidissement ou d'autres équipements entre
  les opérations
- Envoyer des codes spécifiques au contrôleur autour d'un travail
- Marquer une brève pause avec un dwell entre deux opérations

**N'utilisez pas l'étape de commande pour :**

- Répéter du G-code aux limites de calque ou de pièce — les
  [Macros & Hooks](../../machine/hooks-macros.md) se déclenchent automatiquement et fonctionnent
  aussi sur les machines sans G-code
- Exécuter des programmes sur l'ordinateur (non pris en charge dans cette phase)

## Ajouter une étape de commande

1. Ouvrez le flux de travail du calque dans le panneau de droite.
2. Cliquez sur le bouton **Ajouter une étape** et choisissez **Commande**.
3. Saisissez le code machine dans la zone de texte, une commande par ligne. Les lignes vides sont
   ignorées.

L'étape peut être placée avant, entre ou après d'autres étapes, et peut être utilisée plusieurs fois
dans un calque. Sa position dans le flux de travail est celle qu'occupent ses lignes dans le code
machine généré.

## Variables de chemin

Comme les macros, les lignes peuvent contenir des variables de chemin qui sont résolues lors de
l'encodage du travail. Les variables inconnues restent inchangées, et les variables `layer.*` sont
résolues même lorsque l'étape s'exécute au milieu d'un calque.

| Variable           | Exemple    | Description                                           |
| ------------------ | ---------- | ----------------------------------------------------- |
| `{machine.name}`   | `My Laser` | Nom de la machine active                              |
| `{layer.name}`     | `Layer 1`  | Nom du calque en cours de traitement                  |
| `{job.extents[0]}` | `210.0`    | Étendue du travail sur l'axe X (mm)                   |
| `{wcs_offset[0]}`  | `5.0`      | Décalage X du système de coordonnées de travail actif |

Par exemple, un commentaire comme `; découpe de {layer.name} sur {machine.name}` est encodé avec les
noms remplacés.

## Prise en charge des machines

L'étape de commande est disponible sur toutes les machines, mais les lignes ne sont émises que pour
les pilotes qui consomment du code machine. Lorsque le pilote de la machine active ne le fait pas
(p. ex. Ruida), l'étape affiche un avertissement et ses lignes ne font pas partie de la sortie
générée.
