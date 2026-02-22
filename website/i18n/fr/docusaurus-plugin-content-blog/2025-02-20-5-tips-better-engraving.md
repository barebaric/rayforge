---
slug: 5-tips-better-engraving
title: 5 conseils pour de meilleurs résultats de gravure laser avec Rayforge
authors: rayforge_team
tags: [gravure, optimisation, qualité, flux-de-travail]
---

![Aperçu 3D](/screenshots/main-3d.png)

Obtenir des résultats de gravure laser de qualité professionnelle
nécessite plus que du bon matériel — vos paramètres logiciels et votre
flux de travail comptent aussi. Voici cinq conseils pour vous aider à
tirer le meilleur parti de Rayforge.

<!-- truncate -->

## 1. Utilisez l'overscan pour une gravure tramée plus fluide

Lors de la gravure tramée, un problème courant est l'apparition de
lignes visibles ou d'incohérences sur les bords où le laser change de
direction. Cela se produit parce que la tête laser doit décélérer et
accélérer, ce qui peut affecter la qualité de gravure.

**Solution** : Activez l'**Overscan** dans les paramètres de votre
opération tramée.

L'overscan étend le trajet du laser au-delà de la zone de gravure
réelle, permettant à la tête d'atteindre sa pleine vitesse avant
d'entrer dans la zone de travail et de la maintenir tout au long.
Cela donne une gravure beaucoup plus fluide et plus homogène.

Pour activer l'overscan :

1. Sélectionnez votre opération tramée
2. Ouvrez les paramètres de l'opération
3. Activez « Overscan » et définissez la distance (généralement 3-5mm
   fonctionne bien)

En savoir plus dans notre [guide Overscan](/docs/features/overscan).

## 2. Optimisez le temps de déplacement avec l'ordonnancement des trajets

Pour les opérations de contour avec de nombreux trajets séparés,
l'ordre dans lequel le laser visite chaque forme peut avoir un impact
significatif sur le temps total de la tâche.

**Solution** : Utilisez l'**optimisation du temps de déplacement**
intégrée de Rayforge.

Rayforge peut réorganiser automatiquement les trajets pour minimiser
le temps de déplacement sans découpe. Cela est particulièrement utile
pour les tâches avec de nombreux petits objets ou du texte avec
plusieurs lettres.

L'optimisation des trajets est généralement activée par défaut, mais
vous pouvez la vérifier et l'ajuster dans les paramètres de l'opération
Contour.

## 3. Ajoutez des ponts de maintien pour éviter le déplacement des pièces

Rien n'est plus frustrant que de voir une tâche de découpe presque
terminée gâchée parce que la pièce a bougé ou est tombée à travers le
plateau de la machine au dernier moment.

**Solution** : Utilisez des **Ponts de maintien** pour garder les pièces
en place jusqu'à ce que la tâche soit terminée.

Les ponts de maintien sont de petites sections non découpées qui
maintiennent votre pièce connectée au matériau environnant. Une fois
la tâche terminée, vous pouvez facilement retirer la pièce et nettoyer
les ponts avec un couteau ou du papier de verre.

Rayforge prend en charge le placement manuel et automatique des ponts :

- **Manuel** : Cliquez exactement où vous voulez les ponts sur le canevas
- **Automatique** : Spécifiez le nombre de ponts et laissez Rayforge
  les répartir uniformément

Consultez la [documentation des Ponts de maintien](/docs/features/holding-tabs)
pour un guide complet.

## 4. Prévisualisez votre tâche en 3D avant l'exécution

L'une des fonctionnalités les plus précieuses de Rayforge est l'aperçu
3D du G-code. Il est tentant de sauter cette étape et d'envoyer la
tâche directement à la machine, mais prendre un moment pour prévisualiser
peut vous faire gagner du temps et des matériaux.

**Ce qu'il faut rechercher dans l'aperçu** :

- Vérifiez que toutes les opérations s'exécutent dans le bon ordre
- Recherchez tout trajet d'outil ou chevauchement inattendu
- Confirmez que les opérations multi-passes ont le nombre correct de passes
- Assurez-vous que les limites de la tâche s'inscrivent dans votre matériau

Pour ouvrir l'aperçu 3D, cliquez sur le bouton **Aperçu 3D** dans la
barre d'outils principale après avoir généré votre G-code.

En savoir plus sur l'aperçu 3D dans notre
[documentation de l'interface](/docs/ui/3d-preview).

## 5. Utilisez des hooks G-code personnalisés pour des flux de travail cohérents

Si vous vous retrouvez à exécuter les mêmes commandes G-code avant ou
après chaque tâche — comme le retour à l'origine, l'activation d'une
assistance air, ou l'exécution d'une routine de mise au point — vous
pouvez automatiser cela avec des **Macros et Hooks G-code**.

**Cas d'utilisation courants** :

- **Hook pré-tâche** : Retour à l'origine de la machine, activation de
  l'assistance air, exécution d'une routine de mise au point automatique
- **Hook post-tâche** : Désactivation de l'assistance air, retour à la
  position d'origine, lecture d'un son de fin
- **Macros spécifiques à une couche** : Changement de la hauteur de mise
  au point entre les opérations, changement de modules laser

Les hooks prennent en charge la substitution de variables, vous pouvez
donc référencer des propriétés de tâche comme l'épaisseur du matériau,
le type d'opération, et plus encore.

Exemple de hook pré-tâche :

```gcode
G28 ; Retour à l'origine de tous les axes
M8 ; Activer l'assistance air
G0 Z{focus_height} ; Déplacer à la hauteur de mise au point
```

Consultez notre [guide des Macros et Hooks G-code](/docs/machine/hooks-macros)
pour des exemples détaillés et une référence des variables.

---

## Conseil bonus : Testez d'abord sur du matériel de récupération

Bien que cela ne soit pas spécifique à Rayforge, cela vaut la peine d'être
répété : testez toujours les nouveaux paramètres, opérations ou matériaux
sur des chutes d'abord. Utilisez les profils de matériaux et les préréglages
d'opérations de Rayforge pour sauvegarder vos paramètres testés pour une
utilisation future.

---

*Vous avez vos propres astuces et conseils Rayforge ? Partagez-les avec
la communauté sur [GitHub Discussions](https://github.com/barebaric/rayforge/discussions) !*
