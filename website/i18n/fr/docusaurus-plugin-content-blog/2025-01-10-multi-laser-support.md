---
slug: multi-laser-support
title: Prise en charge multi-laser - Choisissez différents lasers pour chaque opération
authors: rayforge_team
tags: [multi-laser, opérations, flux-de-travail]
---

![Superposition de caméra](/images/camera-overlay.png)

L'une des fonctionnalités les plus puissantes de Rayforge est la capacité
d'assigner différents lasers à différentes opérations au sein d'une même
tâche. Cela ouvre des possibilités passionnantes pour les configurations
multi-outils et les flux de travail spécialisés.

<!-- truncate -->

## Qu'est-ce que la prise en charge multi-laser ?

Si votre machine est équipée de plusieurs modules laser — par exemple,
un laser à diode pour la gravure et un laser CO2 pour la découpe, ou
différents lasers à diode de puissance variée optimisés pour différents
matériaux — Rayforge vous permet de tirer pleinement parti de cette
configuration.

Avec la prise en charge multi-laser, vous pouvez :

- **Assigner différents lasers à différentes opérations** dans votre tâche
- **Basculer entre les modules laser** automatiquement pendant l'exécution
  de la tâche
- **Optimiser pour le matériau et la tâche** en utilisant l'outil adapté
  à chaque opération

## Cas d'utilisation

### Gravure et découpe hybrides

Imaginez que vous travailliez sur un projet de panneau en bois :

1. **Opération 1** : Utiliser un laser à diode de faible puissance pour
   graver du texte fin et des graphiques détaillés
2. **Opération 2** : Passer à un laser CO2 de plus grande puissance pour
   découper la forme du panneau

Avec Rayforge, vous assignez simplement chaque opération au laser
approprié dans votre profil de machine, et le logiciel gère le reste.

### Optimisation spécifique au matériau

Différents types de laser excellent avec différents matériaux :

- **Lasers à diode** : Excellents pour la gravure sur bois, le cuir et
  certains plastiques
- **Lasers CO2** : Parfaits pour découper l'acrylique, le bois et
  d'autres matériaux organiques
- **Lasers à fibre** : Idéaux pour le marquage des métaux

Si vous avez plusieurs types de lasers sur un système portique, la
prise en charge multi-laser de Rayforge vous permet d'utiliser l'outil
optimal pour chaque partie de votre projet.

## Comment le configurer

### 1. Configurer plusieurs lasers dans votre profil de machine

Allez dans **Configuration de la machine → Lasers multiples** et
définissez chaque module laser dans votre système. Vous pouvez spécifier :

- Le type de laser et la plage de puissance
- Les positions de décalage (si les lasers sont montés à différents
  emplacements)
- La compatibilité des matériaux

Consultez notre [Guide de configuration laser](/docs/machine/laser)
pour des instructions détaillées.

### 2. Assigner les lasers aux opérations

Lors de la création d'opérations dans votre projet :

1. Sélectionnez l'opération (Contour, Trame, etc.)
2. Dans les paramètres de l'opération, choisissez le laser à utiliser
   dans le menu déroulant
3. Configurez les paramètres de l'opération spécifiques à ce laser

### 3. Prévisualiser et exécuter

Utilisez l'aperçu 3D pour vérifier vos trajets d'outil, puis envoyez la
tâche à votre machine. Rayforge générera automatiquement les commandes
G-code appropriées pour basculer entre les lasers selon les besoins.

## Détails techniques

Sous le capot, Rayforge utilise des commandes G-code pour basculer entre
les modules laser. L'implémentation exacte dépend de votre firmware et
de votre configuration matérielle, mais les approches courantes incluent :

- **M3/M4 avec décalages d'outil** : Basculer entre les lasers en
  utilisant des commandes de changement d'outil
- **Contrôle GPIO** : Utiliser des broches GPIO prises en charge par le
  firmware pour activer/désactiver différents modules laser
- **Macros personnalisées** : Définir des macros pré- et post-opération
  pour la commutation laser

## Pour commencer

La prise en charge multi-laser est disponible dans Rayforge 0.15 et
versions ultérieures. Pour démarrer :

1. Mettez à jour vers la dernière version
2. Configurez votre profil de machine avec plusieurs lasers
3. Testez sur un projet expérimental !

Consultez la [documentation des profils de machine](/docs/machine/general)
pour plus de détails.

---

*Vous avez une configuration multi-laser ? Nous serions ravis d'entendre
parler de votre expérience ! Partagez vos projets et vos commentaires sur
[GitHub](https://github.com/barebaric/rayforge).*
