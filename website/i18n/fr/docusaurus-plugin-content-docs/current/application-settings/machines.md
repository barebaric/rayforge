---
description:
  "Gérez les machines dans Rayforge — ajoutez, configurez, exportez, importez et basculez entre
  différentes découpeuses et graveuses laser pour vos projets."
---

# Machines

![Paramètres des Machines](/screenshots/app-settings-machines.webp)

La page Machines dans les Paramètres de l'Application affiche une liste de toutes les machines
configurées. Chaque entrée affiche le nom de la machine et dispose de boutons pour la modifier ou la
supprimer. La machine actuellement active est marquée d'une icône de coche.

## Ajouter une Machine

1. Cliquez sur le bouton **Add Machine** en bas de la liste
2. Sélectionnez un profil d'appareil dans la liste comme modèle — chaque profil préconfigure les
   paramètres de la machine et le dialecte G-code

![Ajouter une Machine](/screenshots/app-settings-machines-add.webp)

3. Le [dialogue de paramètres de machine](../machine/general.md) s'ouvre pour vous permettre
   d'ajuster la configuration

Alternativement :

- Cliquez sur **Device Not Listed** pour lancer l'
  [Assistant de Configuration](../getting-started/first-time-setup.md), qui vous guide pas à pas
  dans la configuration d'une machine
- Cliquez sur **Import from File…** pour ajouter une machine depuis un profil exporté précédemment
  ou depuis un profil d'appareil LightBurn (.lbdev). Les profils LightBurn incluent le calibrage de
  caméra et les paramètres laser qui sont appliqués à la nouvelle machine.

## Modifier une Machine

Cliquez sur l'icône de modification à côté d'une machine pour ouvrir le
[dialogue de paramètres de machine](../machine/general.md).

## Notes de la machine

La catégorie **Notes**, juste après **General**, regroupe les conseils du profil de l'appareil et
vos propres notes pour cette machine. Les notes de l'appareil sont en lecture seule ; **My Notes**
peuvent être modifiées en Markdown. La syntaxe prise en charge comprend les titres, les paragraphes,
le gras, l'italique, les listes, les liens, le code en ligne, les blocs de code et les sections
dépliables avec `:::details`.

Lorsqu'un profil fournit des instructions de configuration, elles apparaissent également sur la
dernière page **Review & Name** de l'assistant et restent disponibles dans **Machine Settings →
Notes** après la création de la machine. Les notes personnelles sont enregistrées par machine et ne
sont pas remplacées lors de l'examen des mises à jour du profil.

## Changer la Machine Active

Utilisez le menu déroulant des machines dans l'en-tête de la fenêtre principale pour basculer entre
les machines configurées. La sélection est mémorisée entre les sessions.

## Supprimer une Machine

1. Cliquez sur l'icône de suppression à côté de la machine
2. Confirmez la suppression

<!-- prettier-ignore-start -->
:::warning
La suppression d'une machine ne peut pas être annulée. Exportez le profil au préalable si
vous souhaitez conserver la configuration.
:::
<!-- prettier-ignore-end -->
