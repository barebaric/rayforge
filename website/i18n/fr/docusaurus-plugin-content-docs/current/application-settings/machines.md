---
description: "Gérez les machines dans Rayforge — ajoutez, configurez, exportez, importez et basculez entre différentes découpeuses et graveuses laser pour vos projets."
---

# Machines

![Paramètres des Machines](/screenshots/application-machines.png)

La page Machines dans les Paramètres de l'Application affiche une liste de
toutes les machines configurées. Chaque entrée affiche le nom de la machine
et dispose de boutons pour la modifier ou la supprimer. La machine
actuellement active est marquée d'une icône de coche.

## Ajouter une Machine

1. Cliquez sur le bouton **Add Machine** en bas de la liste
2. Sélectionnez un profil d'appareil dans la liste comme modèle — chaque
   profil préconfigure les paramètres de la machine et le dialecte G-code
3. Le [dialogue de paramètres de machine](../machine/general) s'ouvre pour
   vous permettre d'ajuster la configuration

Alternativement, cliquez sur **Import from File...** dans le sélecteur de
profil pour ajouter une machine à partir d'un profil précédemment exporté.

## Modifier une Machine

Cliquez sur l'icône de modification à côté d'une machine pour ouvrir le
[dialogue de paramètres de machine](../machine/general).

## Changer la Machine Active

Utilisez le menu déroulant des machines dans l'en-tête de la fenêtre
principale pour basculer entre les machines configurées. La sélection est
mémorisée entre les sessions.

## Supprimer une Machine

1. Cliquez sur l'icône de suppression à côté de la machine
2. Confirmez la suppression

:::warning
La suppression d'une machine ne peut pas être annulée. Exportez le profil
au préalable si vous souhaitez conserver la configuration.
:::
