# Machines

![Paramètres des Machines](/screenshots/application-machines.png)

La page Machines dans les Paramètres de l'Application vous permet de gérer les profils de machine. Chaque profil contient toute la configuration pour une machine laser spécifique.

## Profils de Machine

Les profils de machine stockent la configuration complète pour une découpeuse ou graveuse laser, incluant :

- **Paramètres généraux** : Nom, vitesses, accélération
- **Paramètres matériels** : Dimensions de la zone de travail, configuration des axes
- **Paramètres laser** : Plage de puissance, fréquence PWM
- **Paramètres de l'appareil** : Port série, débit, type de firmware
- **Paramètres G-code** : Options de dialecte G-code personnalisé
- **Paramètres caméra** : Calibration et alignement de la caméra

## Gérer les Machines

### Ajouter une Nouvelle Machine

1. Cliquez sur le bouton **Ajouter une Nouvelle Machine**
2. Entrez un nom descriptif pour votre machine
3. Configurez les paramètres de la machine (voir
   [Configuration Machine](../machine/general) pour plus de détails)
4. Cliquez sur **Sauvegarder** pour créer le profil

### Basculer Entre les Machines

Utilisez le menu déroulant de sélection de machine dans la fenêtre principale pour basculer entre les machines configurées. Tous les paramètres, incluant la machine sélectionnée, sont mémorisés entre les sessions.

### Dupliquer une Machine

Pour créer un profil de machine similaire :

1. Sélectionnez la machine à dupliquer
2. Cliquez sur le bouton **Dupliquer**
3. Renommez la nouvelle machine et ajustez les paramètres selon vos besoins

### Supprimer une Machine

1. Sélectionnez la machine à supprimer
2. Cliquez sur le bouton **Supprimer**
3. Confirmez la suppression

:::warning
La suppression d'un profil de machine ne peut pas être annulée. Assurez-vous d'avoir
noté tous les paramètres importants avant de supprimer.
:::

## Sujets Connexes

- [Configuration Machine](../machine/general) - Configuration détaillée de la machine
- [Configuration Initiale](../getting-started/first-time-setup) - Guide de
  configuration initiale
