# Fournisseur IA

![Paramètres du Fournisseur IA](/screenshots/application-ai.png)

Configurez les fournisseurs IA que les modules peuvent utiliser pour
ajouter des fonctionnalités intelligentes à Rayforge.

## Fonctionnement

Les modules peuvent utiliser les fournisseurs IA configurés sans avoir
besoin de leurs propres clés API. Cela centralise votre configuration IA
et vous permet de contrôler les fournisseurs disponibles pour les modules.

## Ajouter un Fournisseur

1. Cliquez sur **Ajouter un Fournisseur** pour créer une nouvelle
   configuration
2. Entrez un **Nom** pour identifier ce fournisseur
3. Définissez l'**URL de Base** sur le point de terminaison API de votre
   service IA
4. Entrez votre **Clé API** pour l'authentification
5. Spécifiez un **Modèle par Défaut** à utiliser avec ce fournisseur
6. Cliquez sur **Tester** pour vérifier que votre configuration fonctionne

## Types de Fournisseurs

### Compatible OpenAI

Ce type de fournisseur fonctionne avec tout service utilisant le format
API OpenAI. Cela inclut divers fournisseurs cloud et solutions auto-hébergées.

L'URL de base par défaut est définie sur l'API d'OpenAI, mais vous pouvez
la modifier pour pointer vers n'importe quel service compatible.

## Gérer les Fournisseurs

- **Activer/Désactiver**: Activez ou désactivez un fournisseur sans le
  supprimer
- **Définir par Défaut**: Cliquez sur l'icône de coche pour définir un
  fournisseur comme valeur par défaut
- **Supprimer**: Supprimez un fournisseur dont vous n'avez plus besoin

:::warning
Vos clés API sont stockées localement sur votre ordinateur et ne sont
jamais partagées avec des tiers.
:::

## Sujets Connexes

- [Addons](addons) - Installer et gérer les modules
- [Machines](machines) - Configuration des machines
- [Matériaux](materials) - Bibliothèques de matériaux
