# Suivi d'utilisation

Rayforge inclut un suivi d'utilisation anonyme facultatif pour nous aider à comprendre comment l'application est utilisée et prioriser le développement futur. Cette page explique ce que nous suivons, comment cela fonctionne et votre vie privée.

## Entièrement facultatif

Le suivi d'utilisation est **entièrement facultatif**. Lors du premier démarrage de Rayforge, il vous sera demandé si vous souhaitez participer :

- **Oui** : Les données d'utilisation anonymes seront envoyées à notre serveur d'analyse
- **Non** : Aucune donnée n'est jamais collectée ou transmise

Vous pouvez modifier ce choix à tout moment dans les paramètres généraux.

## Ce que nous suivons

Lorsqu'il est activé, nous collectons uniquement des données anonymes de pages vues, similaires aux analyses de sites web. Voici ce que nous pouvons voir :

| Données                        | Exemple                   |
| ------------------------------ | ------------------------- |
| Résolution d'écran             | 1920x1080                 |
| Paramètre de langue            | fr-FR                     |
| Pages/dialogues consultés      | /machine-settings/general |
| Temps passé sur la page        | 6m 3s                     |

## Ce que nous voyons

Voici un exemple de ce à quoi ressemble le tableau de bord d'analyse :

| Chemin                      | Visiteurs | Visites | Vues | Taux de rebond | Durée de visite |
| --------------------------- | --------- | ------- | ---- | -------------- | --------------- |
| /                           | 1         | 1       | 5    | 0%             | 27m 35s         |
| /machine-settings/general   | 1         | 1       | 5    | 0%             | 27m 27s         |
| /view/3d                    | 1         | 1       | 2    | 0%             | 25m 14s         |
| /camera-alignment-dialog    | 1         | 1       | 2    | 0%             | 6m 3s           |
| /machine-settings/camera    | 1         | 1       | 2    | 0%             | 6m 16s          |
| /settings/general           | 1         | 1       | 2    | 0%             | 16m 36s         |
| /step-settings/rasterizer   | 1         | 1       | 2    | 0%             | 11s             |

## Ce que nous ne suivons PAS

Nous nous engageons à protéger votre vie privée :

- **Aucune information personnelle** – Pas de noms, e-mails ou comptes
- **Aucun contenu de fichier** – Vos conceptions et projets restent privés
- **Aucun identifiant de machine** – Pas de numéros de série ou d'ID uniques
- **Aucune adresse IP stockée** – Nous utilisons Umami Analytics qui ne stocke pas les IPs
- **Aucun suivi intersites** – Les données sont isolées uniquement à Rayforge

## Pourquoi nous suivons

Les données d'utilisation nous aident à :

- **Identifier les fonctionnalités populaires** – Savoir ce qui fonctionne bien
- **Trouver les points de friction** – Voir où les utilisateurs passent du temps ou sont bloqués
- **Prioriser le développement** – Se concentrer sur les fonctionnalités que les gens utilisent vraiment
- **Comprendre la diversité** – Savoir quelles langues et tailles d'écran prendre en charge

## Comment cela fonctionne

Rayforge utilise [Umami](https://umami.is/), une plateforme d'analyse open source axée sur la confidentialité. Le suivi :

- Envoie de petites requêtes HTTP en arrière-plan
- N'affecte pas les performances de l'application
- Fonctionne hors ligne (les requêtes échouées sont ignorées silencieusement)
- Utilise un User-Agent générique pour empêcher l'empreinte numérique

## Désactiver le suivi

Vous pouvez désactiver le suivi à tout moment :

1. Ouvrez **Paramètres** → **Général**
2. Désactivez **Envoyer des statistiques d'utilisation anonymes**

Lorsqu'il est désactivé, absolument aucune donnée n'est envoyée.

## Pages connexes

- **[Paramètres de l'application](../ui/settings)** – Configurer les préférences de suivi
