# Addons

![Paramètres des Addons](/screenshots/app-settings-addons.webp)

La page Addons affiche les addons d'extension installés et vous permet de les gérer.

## Aperçu

Les addons sont des extensions qui ajoutent des fonctionnalités à Rayforge. Ils peuvent fournir :

- De nouvelles opérations
- Des filtres d'import/export
- Des pilotes de machine
- Des améliorations de l'interface utilisateur
- Des bibliothèques de matériaux

## Services en ligne

Le réglage **Activer les services externes d'addons** contrôle l'accès au registre en ligne,
l'installation à distance et les services de licence Gumroad et Patreon. Lorsqu'il est désactivé,
les addons intégrés et ceux déjà installés localement restent disponibles, mais Rayforge n'effectue
aucune requête vers des services externes d'addons.

## Addons Installés

La liste des addons affiche tous les addons installés avec :

- **Nom** : Le nom d'affichage de l'addon
- **Version** : Version actuellement installée
- **Description** : Brève description de la fonctionnalité
- **Statut** : Activé ou désactivé

## Détails de l'extension

![Vue détaillée de l'extension](/screenshots/app-settings-addons-detail.webp)

Cliquez sur la ligne d'une extension — dans la liste des extensions installées ou dans la boîte de
dialogue du registre des extensions — pour ouvrir sa vue détaillée. La vue détaillée affiche :

- **Captures d'écran** : une galerie balayable, lorsque l'auteur de l'extension fournit des captures
- **Description** : une description plus longue de ce que fait l'extension, avec formatage Markdown
- **Auteur, version et licence**
- Un lien vers le dépôt de l'extension
- L'action **Installer**, **Mettre à jour** ou **Déverrouiller**, selon l'état de l'extension

Les captures d'écran et la description longue sont chargées depuis le registre en ligne. Lorsque les
services d'extensions externes sont désactivés ou que le registre est injoignable, la vue détaillée
affiche toujours les informations locales de l'extension.

## Gérer les Addons

### Installer des Addons

1. Cliquez sur le bouton **Installer**
2. Sélectionnez un fichier d'addon (`.zip` ou dépôt d'addon)
3. Confirmez l'installation (les addons communautaires exigent de confirmer un avertissement de
   confiance)
4. Attendez que l'installation se termine

### Mettre à Jour les Addons

Quand des mises à jour sont disponibles, un indicateur apparaît dans le dialogue du Registre des
Addons. Cliquez sur **Mettre à Jour** pour installer les mises à jour disponibles.

### Supprimer des Addons

1. Sélectionnez l'addon à supprimer
2. Cliquez sur le bouton **Supprimer**
3. Confirmez la suppression

<!-- prettier-ignore-start -->
:::warning
Supprimer un addon peut affecter les documents qui utilisent ses fonctionnalités.
:::
<!-- prettier-ignore-end -->

## Addons Communautaires

Les addons qui n'ont pas été créés par l'équipe Rayforge sont marqués comme addons communautaires
avec une icône dédiée dans les listes d'addons.

Lorsque vous installez un addon communautaire, un dialogue de confirmation explique que l'addon
n'est pas fourni par Rayforge lui-même et affiche un lien vers le dépôt de l'addon. Ouvrez le lien
dans votre navigateur pour examiner le code et son auteur avant de continuer.

<!-- prettier-ignore-start -->
:::warning
Les addons ont le même accès à votre ordinateur et à vos fichiers que Rayforge lui-même.
N'installez des addons communautaires que si vous faites confiance à leurs auteurs.
:::
<!-- prettier-ignore-end -->

## Sources d'Addons

Les addons peuvent être installés depuis :

- **Dépôts** : Dépôts Git contenant des addons
- **Registre officiel** : Addons organisés par l'équipe Rayforge

## Sujets Connexes

- [Machines](machines) - Configuration de la machine
- [Matériaux](materials) - Bibliothèques de matériaux
- [Recettes](recipes) - Recettes d'opérations
