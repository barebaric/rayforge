# Dépannage et signalement de problèmes

Si vous rencontrez des problèmes avec Rayforge, en particulier pour vous connecter ou contrôler votre machine, nous sommes là pour vous aider. La meilleure façon d'obtenir du support est de fournir un rapport de débogage détaillé. Rayforge dispose d'un outil intégré qui facilite cette tâche.

## Comment créer un rapport de débogage

Suivez ces étapes simples pour générer et partager un rapport :

#### 1. Sauvegarder le rapport

Allez dans **Aide → Sauvegarder le journal de débogage** dans la barre de menu. Cela regroupera toutes les informations de diagnostic nécessaires dans un seul fichier `.zip`. Sauvegardez ce fichier à un endroit mémorable, comme votre Bureau.

#### 2. Créer un ticket GitHub

Allez sur notre [page des tickets GitHub](https://github.com/barebaric/rayforge/issues/new/choose) et créez un nouveau ticket. Veuillez fournir un titre clair et une description détaillée du problème :

- **Qu'avez-vous fait ?** (par exemple, "J'ai essayé de me connecter à mon laser après avoir démarré l'application.")
- **Que vous attendiez-vous à ce qui se passe ?** (par exemple, "Je m'attendais à ce qu'il se connecte avec succès.")
- **Que s'est-il réellement passé ?** (par exemple, "Il est resté déconnecté et le journal a montré des erreurs de timeout.")

#### 3. Joindre le rapport

**Faites glisser et déposez le fichier `.zip`** que vous avez sauvegardé dans la boîte de description du ticket GitHub. Cela le téléchargera et l'attachera à votre rapport.

## Que contient le rapport de débogage ?

Le fichier `.zip` généré contient des informations techniques qui nous aident à diagnostiquer le problème rapidement. Il inclut :

- **Paramètres machine et application :** Vos configurations machine sauvegardées et préférences d'application, ce qui nous aide à reproduire votre configuration.
- **Dialectes personnalisés :** Tous les dialectes G-code personnalisés que vous avez créés ou modifiés.
- **Configuration des addons :** La liste des addons activés/désactivés.
- **Journaux de communication :** Un enregistrement détaillé des données envoyées entre Rayforge et votre laser.
- **Informations système :** Votre système d'exploitation et les versions de Rayforge et des bibliothèques clés installées.
- **État de l'application :** D'autres informations internes qui peuvent aider à localiser la source d'une erreur.

> **Note de confidentialité :** Le rapport **n'inclut pas** vos fichiers de conception (SVG, DXF, etc.) ou les données personnelles du système d'exploitation. Il ne contient que les informations directement liées à l'application Rayforge et sa connexion à votre laser.

Merci de nous aider à améliorer Rayforge
