# Soumettre des changements

Ce guide couvre le processus de contribution d'améliorations de code à Rayforge.

## Créer une branche de fonctionnalité

Créez une branche descriptive pour vos changements :

```bash
git checkout -b feature/votre-nom-de-fonctionnalite
# ou
git checkout -b fix/numero-ticket-description
```

## Effectuer vos changements

- Suivez le style et les conventions de code existants
- Écrivez des commits propres et ciblés avec des messages clairs
- Ajoutez des tests pour les nouvelles fonctionnalités
- Mettez à jour la documentation si nécessaire

## Tester vos changements

Exécutez la suite de tests complète pour vous assurer que rien n'est cassé :

```bash
# Exécuter tous les tests et linting
pixi run test
pixi run lint
```

## Synchroniser avec l'amont

Avant de créer une pull request, synchronisez avec le dépôt amont :

```bash
# Récupérer les derniers changements
git fetch upstream

# Rebaser votre branche sur le dernier main
git rebase upstream/main
```

## Soumettre une Pull Request

1. Poussez votre branche vers votre fork :

   ```bash
   git push origin feature/votre-nom-de-fonctionnalite
   ```

2. Créez une pull request sur GitHub et remplissez le modèle de pull request :
   - Un titre clair et une description du changement et du problème qu'il résout
   - Pourquoi vous avez choisi votre approche architecturale et quelles alternatives vous avez
     écartées
   - Les étapes exactes que vous avez suivies pour tester le changement localement
   - Référence aux tickets liés le cas échéant
   - Captures d'écran si le changement affecte l'interface
   - Toutes les cases de la liste de contrôle du modèle cochées

<!-- prettier-ignore-start -->
:::warning[Application du modèle]
Les pull requests qui ne remplissent pas le modèle sont fermées automatiquement par un bot.
Modifiez la description de la pull request pour répondre à chaque section et cocher chaque case ;
la pull request est ensuite rouverte automatiquement.
:::
<!-- prettier-ignore-end -->

## Processus de revue de code

- Toutes les pull requests nécessitent une revue avant fusion
- Répondez rapidement aux commentaires et effectuez les changements demandés
- Gardez la discussion ciblée et constructive

## Exigences de fusion

Les pull requests sont fusionnées lorsqu'elles :

- [ ] Passent tous les tests automatisés
- [ ] Suivent le style de codage du projet
- [ ] Incluent les tests appropriés pour les nouvelles fonctionnalités
- [ ] Ont des mises à jour de documentation si nécessaire
- [ ] Sont approuvées par au moins un mainteneur

## Code assisté par IA

Nous accueillons l'utilisation d'outils d'IA pour aider à écrire du code. Cependant, vous êtes
strictement responsable du code que vous soumettez. Vous DEVEZ le comprendre entièrement et être
capable d'expliquer les choix architecturaux.

Le code généré par IA introduit souvent des bugs subtils, des inadéquations architecturales et des
failles de sécurité. Les pull requests qui semblent être des sorties d'IA brutes et non relues
seront fermées pour respecter le temps des mainteneurs.

## Directives supplémentaires

### Messages de commit

Utilisez des messages de commit clairs et descriptifs :

- Commencez par une majuscule
- Gardez la première ligne sous 50 caractères
- Utilisez l'impératif ("Ajouter fonctionnalité" pas "Fonctionnalité ajoutée")
- Incluez plus de détails dans le corps si nécessaire

### Changements petits et ciblés

Gardez les pull requests focalisées sur une seule fonctionnalité ou correction. Les grands
changements devraient être divisés en morceaux plus petits et logiques.

<!-- prettier-ignore-start -->
:::tip[Discutez d'abord]
Pour les changements majeurs, ouvrez d'abord un
[ticket](https://github.com/barebaric/rayforge/issues) pour discuter de votre approche avant
d'investir beaucoup de temps.
:::
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
:::note[Besoin d'aide ?]
Si vous n'êtes sûr d'aucune partie du processus de contribution, n'hésitez
pas à demander de l'aide dans un ticket ou une discussion.
:::
<!-- prettier-ignore-end -->
