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

2. Créez une pull request sur GitHub avec :
   - Un titre clair décrivant le changement
   - Une description détaillée de ce que vous avez changé et pourquoi
   - Référence aux tickets liés le cas échéant
   - Captures d'écran si le changement affecte l'interface

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

## Directives supplémentaires

### Messages de commit

Utilisez des messages de commit clairs et descriptifs :

- Commencez par une majuscule
- Gardez la première ligne sous 50 caractères
- Utilisez l'impératif ("Ajouter fonctionnalité" pas "Fonctionnalité ajoutée")
- Incluez plus de détails dans le corps si nécessaire

### Changements petits et ciblés

Gardez les pull requests focalisées sur une seule fonctionnalité ou correction. Les grands changements devraient être divisés en morceaux plus petits et logiques.

:::tip Discutez d'abord
Pour les changements majeurs, ouvrez d'abord un [ticket](https://github.com/barebaric/rayforge/issues) pour discuter de votre approche avant d'investir beaucoup de temps.
:::


:::note Besoin d'aide ?
Si vous n'êtes sûr d'aucune partie du processus de contribution, n'hésitez pas à demander de l'aide dans un ticket ou une discussion.
:::
