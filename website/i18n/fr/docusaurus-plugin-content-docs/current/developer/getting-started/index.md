# Obtenir le code

Ce guide couvre comment obtenir le code source de Rayforge pour le développement.

## Forker le dépôt

Forkez le [dépôt Rayforge](https://github.com/barebaric/rayforge) sur GitHub pour créer votre propre copie où vous pouvez effectuer des changements.

## Cloner votre fork

```bash
git clone https://github.com/VOTRE_NOM_UTILISATEUR/rayforge.git
cd rayforge
```

## Ajouter le dépôt amont

Ajoutez le dépôt original comme remote amont pour suivre les changements :

```bash
git remote add upstream https://github.com/barebaric/rayforge.git
```

## Vérifier le dépôt

Vérifiez que les remotes sont configurés correctement :

```bash
git remote -v
```

Vous devriez voir à la fois votre fork (origin) et le dépôt amont.

## Prochaines étapes

Après avoir obtenu le code, continuez avec [Configuration](setup) pour configurer votre environnement de développement.
