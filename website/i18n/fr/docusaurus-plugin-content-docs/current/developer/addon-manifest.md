# Manifeste d'extension

Chaque extension a besoin d'un fichier `rayforge-addon.yaml` dans son répertoire racine. Ce manifeste indique à Rayforge des informations sur votre extension — son nom, ce qu'elle fournit et comment la charger.

## Structure de base

Voici un manifeste complet avec tous les champs courants :

```yaml
name: my_custom_addon
display_name: "My Custom Addon"
description: "Adds support for the XYZ laser cutter."
api_version: 9
url: https://github.com/username/my-custom-addon

author:
  name: Jane Doe
  email: jane@example.com

depends:
  - rayforge>=0.27.0

requires:
  - some-other-addon>=1.0.0

provides:
  backend: my_addon.backend
  frontend: my_addon.frontend
  assets:
    - path: assets/profiles.json
      type: profiles

license:
  name: MIT
```

## Champs requis

### `name`

Un identifiant unique pour votre extension. Il doit s'agir d'un nom de module Python valide — uniquement des lettres, des chiffres et des underscores, et il ne peut pas commencer par un chiffre.

```yaml
name: my_custom_addon
```

### `display_name`

Un nom lisible par l'humain affiché dans l'interface. Il peut contenir des espaces et des caractères spéciaux.

```yaml
display_name: "My Custom Addon"
```

### `description`

Une brève description de ce que fait votre extension. Elle apparaît dans le gestionnaire d'extensions.

```yaml
description: "Adds support for the XYZ laser cutter."
```

### `api_version`

La version de l'API ciblée par votre extension. Elle doit être au moins 1 (la version minimale supportée) et au maximum la version actuelle (9). Utiliser une version supérieure à celle supportée entraînera l'échec de la validation de votre extension.

```yaml
api_version: 9
```

Consultez la documentation des [Hooks](./addon-hooks.md#api-version-history) pour savoir ce qui a changé dans chaque version.

### `author`

Informations sur l'auteur de l'extension. Le champ `name` est requis ; `email` est optionnel mais recommandé pour permettre aux utilisateurs de vous contacter.

```yaml
author:
  name: Jane Doe
  email: jane@example.com
```

## Champs optionnels

### `url`

Une URL vers la page d'accueil ou le dépôt de votre extension.

```yaml
url: https://github.com/username/my-custom-addon
```

### `depends`

Contraintes de version pour Rayforge lui-même. Spécifiez la version minimale requise par votre extension.

```yaml
depends:
  - rayforge>=0.27.0
```

### `requires`

Dépendances envers d'autres extensions. Listez les noms des extensions avec leurs contraintes de version.

```yaml
requires:
  - some-other-addon>=1.0.0
```

### `version`

Le numéro de version de votre extension. Il est généralement déterminé automatiquement à partir des tags git, mais vous pouvez le spécifier explicitement. Utilisez le versionnement sémantique (ex: `1.0.0`).

```yaml
version: 1.0.0
```

## Points d'entrée

La section `provides` définit ce que votre extension apporte à Rayforge.

### Backend

Le module backend se charge à la fois dans le processus principal et dans les processus de travail. Utilisez-le pour les pilotes de machine, les types d'étapes, les producteurs d'ops et toute fonctionnalité principale.

```yaml
provides:
  backend: my_addon.backend
```

La valeur est un chemin de module Python avec points relatif au répertoire de votre extension.

### Frontend

Le module frontend ne se charge que dans le processus principal. Utilisez-le pour les composants UI, les widgets GTK et tout ce qui a besoin de la fenêtre principale.

```yaml
provides:
  frontend: my_addon.frontend
```

### Assets

Vous pouvez empaqueter des fichiers de ressources que Rayforge reconnaîtra. Chaque ressource a un chemin et un type :

```yaml
provides:
  assets:
    - path: assets/profiles.json
      type: profiles
    - path: assets/templates
      type: templates
```

Le `path` est relatif à la racine de votre extension et doit exister. Les types de ressources sont définis par Rayforge et peuvent inclure des choses comme des profils de machine, des bibliothèques de matériaux ou des modèles.

## Informations de licence

Le champ `license` décrit comment votre extension est licenciée. Pour les extensions gratuites, spécifiez simplement le nom de la licence en utilisant un identifiant SPDX :

```yaml
license:
  name: MIT
```

Les identifiants SPDX courants incluent `MIT`, `Apache-2.0`, `GPL-3.0` et `BSD-3-Clause`.

## Extensions payantes

Rayforge supporte les extensions payantes via la validation de licences Gumroad. Si vous souhaitez vendre votre extension, vous pouvez la configurer pour exiger une licence valide avant de fonctionner.

### Configuration payante de base

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
```

Quand `required` est vrai, Rayforge vérifiera la présence d'une licence valide avant de charger votre extension. L'URL `purchase_url` est montrée aux utilisateurs qui n'ont pas de licence.

### ID produit Gumroad

Ajoutez votre ID produit Gumroad pour activer la validation de licence :

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_id: "abc123def456"
```

Pour plusieurs ID produits (ex: différents niveaux de tarification) :

```yaml
license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/my-addon
  product_ids:
    - "abc123def456"
    - "xyz789ghi012"
```

### Exemple complet d'extension payante

Voici un manifeste complet pour une extension payante :

```yaml
name: premium_laser_pack
display_name: "Premium Laser Pack"
description: "Advanced features for professional laser cutting."
api_version: 9
url: https://example.com/premium-laser-pack

author:
  name: Your Name
  email: you@example.com

depends:
  - rayforge>=0.27.0

provides:
  backend: premium_pack.backend
  frontend: premium_pack.frontend

license:
  name: BSL-1.1
  required: true
  purchase_url: https://gum.co/premium-laser-pack
  product_ids:
    - "standard_tier_id"
    - "pro_tier_id"
```

### Vérifier le statut de la licence dans le code

Dans le code de votre extension, vous pouvez vérifier si une licence est valide :

```python
@hookimpl
def rayforge_init(context):
    if context.license_validator:
        # Check if user has a valid license for your product
        is_valid = context.license_validator.is_product_valid("your_product_id")
        if not is_valid:
            # Optionally show a message or limit functionality
            logger.warning("License not found - some features disabled")
```

## Règles de validation

Rayforge valide votre manifeste lors du chargement de l'extension. Voici les règles :

Le `name` doit être un identifiant Python valide (lettres, chiffres, underscores, pas de chiffre au début). L'`api_version` doit être un entier entre 1 et la version actuelle. Le champ `author.name` ne peut pas être vide ou contenir du texte générique comme "your-github-username". Les points d'entrée doivent être des chemins de modules valides et les modules doivent exister. Les chemins de ressources doivent être relatifs (pas de `..` ou de `/` au début) et les fichiers doivent exister.

Si la validation échoue, Rayforge journalise une erreur et ignore votre extension. Vérifiez la sortie console pendant le développement pour détecter ces problèmes.
