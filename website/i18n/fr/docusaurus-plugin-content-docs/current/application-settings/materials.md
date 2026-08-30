# Matériaux

![Paramètres des Matériaux](/screenshots/app-settings-materials.webp)

Les bibliothèques de matériaux dans Rayforge vous permettent d'organiser et gérer des collections de
matériaux pour vos projets de découpe et gravure laser. Ce guide explique la différence entre les
bibliothèques principales et utilisateur, et comment créer vos propres bibliothèques et y ajouter
des matériaux.

:::note Assigner un matériau à un brut affecte à la fois son apparence visuelle dans le canevas 2D
et 3D et les [recettes](recipes.md) qui lui sont appliquées : les recettes spécifiques à un matériau
correspondent au matériau assigné. Dans les futures versions, les matériaux seront utilisés pour
dériver davantage de paramètres fonctionnels. :::

## Créer une Nouvelle Bibliothèque

Pour créer votre propre bibliothèque de matériaux :

1. Ouvrez le menu **Paramètres** et sélectionnez **Matériaux**
2. Cliquez sur le bouton **Ajouter une Nouvelle Bibliothèque** pour créer une nouvelle bibliothèque
3. Entrez un nom descriptif pour votre bibliothèque (ex : "Matériaux Mon Atelier")
4. Cliquez sur **Créer** pour finaliser

Votre nouvelle bibliothèque sera créée dans le répertoire de données utilisateur et sera disponible
immédiatement.

## Ajouter des Matériaux aux Bibliothèques

### Créer un Nouveau Matériau

1. Sélectionnez la bibliothèque où vous voulez ajouter le matériau
2. Cliquez sur le bouton **Ajouter un Nouveau Matériau** dans la liste des matériaux
3. Remplissez les propriétés du matériau :
   - **Nom** : Nom lisible par l'homme
   - **Catégorie** : Catégorie de groupement (ex : "Bois", "Acrylique")
   - **Apparence** : Propriétés visuelles (voir ci-dessous)
4. Cliquez sur **Sauvegarder** pour ajouter le matériau à la bibliothèque

### Propriétés des Matériaux Expliquées

#### Nom

- Nom lisible par l'homme affiché dans l'interface
- Peut contenir des espaces et caractères spéciaux

#### Catégorie

- Utilisée pour organiser les matériaux dans la bibliothèque
- Catégories courantes : Bois, Acrylique, Métal, Papier, Cuir
- Vous pouvez créer des catégories personnalisées selon vos besoins

#### Texture

Une image de texture (WebP ou PNG) qui est répétée en mosaïque sur la surface du matériau.
Lorsqu'elle est définie, le matériau est rendu avec la texture au lieu d'une couleur unie. Les
textures peuvent être optimisées en WebP avec le script `scripts/optimize_material_textures.py` pour
garder les fichiers de matériau légers.

#### Échelle de la texture

La taille (en mm) qu'une tuile de texture couvre sur le matériau. Des valeurs plus petites répètent
la texture plus souvent sur la même surface.

#### Couleur

Une couleur de teinte facultative. Lorsqu'elle est définie, la texture du matériau est teintée avec
cette couleur ; sinon, la texture est affichée telle quelle. Cela permet à un seul matériau texturé
(ex : "Acrylique") de couvrir plusieurs variantes de couleur : la couleur est appliquée par brut
dans la boîte de dialogue [Propriétés du brut](../features/stock-handling.md). La couleur est
uniquement utilisée pour l'apparence visuelle sur la surface de travail - elle n'affecte pas le
parcours laser de quelque manière que ce soit.

#### Rugosité

Une valeur de 0 à 1 décrivant à quel point la surface apparaît rugueuse ou polie dans la vue 3D. Les
valeurs plus basses semblent brillantes, les valeurs plus hautes semblent mates.

#### Métallique

Une valeur de 0 à 1 décrivant si la surface réfléchit la lumière comme un métal dans la vue 3D.
Réglez sur 1 pour les matériaux métalliques, 0 pour les matériaux non métalliques.

#### Absorption {#absorption}

:::note Nouveau en 1.11 Les données d'absorption pilotent le
[modèle de brûlure physique](../ui/3d-preview.md#physical-burn-model) dans l'aperçu 3D. :::

Les coefficients d'absorption par longueur d'onde (0–1) décrivent quelle partie de l'énergie du
laser le matériau absorbe à une longueur d'onde donnée. L'aperçu 3D les utilise, avec la longueur
d'onde, la puissance optique et la taille du spot de ta tête laser, pour calculer la fluence (J/cm²)
délivrée et rendre un effet de carbonisation physiquement motivé sur le brut.

Ajoute un bloc `absorption` sous `appearance` dans le YAML du matériau :

```yaml
appearance:
  absorption:
    blue: 0.7 # ~445 nm lasers à diode
    ir: 0.25 # ~1064 nm lasers fibrés / IR
    co2: 0.9 # ~10600 nm lasers CO2
  # ...autres propriétés d'apparence
```

| Bande  | Longueur d'onde représentative | Lasers typiques      |
| ------ | ------------------------------ | -------------------- |
| `blue` | 445 nm                         | Lasers à diode bleue |
| `ir`   | 1064 nm                        | Lasers fibrés        |
| `co2`  | 10600 nm                       | Lasers à tube CO2    |

Lorsqu'une bande est manquante, une valeur par défaut conservatrice est utilisée. La bibliothèque de
matériaux fournie contient des valeurs d'absorption recherchées pour tous les matériaux inclus ; le
modèle de brûlure n'est pas encore entièrement calibré, les contributions de données de test du
monde réel sont les bienvenues.

## Gérer les Matériaux Existants

### Éditer les Matériaux

1. Sélectionnez le matériau que vous voulez éditer
2. Cliquez sur le bouton **Éditer**
3. Modifiez les propriétés souhaitées
4. Cliquez sur **Sauvegarder** pour appliquer les changements

### Supprimer des Matériaux

1. Sélectionnez le matériau que vous voulez supprimer
2. Cliquez sur le bouton **Supprimer**
3. Confirmez la suppression dans la boîte de dialogue

:::warning La suppression d'un matériau est permanente et ne peut pas être annulée. :::
