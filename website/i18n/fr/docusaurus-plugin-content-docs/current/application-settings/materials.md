# Matériaux

![Paramètres des Matériaux](/screenshots/app-settings-materials.png)

Les bibliothèques de matériaux dans Rayforge vous permettent d'organiser et gérer des collections de matériaux pour vos projets de découpe et gravure laser. Ce guide explique la différence entre les bibliothèques principales et utilisateur, et comment créer vos propres bibliothèques et y ajouter des matériaux.

:::note
 Assigner un matériau à un brut affecte à la fois son apparence visuelle
 dans le canevas 2D et 3D et les [recettes](recipes.md) qui lui sont
 appliquées : les recettes spécifiques à un matériau correspondent au
 matériau assigné. Dans les futures versions, les matériaux seront
 utilisés pour dériver davantage de paramètres fonctionnels.
 :::


## Créer une Nouvelle Bibliothèque

Pour créer votre propre bibliothèque de matériaux :

1. Ouvrez le menu **Paramètres** et sélectionnez **Matériaux**
2. Cliquez sur le bouton **Ajouter une Nouvelle Bibliothèque** pour créer une nouvelle bibliothèque
3. Entrez un nom descriptif pour votre bibliothèque (ex : "Matériaux Mon Atelier")
4. Cliquez sur **Créer** pour finaliser

Votre nouvelle bibliothèque sera créée dans le répertoire de données utilisateur et sera disponible immédiatement.


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

Une image de texture (WebP ou PNG) qui est répétée en mosaïque sur la
surface du matériau. Lorsqu'elle est définie, le matériau est rendu avec
la texture au lieu d'une couleur unie. Les textures peuvent être
optimisées en WebP avec le script
`scripts/optimize_material_textures.py` pour garder les fichiers de
matériau légers.

#### Échelle de la texture

La taille (en mm) qu'une tuile de texture couvre sur le matériau. Des
valeurs plus petites répètent la texture plus souvent sur la même
surface.

#### Couleur

La couleur de base du matériau. Lorsqu'une texture est définie et que le
matériau est teintable, la couleur teinte la texture. La couleur est
uniquement utilisée pour l'apparence visuelle sur la surface de travail -
elle n'affecte pas le parcours laser de quelque manière que ce soit.

#### Teintable

Lorsqu'elle est activée, la texture du matériau peut être teintée avec la
couleur ci-dessus. Cela permet à un seul matériau texturé (ex :
"Acrylique") de couvrir plusieurs variantes de couleur : la couleur est
appliquée par brut dans la boîte de dialogue [Propriétés du
brut](../features/stock-handling.md).

#### Rugosité

Une valeur de 0 à 1 décrivant à quel point la surface apparaît rugueuse
ou polie dans la vue 3D. Les valeurs plus basses semblent brillantes, les
valeurs plus hautes semblent mates.

#### Métallique

Une valeur de 0 à 1 décrivant si la surface réfléchit la lumière comme un
métal dans la vue 3D. Réglez sur 1 pour les matériaux métalliques, 0 pour
les matériaux non métalliques.


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

:::warning
La suppression d'un matériau est permanente et ne peut pas être annulée.
:::
