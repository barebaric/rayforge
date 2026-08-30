# Recettes et Paramètres

![Paramètres des Recettes](/screenshots/app-settings-recipes.webp)

Rayforge fournit un système de recettes puissant qui vous permet de créer, gérer et appliquer des
paramètres cohérents à travers vos projets de découpe laser. Ce guide couvre le parcours utilisateur
complet, de la création de recettes dans les paramètres généraux à leur application aux opérations
et la gestion des paramètres au niveau de l'étape.

## Aperçu

Le système de recettes se compose de trois composants principaux :

1. **Gestion des Recettes** : Créer et gérer des préréglages de paramètres réutilisables
2. **Gestion du Matériau de Stock** : Définir les propriétés et l'épaisseur du matériau
3. **Paramètres d'Étape** : Appliquer et affiner les paramètres pour des opérations individuelles

## Gestion des Recettes

### Créer des Recettes

Les recettes sont des préréglages nommés qui contiennent tous les paramètres nécessaires pour des
opérations spécifiques. Vous pouvez créer des recettes via l'interface des paramètres principaux :

#### 1. Accéder au Gestionnaire de Recettes

Menu : Édition → Paramètres, puis sélectionnez Recettes

#### 2. Créer une Nouvelle Recette

Cliquez sur "Ajouter une Nouvelle Recette" pour ouvrir la boîte de dialogue de l'éditeur de
recettes.

**Onglet Général** - Définir le nom et la description de la recette :

![Éditeur de Recettes - Onglet Général](/screenshots/recipe-editor-general.webp)

Remplissez les informations de base :

- **Nom** : Nom descriptif (ex : "Coupe Contreplaqué 3mm")
- **Description** : Description détaillée optionnelle

#### 3. Définir les Critères d'Applicabilité

**Onglet Applicabilité** - Définir quand cette recette devrait être suggérée :

![Éditeur de Recettes - Onglet Applicabilité](/screenshots/recipe-editor-applicability.webp)

Tous les critères sont optionnels - laissez n'importe quel champ à sa valeur "Tout" pour tout faire
correspondre :

- **Machine** : Choisissez une machine spécifique ou laissez "Toute Machine"
- **Type de Tâche** : Sélectionnez la catégorie d'opération à laquelle cette recette s'applique
  (Coupe, Gravure, etc.), ou laissez "Tout" pour l'appliquer à tous les types de tâches
- **Type d'Étape** : Restreignez la recette à un type d'opération spécifique (ex : "Contour" ou
  "Raster"). La liste est filtrée selon les types d'étape qui prennent en charge le type de tâche
  sélectionné. Laissez "Tout Type" pour correspondre à chaque type d'étape dans la tâche
- **Matériau** : Sélectionnez un type de matériau ou laissez ouvert pour tout matériau
- **Épaisseur Min/Max** : Définissez les valeurs d'épaisseur minimum et maximum du matériau de stock

#### 4. Configurer les Paramètres

**Onglet Paramètres** - Ajuster la puissance, vitesse et autres paramètres. Lorsque la recette cible
un **type d'étape** spécifique, l'éditeur affiche deux pages de paramètres : une page "Laser" avec
les paramètres de processus partagés (puissance, assistance d'air, etc.) et une page "Paramètres
d'Étape" avec les attributs spécifiques à ce type d'étape (ex : côté de coupe, ordre de coupe) :

![Éditeur de Recettes - Onglet Laser](/screenshots/recipe-editor-laser.webp)

![Éditeur de Recettes - Onglet Paramètres d'Étape](/screenshots/recipe-editor-step-settings.webp)

- Sélectionner uniquement un **type de tâche** (avec "Tout Type" comme type d'étape) affiche une
  seule page "Paramètres" avec les paramètres de processus pour cette tâche
- Laisser les deux à "Tout" affiche uniquement les paramètres de mouvement de base (vitesse de coupe
  et vitesse de déplacement) partagés par toutes les étapes

Chaque ligne de paramètre est accompagnée d'un bouton d'application (une case à cocher à côté de la
ligne) :

- **Activé** : la recette applique ce paramètre à l'étape lors de son application
- **Désactivé** : la recette laisse ce paramètre de l'étape inchangé

**Onglet Post-Traitement** - Stockez les paramètres de post-traitement (entrée/sortie, passes
multiples, overscan et autres transformateurs) sur la recette afin qu'ils soient appliqués aux
étapes ciblées :

![Éditeur de Recettes - Onglet Post-Traitement](/screenshots/recipe-editor-post-processing.webp)

Chaque transformateur est accompagné d'un bouton d'application (une case à cocher à côté de la
ligne) :

- **Activé** : la recette applique les paramètres du transformateur à l'étape (son propre
  interrupteur d'activation décide s'il est activé ou désactivé)
- **Désactivé** : la recette ne touche pas à ce transformateur lorsqu'elle est appliquée

Lorsque la recette cible plusieurs types d'étape, seuls les transformateurs communs à tous sont
affichés.

### Système de Correspondance des Recettes

Rayforge suggère et applique automatiquement les recettes les plus appropriées selon :

- **Compatibilité machine** : Les recettes peuvent être spécifiques à une machine
- **Compatibilité de tête laser** : Les recettes peuvent imposer une tête spécifique sur la machine
- **Correspondance de matériau** : Les recettes peuvent cibler des matériaux spécifiques
- **Plages d'épaisseur** : Les recettes s'appliquent dans les limites d'épaisseur définies
- **Correspondance de type de tâche** : Les recettes sont liées à des catégories d'opérations
  spécifiques
- **Correspondance de type d'étape** : Les recettes peuvent cibler un type d'opération spécifique
  (ex : uniquement les étapes "Contour")

Une recette ne correspond que lorsque tous ses critères sont satisfaits. Lorsqu'une nouvelle étape
est créée, Rayforge recherche dans la bibliothèque de recettes celles qui correspondent et applique
automatiquement la meilleure. Le système utilise un algorithme de score de spécificité pour
prioriser les recettes les plus pertinentes :

1. Les recettes spécifiques à une machine sont mieux classées que les génériques
2. Les recettes spécifiques à une tête laser sont mieux classées
3. Les recettes spécifiques à un matériau sont mieux classées
4. Les recettes spécifiques à une épaisseur sont mieux classées
5. Les recettes spécifiques à un type d'étape sont mieux classées

### Appliquer des Recettes aux Étapes

Les recettes sont appliquées par étape. Ouvrez les paramètres de n'importe quelle étape et trouvez
la ligne "Recette" dans la section "Général" :

- **Choisir...** : Ouvre une liste filtrable de recettes. Utilisez le champ de recherche ou le
  bouton bascule "Afficher uniquement les recettes compatibles" pour réduire la liste ; les recettes
  compatibles correspondent au type de tâche, au type d'étape, à la machine et aux matériaux de
  stock de l'étape. Sélectionner une recette applique tous ses paramètres à l'étape.
- **Enregistrer Sous...** : Ouvre l'éditeur de recettes pré-rempli avec les paramètres, la machine,
  le matériau et l'épaisseur actuels de l'étape. Enregistrer la nouvelle recette l'applique
  immédiatement à l'étape.
- **Mettre à Jour** : Apparaît lorsque les paramètres de l'étape ont divergé de la recette qui lui a
  été appliquée (ex : après avoir modifié une valeur manuellement). Cliquer dessus écrase la recette
  enregistrée avec les paramètres actuels de l'étape.

Le nom de la recette actuellement appliquée est affiché dans la ligne. Les étapes sans recette
appliquée sont étiquetées "Paramètres Manuels".

---

**Sujets Connexes** :

- [Matériaux](materials) - Gérer les propriétés des matériaux
- [Gestion du Matériau](../features/stock-handling.md) - Travailler avec les matériaux de stock
- [Configuration Machine](../machine/general.md) - Configurer les machines et têtes laser
- [Aperçu des Opérations](../features/operations/contour.md) - Comprendre les différents types
  d'opérations
- [Règles de couleur](color-rules) - Faire correspondre les couleurs SVG aux types d'étape à
  l'importation
