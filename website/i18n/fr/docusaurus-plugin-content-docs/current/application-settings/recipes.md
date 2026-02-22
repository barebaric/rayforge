# Recettes et Paramètres

![Paramètres des Recettes](/screenshots/application-recipes.png)

Rayforge fournit un système de recettes puissant qui vous permet de créer, gérer et appliquer des paramètres cohérents à travers vos projets de découpe laser. Ce guide couvre le parcours utilisateur complet, de la création de recettes dans les paramètres généraux à leur application aux opérations et la gestion des paramètres au niveau de l'étape.

## Aperçu

Le système de recettes se compose de trois composants principaux :

1. **Gestion des Recettes** : Créer et gérer des préréglages de paramètres réutilisables
2. **Gestion du Matériau de Stock** : Définir les propriétés et l'épaisseur du matériau
3. **Paramètres d'Étape** : Appliquer et affiner les paramètres pour des opérations individuelles

## Gestion des Recettes

### Créer des Recettes

Les recettes sont des préréglages nommés qui contiennent tous les paramètres nécessaires pour des opérations spécifiques.
Vous pouvez créer des recettes via l'interface des paramètres principaux :

#### 1. Accéder au Gestionnaire de Recettes

Menu : Édition → Préférences → Recettes

#### 2. Créer une Nouvelle Recette

Cliquez sur "Ajouter une Nouvelle Recette" pour ouvrir la boîte de dialogue de l'éditeur de recettes.

**Onglet Général** - Définir le nom et la description de la recette :

![Éditeur de Recettes - Onglet Général](/screenshots/recipe-editor-general.png)

Remplissez les informations de base :

- **Nom** : Nom descriptif (ex : "Coupe Contreplaqué 3mm")
- **Description** : Description détaillée optionnelle

#### 3. Définir les Critères d'Applicabilité

**Onglet Applicabilité** - Définir quand cette recette devrait être suggérée :

![Éditeur de Recettes - Onglet Applicabilité](/screenshots/recipe-editor-applicability.png)

- **Type de Tâche** : Sélectionnez le type d'opération (Coupe, Gravure, etc.)
- **Machine** : Choisissez une machine spécifique ou laissez "Toute Machine"
- **Matériau** : Sélectionnez un type de matériau ou laissez ouvert pour tout matériau
- **Plage d'Épaisseur** : Définissez les valeurs d'épaisseur minimum et maximum

#### 4. Configurer les Paramètres

**Onglet Paramètres** - Ajuster la puissance, vitesse et autres paramètres :

![Éditeur de Recettes - Onglet Paramètres](/screenshots/recipe-editor-settings.png)

- Ajuster la puissance, vitesse et autres paramètres
- Les paramètres s'adaptent automatiquement selon le type de tâche sélectionné

### Système de Correspondance des Recettes

Rayforge suggère automatiquement les recettes les plus appropriées selon :

- **Compatibilité machine** : Les recettes peuvent être spécifiques à une machine
- **Correspondance de matériau** : Les recettes peuvent cibler des matériaux spécifiques
- **Plages d'épaisseur** : Les recettes s'appliquent dans les limites d'épaisseur définies
- **Correspondance de capacité** : Les recettes sont liées à des types d'opérations spécifiques

Le système utilise un algorithme de score de spécificité pour prioriser les recettes les plus pertinentes :

1. Les recettes spécifiques à une machine sont mieux classées que les génériques
2. Les recettes spécifiques à une tête laser sont mieux classées
3. Les recettes spécifiques à un matériau sont mieux classées
4. Les recettes spécifiques à une épaisseur sont mieux classées

---

**Sujets Connexes** :

- [Matériaux](materials) - Gérer les propriétés des matériaux
- [Gestion du Matériau](../features/stock-handling) - Travailler avec les matériaux de stock
- [Configuration Machine](../machine/general) - Configurer les machines et têtes laser
- [Aperçu des Opérations](../features/operations/contour) - Comprendre les différents types d'opérations
