# Gestion du Matériau

Le matériau dans Rayforge représente le matériel physique que vous allez couper ou graver. Le matériau est un concept **global au document**—votre document peut avoir un ou plusieurs éléments de matériau, et ils existent indépendamment des calques.

## Ajouter du Matériau

Le matériau représente la pièce physique de matériel avec lequel vous allez travailler. Pour ajouter du matériau à votre document :

1. Ouvrez l'onglet **Actifs** dans le panneau inférieur
2. Cliquez sur **Ajouter Matériau** — ce bouton est affiché lorsque la liste des actifs est vide
3. Un nouvel élément de matériau sera créé avec les dimensions par défaut (80% de l'espace de travail de votre machine), placé avec un coin à l'origine de la machine
4. Le matériau apparaîtra comme un rectangle dans l'espace de travail

Vous pouvez également ajouter du matériau en :

- Utilisant le bouton **+** dans l'onglet Actifs et en choisissant **Ajouter Matériau**
- Faisant un clic droit sur une zone vide de la liste des actifs et en sélectionnant **Nouveau Matériau**
- Faisant un clic droit sur une zone vide du canevas et en sélectionnant **Nouveau Matériau**
- Appuyant sur `Ctrl+Alt+S`

Vous pouvez aussi glisser un actif de matériau depuis l'onglet Actifs vers le canevas pour placer un autre élément de matériau.

### Propriétés du Matériau

Double-cliquez sur un actif de matériau dans l'onglet Actifs pour ouvrir la boîte de dialogue **Propriétés du Matériau**. Chaque élément de matériau a les propriétés suivantes :

- **Nom** : Un nom descriptif pour l'identification (numérotation automatique comme "Matériau 1", "Matériau 2", etc.)
- **Épaisseur** : L'épaisseur du matériau (optionnel mais recommandé pour un aperçu 3D précis)
- **Matériau** : Le type de matériau (assigné à l'étape suivante)

Les dimensions du matériau sont ajustées en sélectionnant le matériau dans l'espace de travail et en glissant ses poignées d'angle.

### Gérer les Éléments de Matériau

- **Renommer** : Ouvrez la boîte de dialogue Propriétés du Matériau (double-clic sur l'actif) et éditez le champ nom
- **Redimensionner** : Sélectionnez l'élément de matériau dans l'espace de travail et glissez les poignées d'angle pour redimensionner
- **Déplacer** : Sélectionnez l'élément de matériau dans l'espace de travail et glissez pour le repositionner
- **Éditer les propriétés** : Double-cliquez sur l'actif de matériau dans l'onglet Actifs
- **Dupliquer** : Faites un clic droit sur l'actif de matériau dans l'onglet Actifs et sélectionnez **Dupliquer**
- **Supprimer** : Faites un clic droit sur l'actif de matériau dans l'onglet Actifs et sélectionnez **Supprimer**, ou sélectionnez l'actif et appuyez sur `Suppr`

## Assigner un Matériau

Une fois que vous avez défini le matériau, vous pouvez lui assigner un type de matériau :

1. Double-cliquez sur l'actif de matériau dans l'onglet **Actifs** pour ouvrir la boîte de dialogue Propriétés du Matériau
2. Dans la boîte de dialogue Propriétés du Matériau, cliquez sur le bouton **Sélectionner** à côté du champ Matériau
3. Parcourez vos bibliothèques de matériaux et sélectionnez le matériau approprié
4. Le matériau se mettra à jour pour afficher l'apparence visuelle du matériau

### Propriétés du Matériau

Les matériaux définissent les propriétés visuelles de votre matériau :

- **Apparence visuelle** : Couleur et motif pour la visualisation
- **Catégorie** : Groupement (ex : "Bois", "Acrylique", "Métal")
- **Description** : Informations supplémentaires sur le matériau

Note : Les propriétés de matériau sont définies dans les bibliothèques de matériaux et ne peuvent pas être éditées via la boîte de dialogue des propriétés du matériau. Les propriétés du matériau vous permettent seulement d'assigner un matériau à un élément de matériau.

## Convertir des Pièces en Matériau

Vous pouvez convertir n'importe quelle pièce en un élément de matériau. C'est utile lorsque vous avez une pièce de matériau de forme irrégulière et souhaitez utiliser son contour exact comme limite de matériau.

Pour convertir une pièce en matériau :

1. Faites un clic droit sur la pièce dans le canevas
2. Sélectionnez **Convertir en Matériau** dans le menu contextuel
3. La pièce sera remplacée par un nouvel élément de matériau avec la même forme et position

Le nouvel élément de matériau :

- Utilise la géométrie de la pièce comme sa limite
- Hérite du nom de la pièce
- Peut se voir attribuer un matériau comme tout autre élément de matériau

## Disposition Automatique

La fonction de disposition automatique vous aide à organiser efficacement vos éléments de design dans les limites du matériau :

1. Sélectionnez les éléments que vous voulez organiser (ou ne sélectionnez rien pour organiser tous les éléments du calque actif)
2. Cliquez sur le bouton **Organiser** dans la barre d'outils et sélectionnez **Disposition Automatique (empaqueter les pièces)**
3. Rayforge organisera automatiquement les éléments pour optimiser l'utilisation du matériau

### Comportement de la Disposition Automatique

L'algorithme de disposition automatique organise les éléments dans les limites des éléments de matériau visibles dans votre document :

- **Si des éléments de matériau sont définis** : Les éléments sont organisés dans les limites des éléments de matériau visibles
- **Si aucun matériau n'est défini** : Les éléments sont organisés sur tout l'espace de travail machine

L'algorithme considère :

- **Limites des éléments** : Respecte les dimensions de chaque élément de design
- **Rotation** : Peut faire pivoter les éléments par incréments de 90 degrés pour un meilleur ajustement
- **Espacement** : Maintient une marge entre les éléments (0.5mm par défaut)
- **Limites du matériau** : Garde tous les éléments dans les limites définies

### Alternatives de Disposition Manuelle

Si vous préférez plus de contrôle, Rayforge offre aussi des outils de disposition manuelle :

- **Outils d'alignement** : Aligner à gauche, à droite, au centre, en haut, en bas
- **Outils de distribution** : Répartir les éléments horizontalement ou verticalement
- **Positionnement individuel** : Cliquez et glissez les éléments pour les placer manuellement

## Conseils pour une Gestion Efficace du Matériau

1. **Commencez avec des dimensions de matériau précises** - Mesurez votre matériau avec précision pour de meilleurs résultats
2. **Utilisez des noms descriptifs** - Nommez clairement vos éléments de matériau (ex : "Contreplaqué Bouleau 3mm")
3. **Définissez l'épaisseur du matériau** - Cela peut être utile pour des calculs et références futurs
4. **Assignez les matériaux tôt** - Cela assure une représentation visuelle appropriée dès le début
5. **Utilisez du matériau irrégulier pour les chutes** - Convertissez des pièces en matériau lorsque vous utilisez du matériel restant avec des formes personnalisées
6. **Vérifiez l'ajustement avant de couper** - Utilisez la vue 2D pour vérifier que tout tient sur votre matériau

## Dépannage

### La disposition automatique ne fonctionne pas comme prévu

- Assurez-vous qu'au moins un élément de matériau est défini (la disposition automatique utilise le premier élément de matériau visible comme limite)
- Assurez-vous que les éléments ne sont pas groupés (dégroupez-les d'abord)
- Essayez de réduire le nombre d'éléments sélectionnés à la fois
- Vérifiez que les éléments tiennent dans les limites du matériau
