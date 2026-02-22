# Flux de Travail de Gestion du Matériau

La gestion du matériau dans Rayforge est un processus séquentiel qui vous permet de définir le matériau physique avec lequel vous travaillerez, de lui assigner des propriétés, puis d'organiser vos éléments de design dessus. Ce guide vous accompagne dans le flux de travail complet, de l'ajout de matériau à la disposition automatique de votre design.

## 1. Ajouter du Matériau

Le matériau représente la pièce physique de matériau que vous allez couper ou graver. Pour ajouter du matériau à votre document :

1. Dans le panneau **Matériau** dans la barre latérale, cliquez sur le bouton **Ajouter Matériau**
2. Un nouvel élément de matériau sera créé avec les dimensions par défaut (80% de l'espace de travail de votre machine)
3. Le matériau apparaîtra comme un rectangle dans l'espace de travail, centré sur le lit de la machine

### Propriétés du Matériau

Chaque élément de matériau a les propriétés suivantes :
- **Nom** : Un nom descriptif pour l'identification (numérotation automatique comme "Matériau 1", "Matériau 2", etc.)
- **Dimensions** : Largeur et hauteur du matériau
- **Épaisseur** : L'épaisseur du matériau (optionnel mais recommandé pour un aperçu 3D précis)
- **Matériau** : Le type de matériau (assigné à l'étape suivante)
- **Visibilité** : Basculer pour afficher/masquer le matériau dans l'espace de travail

### Gérer les Éléments de Matériau

- **Renommer** : Ouvrez la boîte de dialogue Propriétés du Matériau et éditez le champ nom
- **Redimensionner** : Sélectionnez l'élément de matériau dans l'espace de travail et glissez les poignées d'angle pour redimensionner
- **Déplacer** : Sélectionnez l'élément de matériau dans l'espace de travail et glissez pour le repositionner
- **Supprimer** : Cliquez sur le bouton de suppression (icône poubelle) à côté de l'élément de matériau dans le panneau Matériau
- **Éditer les propriétés** : Cliquez sur le bouton propriétés (icône document) pour ouvrir la boîte de dialogue Propriétés du Matériau
- **Basculer la visibilité** : Cliquez sur le bouton visibilité (icône œil) pour afficher/masquer l'élément de matériau

## 2. Assigner un Matériau

Une fois que vous avez défini le matériau, vous pouvez lui assigner un type de matériau :

1. Dans le panneau **Matériau**, cliquez sur le bouton propriétés (icône document) sur l'élément de matériau
2. Dans la boîte de dialogue Propriétés du Matériau, cliquez sur le bouton **Sélectionner** à côté du champ Matériau
3. Parcourez vos bibliothèques de matériaux et sélectionnez le matériau approprié
4. Le matériau se mettra à jour pour afficher l'apparence visuelle du matériau

### Propriétés du Matériau

Les matériaux définissent les propriétés visuelles de votre matériau :
- **Apparence visuelle** : Couleur et motif pour la visualisation
- **Catégorie** : Groupement (ex : "Bois", "Acrylique", "Métal")
- **Description** : Informations supplémentaires sur le matériau

Note : Les propriétés de matériau sont définies dans les bibliothèques de matériaux et ne peuvent pas être éditées via la boîte de dialogue des propriétés du matériau. Les propriétés du matériau vous permettent seulement d'assigner un matériau à un élément de matériau.

## 3. Assigner du Matériau aux Calques

Après avoir défini votre matériau et assigné des matériaux, vous pouvez associer des calques avec des éléments de matériau spécifiques :

1. Dans le panneau **Calques**, localisez le calque que vous voulez assigner au matériau
2. Cliquez sur le bouton d'assignation de matériau (affiche "Surface Entière" par défaut)
3. Dans le menu déroulant, sélectionnez l'élément de matériau que vous voulez associer avec ce calque
4. Le contenu de ce calque sera maintenant contraint aux limites du matériau assigné

Vous pouvez aussi choisir "Surface Entière" pour utiliser tout l'espace de travail machine au lieu d'un élément de matériau spécifique.

### Pourquoi Assigner du Matériau aux Calques ?

- **Limites de disposition** : Fournit des limites pour que l'algorithme de disposition automatique fonctionne à l'intérieur
- **Organisation visuelle** : Aide à organiser votre design en associant des calques avec des matériaux physiques
- **Visualisation du matériau** : Affiche l'apparence visuelle du matériau assigné sur le matériau

## 4. Disposition Automatique

La fonction de disposition automatique vous aide à organiser efficacement vos éléments de design :

1. Sélectionnez les éléments que vous voulez organiser (ou ne sélectionnez rien pour organiser tous les éléments du calque actif)
2. Cliquez sur le bouton **Organiser** dans la barre d'outils et sélectionnez **Disposition Automatique (empaqueter les pièces)**
3. Rayforge organisera automatiquement les éléments pour optimiser l'utilisation du matériau

### Comportement de la Disposition Automatique

L'algorithme de disposition automatique fonctionne différemment selon votre configuration de calque :

- **Si un élément de matériau est assigné au calque** : Les éléments sont organisés dans les limites de cet élément de matériau spécifique
- **Si "Surface Entière" est sélectionné** : Les éléments sont organisés sur tout l'espace de travail machine

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
5. **Utilisez des calques pour l'organisation** - Séparez les différentes parties de votre design en calques avant d'assigner au matériau
6. **Vérifiez l'ajustement avant de couper** - Utilisez la vue 2D pour vérifier que tout tient sur votre matériau

## Dépannage

### La disposition automatique ne fonctionne pas comme prévu
- Vérifiez si votre calque a un matériau assigné
- Assurez-vous que les éléments ne sont pas groupés (dégroupez-les d'abord)
- Essayez de réduire le nombre d'éléments sélectionnés à la fois
- Vérifiez que les éléments tiennent dans les limites (matériau ou surface entière)
