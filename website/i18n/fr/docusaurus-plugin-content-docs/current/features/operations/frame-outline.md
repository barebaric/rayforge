# Contour de Cadrage

Le Contour de Cadrage crée un parcours de coupe rectangulaire simple autour de votre design complet. C'est la façon la plus rapide d'ajouter une bordure propre ou de couper votre travail libre de la feuille de matériau.

## Aperçu

Les opérations de Contour de Cadrage :

- Créent une limite rectangulaire autour de tout le contenu
- Ajoutent un décalage/marge configurable depuis le design
- Supportent la compensation de kerf pour un dimensionnement précis
- Fonctionnent avec toute combinaison d'objets sur le canevas

![Paramètres d'étape contour de cadrage](/screenshots/step-settings-frame-outline-general.png)

## Quand Utiliser le Contour de Cadrage

Utilisez le contour de cadrage pour :

- Ajouter une bordure décorative autour de votre design
- Couper votre travail libre de la feuille de matériau
- Créer une limite rectangulaire simple
- Cadrage rapide sans calculs de parcours complexes

**N'utilisez pas le contour de cadrage pour :**

- Formes irrégulières autour de plusieurs objets (utilisez [Shrink Wrap](shrink-wrap) à la place)
- Couper des pièces individuelles (utilisez [Contour](contour) à la place)
- Suivre la forme exacte de votre design

## Créer une Opération de Contour de Cadrage

### Étape 1 : Organiser Votre Design

1. Placez tous les objets sur le canevas
2. Positionnez-les où vous voulez qu'ils soient par rapport au cadre
3. Le cadre sera calculé autour de la boîte englobante de tout le contenu

### Étape 2 : Ajouter une Opération de Contour de Cadrage

- **Menu :** Opérations → Ajouter Contour de Cadrage
- **Clic droit :** Menu contextuel → Ajouter Opération → Contour de Cadrage

### Étape 3 : Configurer les Paramètres

Configurez les paramètres du cadre :

- **Puissance & Vitesse :** Correspondez aux exigences de coupe de votre matériau
- **Décalage :** Distance du bord du contenu au cadre
- **Décalage de Parcours :** Coupe intérieur, extérieur ou sur la ligne centrale

## Paramètres Clés

### Puissance & Vitesse

**Puissance (%) :**

- Intensité laser pour couper le cadre
- Correspondez aux exigences de coupe de votre matériau

**Vitesse (mm/min) :**

- À quelle vitesse le laser se déplace
- Plus lent pour les matériaux plus épais

**Passes :**

- Nombre de fois pour couper le cadre
- Généralement 1-2 passes
- Ajoutez des passes pour les matériaux plus épais

### Distance de Décalage

**Décalage (mm) :**

- Distance de la boîte englobante du design au cadre
- Crée une marge/bordure autour de votre travail

**Valeurs typiques :**

- **0mm :** Le cadre touche le bord du design
- **2-5mm :** Petite marge pour une apparence propre
- **10mm+ :** Grande bordure pour le montage ou la manipulation

### Décalage de Parcours (Côté de Coupe)

Contrôle où le laser coupe par rapport au parcours du cadre :

| Côté de Coupe | Description | Utilisation |
| ------------- | ----------- | ----------- |
| **Ligne Centrale** | Coupe directement sur le parcours | Coupe standard |
| **Extérieur** | Coupe à l'extérieur du parcours du cadre | Rend le cadre légèrement plus grand |
| **Intérieur** | Coupe à l'intérieur du parcours du cadre | Rend le cadre légèrement plus petit |

### Compensation de Kerf

Le contour de cadrage supporte la compensation de kerf :

- Ajuste automatiquement pour la largeur du faisceau laser
- Assure des dimensions finales précises
- Utilise la valeur de kerf de vos paramètres de tête laser

## Options de Post-Traitement

![Paramètres de post-traitement du contour de cadrage](/screenshots/step-settings-frame-outline-post.png)

### Passe Multiple

Coupez le cadre plusieurs fois :

- **Passes :** Nombre de répétitions
- **Avance en Z :** Abaisser Z entre les passes (nécessite axe Z)
- Utile pour les matériaux épais

### Ponts de Maintien

Ajoutez des ponts pour maintenir la pièce encadrée attachée :

- Empêche les pièces de tomber pendant la coupe
- Configurez la largeur, hauteur et espacement des ponts
- Voir [Ponts de Maintien](../holding-tabs) pour plus de détails

## Cas d'Utilisation

### Bordure Décorative

**Scénario :** Ajouter une bordure rectangulaire propre autour d'une plaque ou enseigne

**Processus :**

1. Concevez votre contenu (texte, logos, etc.)
2. Ajoutez un Contour de Cadrage avec un décalage de 3-5mm
3. Coupez avec des paramètres de scorring décoratif (faible puissance)

**Résultat :** Pièce encadrée d'apparence professionnelle

### Couper Libre de la Feuille

**Scénario :** Retirer votre travail fini de la feuille de matériau

**Processus :**

1. Complétez toutes les autres opérations (gravure, coupes de contour)
2. Ajoutez un Contour de Cadrage comme dernière opération
3. Définissez le décalage pour inclure une petite marge

**Avantages :**

- Séparation propre de la feuille
- Qualité de bord cohérente
- Facile à exécuter comme étape finale

### Limite de Traitement par Lot

**Scénario :** Créer une limite de coupe pour plusieurs pièces imbriquées

**Processus :**

1. Organisez toutes les pièces sur le canevas
2. Ajoutez des opérations de contour individuelles pour les pièces
3. Ajoutez un Contour de Cadrage autour de tout
4. Le cadre coupe en dernier (dans un calque séparé)

**Ordre :** Gravure → Contours des pièces → Contour de cadrage

## Conseils & Meilleures Pratiques

### Ordre des Calques

**Meilleure pratique :**

- Placez le Contour de Cadrage dans son propre calque
- Exécutez le cadre comme **dernière** opération
- Cela assure que tout autre travail se termine d'abord

**Pourquoi en dernier ?**

- Le matériau reste sécurisé pendant les autres opérations
- Empêche les pièces de bouger
- Résultat final plus propre

### Sélection du Décalage

**Choisir le décalage :**

- **0-2mm :** Ajustement serré, gaspillage de matériau minimal
- **3-5mm :** Marge standard, apparence professionnelle
- **10mm+ :** Matériau supplémentaire pour manipulation/montage

**Considérez :**

- L'utilisation finale de la pièce
- Si les bords seront visibles
- Coût et disponibilité du matériau

### Paramètres de Qualité

**Pour des coupes de cadre propres :**

- Utilisez l'assistance air
- Assurez une mise au point appropriée
- Passes multiples plus rapides souvent meilleures qu'une passe lente
- Gardez le matériau plat et sécurisé

## Combiner avec d'Autres Opérations

### Cadre + Gravure + Contour

Flux de travail typique pour une pièce finie :

1. **Calque 1 :** Gravure des détails (texte, images)
2. **Calque 2 :** Coupe de contour des pièces individuelles
3. **Calque 3 :** Contour de cadrage (coupe libre)

**L'ordre d'exécution assure :**

- La gravure se produit pendant que le matériau est plat et sécurisé
- Les détails des pièces sont coupés avant la séparation finale
- Le cadre coupe tout libre à la fin

### Cadre vs Shrink Wrap

| Fonctionnalité | Contour de Cadrage | Shrink Wrap |
| ------------- | ------------------ | ----------- |
| **Forme** | Toujours rectangulaire | Suit les contours des objets |
| **Vitesse** | Très rapide (4 lignes) | Dépend de la complexité |
| **Cas d'utilisation** | Bordures simples, coupe libre | Utilisation efficace du matériau |
| **Flexibilité** | Rectangle fixe | S'adapte au design |

**Choisissez Contour de Cadrage quand :**

- Vous voulez une bordure rectangulaire
- La simplicité est préférée
- Couper libre de la feuille

**Choisissez Shrink Wrap quand :**

- Vous voulez minimiser le gaspillage de matériau
- Le design a une forme irrégulière
- L'efficacité est importante

## Dépannage

### Le cadre est trop serré/lâche

- **Ajustez :** Le paramètre de distance de décalage
- **Vérifiez :** Le décalage de parcours (intérieur/extérieur/ligne centrale)
- **Vérifiez :** La compensation de kerf est correcte

### Le cadre n'apparaît pas

- **Vérifiez :** Les objets sont sur le canevas
- **Vérifiez :** L'opération est activée
- **Regardez :** Le cadre peut être en dehors de la zone visible (dézoomez)

### Le cadre coupe dans le design

- **Augmentez :** La distance de décalage
- **Vérifiez :** Les objets sont correctement positionnés
- **Vérifiez :** Le calcul de la boîte englobante inclut tous les objets

### Profondeur de coupe incohérente

- **Vérifiez :** Le matériau est plat
- **Vérifiez :** La distance de mise au point est correcte
- **Essayez :** Passes multiples à puissance plus basse

## Détails Techniques

### Calcul de la Boîte Englobante

Le contour de cadrage utilise la boîte englobante combinée de :

- Toutes les pièces sur le canevas
- Leurs positions transformées finales
- Incluant toutes rotations/mises à l'échelle appliquées

### Génération de Parcours

1. Calculer la boîte englobante combinée
2. Appliquer la distance de décalage
3. Appliquer le décalage de parcours (intérieur/extérieur/ligne centrale)
4. Appliquer la compensation de kerf
5. Générer le parcours G-code rectangulaire

### Exemple G-code

```gcode
G0 X5 Y5           ; Déplacer au début du cadre (avec décalage)
M3 S200            ; Laser activé à 80% puissance
G1 X95 Y5 F500     ; Couper le bord inférieur
G1 X95 Y95         ; Couper le bord droit
G1 X5 Y95          ; Couper le bord supérieur
G1 X5 Y5           ; Couper le bord gauche (compléter)
M5                 ; Laser éteint
```

## Sujets Connexes

- **[Coupe de Contour](contour)** - Contours d'objets individuels de coupe
- **[Shrink Wrap](shrink-wrap)** - Limites irrégulières efficaces
- **[Ponts de Maintien](../holding-tabs)** - Maintenir les pièces sécurisées pendant la coupe
- **[Flux de Travail Multi-Couches](../multi-layer)** - Organiser efficacement les opérations
- **[Compensation de Kerf](../kerf)** - Améliorer la précision dimensionnelle
