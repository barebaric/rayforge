# Rognage au Stock

Le rognage au stock limite les parcours de coupe à votre limite de matériau. Toutes les coupes qui s'étendent au-delà de la zone de stock sont rognées, empêchant le laser de couper en dehors de votre matériau.

## Comment Ça Fonctionne

Le transformateur compare vos parcours de coupe à la limite de stock définie. Les segments de parcours en dehors de cette limite sont supprimés ou coupés au bord du stock.

## Paramètres

### Activer le Rognage au Stock

Activez ou désactivez le rognage. Désactivé par défaut.

### Décalage

Ajuste la limite de stock effective avant le rognage (-100 à +100 mm).

- **Valeurs positives :** Réduire la limite (coupes plus conservatrices)
- **Valeurs négatives :** Étendre la limite (permet des coupes plus proches du bord)
- **0 mm :** Utiliser la limite de stock exacte

Utilisez le décalage quand vous voulez une marge de sécurité du bord du stock, ou quand le placement de votre matériau n'est pas parfaitement aligné.

## Quand Utiliser le Rognage au Stock

**Designs partiels :** Votre design est plus grand que votre matériau, mais vous voulez couper uniquement la partie qui rentre.

**Marge de sécurité :** Empêche les coupes accidentelles au-delà des bords du matériau.

**Feuilles imbriquées :** Coupez uniquement les parties qui rentrent sur votre pièce actuelle de matériau.

**Coupes de test :** Limitez un test à une zone spécifique de votre matériau.

## Exemple

Vous avez un grand design mais seulement une petite pièce de matériau :

1. Définissez la taille de votre stock pour correspondre à votre matériau
2. Activez le Rognage au Stock
3. Définissez le décalage à 2mm pour la marge de sécurité
4. Seules les portions dans votre limite de matériau seront coupées

---

## Sujets Connexes

- [Gestion du Stock](stock-handling) - Configurer les limites de matériau
- [Coupe de Contour](operations/contour) - Opération de coupe principale
