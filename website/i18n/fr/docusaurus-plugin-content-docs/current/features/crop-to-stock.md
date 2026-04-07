# Rognage au Stock

Le rognage au stock limite les parcours de coupe à ta limite de matériau. Toutes les coupes qui s'étendent au-delà de la zone de stock sont rognées, empêchant le laser de couper en dehors de ton matériau.

## Comment ça fonctionne

Le transformateur compare tes parcours de coupe à la limite de stock définie. Les segments de parcours en dehors de cette limite sont supprimés ou coupés au bord du stock.

Si aucun élément de stock n'est défini dans ton document, la limite de rognage utilise la zone de travail de la machine à la place.

## Paramètres

### Activer le rognage au stock

Active ou désactive le rognage. Désactivé par défaut.

### Décalage

Ajuste la limite de stock effective avant le rognage (-100 à +100 mm).

- **Valeurs positives :** Réduire la limite (coupes plus conservatrices)
- **Valeurs négatives :** Étendre la limite (permet des coupes plus proches du bord)
- **0 mm :** Utiliser la limite de stock exacte

Utilise le décalage quand tu veux une marge de sécurité du bord du stock, ou quand le placement de ton matériau n'est pas parfaitement aligné.

## Quand utiliser le rognage au stock

**Designs partiels :** Ton design est plus grand que ton matériau, mais tu veux couper uniquement la partie qui rentre.

**Marge de sécurité :** Empêche les coupes accidentelles au-delà des bords du matériau.

**Feuilles imbriquées :** Coupe uniquement les parties qui rentrent sur ta pièce actuelle de matériau.

**Coupes de test :** Limite un test à une zone spécifique de ton matériau.

## Exemple

Tu as un grand design mais seulement une petite pièce de matériau :

1. Définis la taille de ton stock pour correspondre à ton matériau
2. Active le rognage au stock
3. Définis le décalage à 2mm pour la marge de sécurité
4. Seules les portions dans ta limite de matériau seront coupées

---

## Sujets connexes

- [Gestion du Stock](stock-handling) - Configurer les limites de matériau
- [Coupe de Contour](operations/contour) - Opération de coupe principale
