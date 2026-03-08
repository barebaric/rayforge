# Passe Multiple

La passe multiple répète le parcours de coupe ou de gravure plusieurs fois, avec optionnellement une descente en Z entre les passes. C'est utile pour les matériaux épais ou la création de gravures profondes.

## Comment Ça Fonctionne

Chaque passe trace le même parcours à nouveau. Avec l'avance en Z activée, le laser se rapproche du matériau entre les passes, coupant progressivement plus profond.

## Paramètres

### Nombre de Passes

Combien de fois répéter l'étape complète (1-100). Chaque passe suit le même parcours.

- **1 passe :** Coupe unique (défaut)
- **2-3 passes :** Courant pour les matériaux moyennement épais
- **4+ passes :** Matériaux très épais ou durs

### Avance en Z par Passe

Distance pour abaisser l'axe Z entre les passes (0-50 mm). Fonctionne uniquement si votre machine a un contrôle d'axe Z.

- **0 mm :** Toutes les passes à la même profondeur (défaut)
- **Épaisseur du matériau ÷ passes :** Coupe à profondeur progressive
- **Petits incréments (0.1-0.5mm) :** Contrôle fin pour la gravure profonde

:::warning Axe Z Requis
L'avance en Z fonctionne uniquement avec les machines qui ont un contrôle d'axe Z motorisé. Pour les machines sans axe Z, toutes les passes se produisent à la même hauteur de mise au point.
:::

## Quand Utiliser la Passe Multiple

**Coupe de matériaux épais :**

Des passes multiples à la même profondeur coupent souvent plus proprement qu'une seule passe lente. La première passe crée un kerf, et les passes suivantes suivent le même parcours plus efficacement.

**Gravure profonde :**

Avec l'avance en Z, vous pouvez sculpter des motifs en relief profonds ou des gravures qui seraient impossibles en une seule passe.

**Qualité de bord améliorée :**

Des passes multiples plus rapides produisent souvent des bords plus propres qu'une passe lente, surtout dans les matériaux qui noircissent facilement.

## Conseils

- Commencez avec 2-3 passes à votre vitesse de coupe normale
- Pour les matériaux épais, augmentez les passes plutôt que de ralentir
- Activez l'avance en Z uniquement si votre machine la supporte
- Testez sur du matériau de rebut pour trouver le nombre de passes optimal

---

## Sujets Connexes

- [Coupe de Contour](operations/contour) - Opération de coupe principale
- [Gravure](operations/engrave) - Opérations de gravure
