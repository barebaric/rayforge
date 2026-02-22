# Compensation de Kerf

Le kerf est le matériau retiré par le faisceau laser pendant la coupe. La compensation de kerf ajuste les parcours d'outil pour en tenir compte, assurant que les pièces coupées correspondent à leurs dimensions de conception.

## Qu'est-ce que le Kerf ?

**Kerf** = la largeur du matériau retiré par le processus de coupe.

**Exemple :**
- Taille du spot laser : 0.2mm
- Interaction matéri : ajoute ~0.1mm de chaque côté
- **Kerf total :** ~0.4mm

---

## Comment Fonctionne la Compensation de Kerf

La compensation de kerf **décale le parcours d'outil** vers l'intérieur ou l'extérieur pour tenir compte du retrait de matériau :

**Pour les coupes extérieures (couper une pièce) :**
- Décaler le parcours **vers l'extérieur** de la moitié de la largeur du kerf
- Résultat : La pièce finale a la bonne taille

**Pour les coupes intérieures (couper un trou) :**
- Décaler le parcours **vers l'intérieur** de la moitié de la largeur du kerf
- Résultat : Le trou final a la bonne taille

**Exemple avec un kerf de 0.4mm :**

```
Parcours original :  carré de 50mm
Compensation :       Décalage vers l'extérieur de 0.2mm (demi kerf)
Le laser suit :      carré de 50.4mm
Après coupe :        La pièce mesure 50.0mm (parfait !)
```

---

## Mesurer le Kerf

**Procédure de mesure précise du kerf :**

1. **Créez un fichier de test :**
   - Dessinez un carré de 50mm x 50mm
   - Dessinez un cercle (n'importe quelle taille, pour test de coupe intérieure)

2. **Coupez le test :**
   - Utilisez vos paramètres de coupe normaux
   - Coupez complètement à travers
   - Laissez le matériau refroidir

3. **Mesurez :**
   - **Carré extérieur (pièce) :** Mesurez avec un pied à coulisse
     - Si < 50mm, le kerf a été retiré vers l'extérieur
     - Kerf = (50 - mesuré) x 2
   - **Cercle intérieur (trou) :** Mesurez le diamètre
     - Si > diamètre de conception, le kerf a été retiré vers l'intérieur
     - Kerf = (mesuré - conçu) / 2

4. **Moyenne :** Utilisez la moyenne de plusieurs mesures

**Variables affectant le kerf :**
- Puissance laser (plus élevée = plus large)
- Vitesse de coupe (plus lente = plus large)
- Type et densité du matériau
- Distance de mise au point
- Pression de l'assistance air

---

## Compensation de Kerf Manuelle

Si la compensation de kerf automatisée n'est pas disponible, compensez dans votre logiciel de conception :

**Inkscape :**

1. **Sélectionnez le parcours**
2. **Parcours → Décalage Dynamique** (Ctrl+J)
3. **Glissez pour décaler** de la moitié de votre mesure de kerf
   - Vers l'extérieur pour les pièces (pour agrandir le parcours)
   - Vers l'intérieur pour les trous (pour rétrécir le parcours)
4. **Parcour → Objet en Parcour** pour finaliser

**Illustrator :**

1. **Sélectionnez le parcours**
2. **Objet → Parcour → Décalage de Parcour**
3. **Entrez la valeur de décalage :** (kerf / 2)
   - Positif pour vers l'extérieur, négatif pour vers l'intérieur
4. **OK** pour appliquer

**Fusion 360 / CAO :**

- Décalez les entités d'esquisse avant l'exportation
- Utilisez la dimension kerf/décalage

---

## Pages Connexes

- [Opération Contour](operations/contour) - Opérations de coupe
- [Grille de Test de Matériau](operations/material-test-grid) - Trouver les paramètres optimaux
