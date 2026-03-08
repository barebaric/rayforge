# Lissage de Parcours

Le lissage de parcours réduit les bords irréguliers et les transitions brusques dans vos parcours de coupe, résultant en des courbes plus propres et un mouvement machine plus fluide.

## Comment Ça Fonctionne

Le lissage applique un filtre à votre géométrie de parcours qui arrondit les coins angulaires et lisse les bords rugueux. Le laser suit une trajectoire plus douce au lieu de faire des changements de direction abrupts.

## Paramètres

### Activer le Lissage

Activez ou désactivez le lissage pour cette opération. Le lissage est désactivé par défaut.

### Douceur

Contrôle le niveau de lissage du parcours (0-100). Des valeurs plus élevées produisent des courbes plus arrondies mais peuvent s'écarter davantage du parcours original.

- **Bas (0-30) :** Lissage minimal, préserve les détails nets
- **Moyen (30-60) :** Lissage équilibré pour la plupart des designs
- **Élevé (60-100) :** Lissage agressif, idéal pour les formes organiques

### Seuil d'Angle de Coin

Les angles plus aigus que cette valeur sont préservés comme coins plutôt que lissés (0-179 degrés). Cela empêche les caractéristiques pointues importantes d'être arrondies.

- **Valeurs plus basses :** Plus de coins sont lissés, résultat plus arrondi
- **Valeurs plus élevées :** Plus de coins sont préservés, résultat plus net

## Quand Utiliser le Lissage

**Bon pour :**

- Designs importés de sources basées sur pixels avec effets d'escalier
- Réduire le stress mécanique sur les changements de direction rapides
- Améliorer la qualité de coupe sur les courbes
- Designs avec beaucoup de petits segments de ligne

**Non nécessaire pour :**

- Illustrations vectorielles propres avec des courbes bezier lisses
- Designs où les coins nets doivent être préservés exactement
- Dessins techniques nécessitant une géométrie précise

---

## Sujets Connexes

- [Coupe de Contour](operations/contour) - Opération de coupe principale
- [Optimisation de Parcours](path-optimization) - Réduire la distance de déplacement
