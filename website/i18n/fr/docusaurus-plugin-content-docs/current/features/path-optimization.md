# Optimisation de Parcours

L'optimisation de parcours réordonne les segments de coupe pour minimiser la distance de déplacement. Le laser se déplace efficacement entre les coupes au lieu de sauter aléatoirement sur la zone de travail.

## Comment Ça Fonctionne

Sans optimisation, les parcours sont coupés dans l'ordre où ils apparaissent dans votre fichier de design. L'optimisation analyse tous les segments de parcours et les réarrange pour que le laser parcoure la distance totale la plus courte entre les coupes.

**Avant optimisation :** Le laser va et vient sur le matériau
**Après optimisation :** Le laser se déplace séquentiellement de coupe en coupe

## Paramètres

### Activer l'Optimisation

Activez ou désactivez l'optimisation de parcours. Activée par défaut pour la plupart des opérations.

## Quand Utiliser l'Optimisation

**Activer pour :**

- Designs avec beaucoup de formes séparées
- Réduire le temps total du travail
- Minimiser l'usure du système de mouvement
- Dispositions imbriquées complexes

**Désactiver pour :**

- Designs où l'ordre de coupe compte (ex: caractéristiques intérieures avant extérieures)
- Débogage de problèmes de parcours
- Quand vous avez besoin d'un ordre d'exécution prévisible et répétable

## Comment Ça Affecte Votre Travail

**Gain de temps :** Peut réduire le temps de travail de 20-50% pour les designs avec beaucoup de coupes séparées.

**Efficacité de mouvement :** Moins de mouvement rapide signifie moins d'usure sur les courroies, moteurs et roulements.

**Distribution de chaleur :** Les parcours optimisés peuvent concentrer la chaleur dans une zone. Pour les matériaux sensibles à la chaleur, considérez si l'ordre compte.

:::tip
L'optimisation s'exécute automatiquement. Activez-la simplement et le logiciel gère le reste.
:::

---

## Sujets Connexes

- [Coupe de Contour](operations/contour) - Opération de coupe principale
- [Ponts de Maintien](holding-tabs) - Maintenir les pièces sécurisées pendant la coupe
