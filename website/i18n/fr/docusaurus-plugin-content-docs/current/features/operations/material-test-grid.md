---
description:
  "Génère une grille de test de matériau pour trouver les réglages de puissance et de vitesse
  optimaux pour n'importe quel matériau. Calibre ton découpeur laser de façon systématique."
---

# Grille de Test de Matériau

Chaque matériau — et souvent chaque couleur et chaque épaisseur d'un même matériau — réagit
différemment à la puissance et à la vitesse du laser. La Grille de Test de Matériau élimine les
conjectures dans la recherche de la bonne combinaison : elle génère un motif de cellules de test
dans lesquelles chaque cellule est gravée ou découpée avec un réglage légèrement différent, le tout
en un seul travail. Après une seule exécution, tu vois d'un coup d'œil quelle combinaison produit le
résultat souhaité.

Crée-en une via **Outils → Créer une Grille de Test de Matériau**. Rayforge ajoute une pièce
spéciale sur le canevas ainsi qu'une opération correspondante, et tu configures la grille dans sa
boîte de dialogue de paramètres.

![Paramètres de la Grille de Test de Matériau](/screenshots/material-test.webp)

## Préréglages

La boîte de dialogue des paramètres propose des préréglages pour les types de laser courants. Ils
remplissent une plage de vitesse, une plage de puissance et un type de test raisonnables afin que tu
démarres avec une base solide :

| Préréglage        | Plage de Vitesse  | Plage de Puissance | Type de Test |
| ----------------- | ----------------- | ------------------ | ------------ |
| **Gravure Diode** | 1000-10000 mm/min | 10-100%            | Gravure      |
| **Coupe Diode**   | 100-5000 mm/min   | 50-100%            | Coupe        |
| **Gravure CO2**   | 3000-20000 mm/min | 10-50%             | Gravure      |
| **Coupe CO2**     | 1000-20000 mm/min | 30-100%            | Coupe        |

Un préréglage n'est qu'un point de départ — chaque valeur reste ajustable ensuite, et les plages de
vitesse sont automatiquement limitées aux capacités de ta machine.

## Modes de Grille

Une grille de test fait varier deux paramètres à la fois : un sur les colonnes et un sur les lignes.
Le mode de grille détermine lesquels. **Puissance vs Vitesse** est le mode par défaut et couvre la
question la plus courante — la puissance sur les colonnes, la vitesse sur les lignes.

**Puissance vs Passes** et **Vitesse vs Passes** maintiennent l'un des deux fixe et font varier le
nombre de passes à la place, ce qui est utile pour couper des matériaux épais. **Vitesse vs
Décalage** est un mode de calibration spécial pour la gravure bidirectionnelle : il fait varier le
décalage horizontal de balayage afin que tu puisses corriger le désalignement entre les lignes.
Comme cela n'a de sens que pour un travail raster, sa sélection bascule la grille en Gravure et
élargit l'espacement des lignes pour que tout désalignement soit facile à voir. Dans chaque ligne,
la puissance est mise à l'échelle avec la vitesse, afin que toutes les cellules restent visuellement
comparables.

## Configuration de la Grille

La boîte de dialogue des paramètres regroupe les paramètres en trois sections.

La section **Grille** contrôle le test lui-même. Le type de test détermine si chaque cellule découpe
le contour d'un carré ou le remplit de lignes raster. Les dimensions de la grille définissent
combien de colonnes et de lignes tester — chaque colonne représente un pas du premier paramètre du
mode et chaque ligne un pas du second, du minimum au maximum de la plage saisie. Entre 2 et 20 pas
sont autorisés par axe ; 5×5 est une bonne valeur par défaut. La taille de la forme (10 mm par
défaut) et l'espacement (2 mm par défaut) déterminent la taille de la grille. Pour le type de test
Gravure, l'intervalle de ligne contrôle la distance entre les lignes de balayage — des valeurs plus
petites remplissent plus densément mais prennent plus de temps. Laisse-le à zéro pour utiliser la
taille du spot de ton laser, ce qui convient bien à la plupart des gravures.

La section **Étiquettes** contrôle les annotations gravées à côté de la grille. Les étiquettes sont
activées par défaut et sont gravées en premier, afin que le motif de test ne puisse pas les masquer.
Elles ont leur propre puissance (10% par défaut) et vitesse (1000 mm/min par défaut), et les valeurs
de vitesse sont affichées dans ton unité d'affichage préférée.

La section **Paramètres** contient les plages que la grille fait varier — vitesse, puissance, passes
ou décalage, selon le mode sélectionné. Les modes qui gardent un paramètre fixe (par exemple la
vitesse dans Puissance vs Passes) te permettent également de définir cette constante ici.

## Comprendre la Disposition

Dans le mode Puissance vs Vitesse par défaut, la puissance augmente de gauche à droite et la vitesse
de haut en bas :

```
                   Puissance (%)
                 10       55       100
Vitesse    100  [  ]     [  ]     [  ]
(mm/min)   300  [  ]     [  ]     [  ]
           500  [  ]     [  ]     [  ]
```

Les étiquettes sur les bords gauche et supérieur affichent la valeur exacte de chaque ligne et
colonne, donc tu n'as jamais besoin de compter les cellules.

La taille globale découle directement des dimensions de la grille : chaque axe mesure _pas × taille
de forme + (pas − 1) × espacement_, plus la place des étiquettes à gauche et en haut (15 mm au
maximum, et uniquement lorsque les étiquettes sont activées). Une grille 5×5 de carrés de 20 mm avec
un espacement de 5 mm fait 120 mm de côté sans étiquettes et 135 mm avec.

## Déroulement de la Grille {#how-the-grid-runs}

Les cellules ne s'exécutent délibérément **pas** dans l'ordre de lecture. Rayforge les exécute dans
un ordre optimisé pour le risque : la vitesse la plus élevée d'abord, la puissance la plus basse à
chaque vitesse, et le moins de passes à chaque puissance. Les combinaisons lentes et à haute
puissance sont celles qui risquent le plus de carboniser le matériau ou de déclencher un incendie,
donc elles s'exécutent en dernier. Cet ordre est intentionnel et ne peut pas être modifié.

## Exécution du Test

Charge le matériau que tu veux caractériser — du rebut, pas ta pièce finale — et fais la mise au
point du laser comme pour un travail réel, car la distance de mise au point change le résultat.
Lance le travail et reste près de la machine : si une cellule commence à carboniser gravement ou à
fumer excessivement, arrête le travail plutôt que de le laisser se terminer.

Lorsque le test est terminé, examine chaque cellule. Si la gravure est trop claire, va vers plus de
puissance ou une vitesse plus lente ; si elle est trop foncée ou brûlée, va vers moins de puissance
ou une vitesse plus élevée. Pour les tests de coupe, cherche la cellule qui coupe proprement avec le
moins de carbonisation. Pour cibler le point idéal, exécute une deuxième grille plus fine : si un
test grossier 5×5 a trouvé sa meilleure cellule autour de 40% de puissance et 4000 mm/min, une
grille de suivi couvrant 35-45% et 3000-5000 mm/min la localisera précisément.

<!-- prettier-ignore-start -->
:::tip[Enregistre-la comme recette]
Au lieu de garder un carnet des meilleurs réglages, stocke-les comme une
[recette](../../application-settings/recipes.md) : nomme-la (par exemple « Coupe Contreplaqué
3 mm »), lie-la à la machine, à l'opération, au matériau et à l'épaisseur testés, et Rayforge
suggérera exactement ces réglages la prochaine fois que tu coupes le même matériau.
:::
<!-- prettier-ignore-end -->

## Utilisation Avancée

Les grilles de test de matériau sont des pièces ordinaires, donc elles se combinent librement avec
d'autres opérations. Un schéma courant consiste à ajouter une opération de contour autour de la
grille terminée et à découper la pièce de test du matériau de stock une fois la gravure terminée.

Exécuter la même configuration de grille sur différents matériaux est un moyen rapide de constituer
une bibliothèque de réglages fiables — et les recettes rendent cette bibliothèque consultable par
matériau et épaisseur plus tard.

## Conseils & Bonnes Pratiques

Quelques habitudes rendent les résultats de test plus fiables :

- Commence à partir d'un préréglage et ajuste à partir de là plutôt que de tout configurer à partir
  de zéro.
- Donne de l'espace aux cellules : des carrés de 15-20 mm sont bien plus faciles à évaluer que des
  minuscules.
- Ne change qu'une variable à la fois lorsque tu affines — une grille fine qui fait varier les deux
  axes sur de larges plages est difficile à interpréter.
- Laisse le matériau refroidir entre des tests consécutifs sur la même pièce.
- Utilise la même distance de mise au point pour chaque test, y compris le travail final.

Et les règles de sécurité laser habituelles s'appliquent doublement aux grilles de test, qui
explorent intentionnellement un territoire inconnu :

- Ne laisse jamais un test en cours sans surveillance.
- Commence avec des plages de puissance prudentes et monte progressivement.
- Assure-toi que l'extraction des fumées fonctionne avant de commencer.
- Garde un extincteur à portée de main.

## Dépannage

**Les cellules s'exécutent dans un ordre étrange.** C'est l'ordre d'exécution optimisé pour le
risque décrit dans [Déroulement de la Grille](#how-the-grid-runs) — les combinaisons les plus
rapides et les plus faibles d'abord. C'est intentionnel.

**Les résultats sont incohérents entre les exécutions.** Assure-toi que le matériau repose à plat et
est fixé, que la mise au point est identique sur toute la grille, et que ton alimentation délivre
une puissance stable. Si une seule région de la grille semble incorrecte, le matériau lui-même est
peut-être irrégulier.

## Sujets Connexes

- **[Aperçu 3D](../../ui/3d-preview.md)** - Prévisualise l'exécution du test avant de le lancer
- **[Recettes](../../application-settings/recipes.md)** - Réutilise automatiquement tes résultats de
  test
- **[Gravure](engrave)** - Comprendre les opérations de gravure
- **[Coupe de Contour](contour)** - Comprendre les opérations de coupe
