# Front d'Onde

Le défrichage adaptatif par front d'onde remplit les formes vectorielles
fermées avec des trajectoires d'outil concentriques qui s'étendent vers
l'extérieur depuis le centre de la poche comme des ondulations dans un
étang. Les anneaux en expansion gèrent automatiquement les îles
intérieures et produisent des trajectoires lisses et continues sans les
inversions brusques du balayage raster.

## Aperçu

Contrairement à la gravure raster traditionnelle, qui balaie d'avant en
arrière en lignes parallèles, le front d'onde génère des passes
concentriques qui rayonnent depuis le centre de chaque poche. Cela
produit une finition uniforme, semblable à des ondulations, bien adaptée
aux applications où le motif de remplissage lui-même contribue au
résultat visuel.

Les opérations de front d'onde :

- Remplissent les formes vectorielles fermées (poches) avec des passes
  concentriques
- S'étendent vers l'extérieur depuis le centre de la poche
- Contournent automatiquement les îles intérieures (trous dans la poche)
- Produisent des trajectoires lisses sans inversions de direction

## Quand Utiliser le Front d'Onde

Le front d'onde est un motif de remplissage alternatif pour les zones
de poche. Ses anneaux concentriques peuvent être visuellement plus
attrayants que les lignes raster parallèles, et le motif en expansion
complète naturellement les formes circulaires ou organiques.

Utilisez le défrichage adaptatif par front d'onde pour :

- Remplir des poches dans les conceptions vectorielles
- Fabrication de tampons et matrices — le front d'onde défriche la
  poche d'arrière-plan tout en préservant les caractéristiques en relief
  comme îles intérieures
- Applications où la texture de remplissage est visible dans la pièce
  finie

**N'utilisez pas le front d'onde pour :**

- Couper le long des contours (utilisez [Contour](contour) à la place)
- Remplir des images bitmap (utilisez [Gravure](engrave) à la place)
- Sections de paroi mince où aucune poche n'existe

## Créer une Opération de Front d'Onde

### Étape 1 : Sélectionner les Objets

1. Importez ou dessinez des formes vectorielles fermées sur le canevas
2. Sélectionnez les objets définissant la limite de la poche
3. Assurez-vous que les formes sont des parcours fermés

### Étape 2 : Ajouter une Opération de Front d'Onde

- **Menu :** Opérations → Ajouter Front d'Onde
- **Clic droit :** Menu contextuel → Ajouter Opération → Front d'Onde

### Étape 3 : Configurer les Paramètres

Ajustez le pas et le décalage en fonction de votre matériau et de la
finition souhaitée.

![Résultat de l'opération front d'onde](/screenshots/operations-wavefront.png)

## Paramètres Clés

### Pas (Step Over)

La distance entre les passes consécutives du front d'onde (mm). Des
valeurs plus petites donnent une couverture plus dense avec plus de
passes et des temps de travail plus longs. Des valeurs plus grandes
espacent davantage les passes pour une exécution plus rapide.

**Le Pas est défini par défaut sur la taille du spot laser** et a une
plage de 0,05–50,0 mm.

| Pas     | Densité de ligne         | Temps de travail |
| ------- | ------------------------ | ---------------- |
| 0,1 mm  | Dense, nombreuses lignes | Le plus lent     |
| 0,3 mm  | Modérée                  | Moyen            |
| 1,0 mm+ | Éparse, moins de lignes  | Rapide           |

Les valeurs typiques sont de 0,1–0,5 mm pour la plupart des applications.

### Décalage (Offset)

Dégagement supplémentaire par rapport à la paroi de la poche (mm). Crée
une marge entre la passe de front d'onde la plus externe et le contour
de la limite. Ceci est utile lorsqu'une passe de [Contour](contour)
séparée finira le bord, ou lorsque vous souhaitez laisser une bordure
délibérée autour de la poche.

Plage : 0,0–20,0 mm. La valeur par défaut est 0,0 (les passes de front
d'onde s'étendent jusqu'à la limite).

## Comment Fonctionne le Front d'Onde

1. **Passe d'entrée** — Une entrée hélicoïdale plonge dans le centre de
   la poche pour établir une zone défrichée initiale
2. **Expansion du front d'onde** — En partant du centre défriché, les
   anneaux concentriques s'étendent vers l'extérieur. Chaque anneau
   s'étend au-delà du précédent de la distance de pas configurée
3. **Gestion des îles** — À mesure que le front d'onde grandit, il
   rencontre et contourne toutes les îles intérieures, les laissant
   debout
4. **Achèvement** — L'expansion se poursuit jusqu'à ce que toute la
   zone de la poche soit couverte

## Post-Traitement

Les opérations de front d'onde supportent :

- **[Lissage de Parcours](../smooth)** — Réduire les bords irréguliers
  dans les trajectoires d'outil
- **[Optimisation de Parcours](../path-optimization)** — Minimiser la
  distance de déplacement entre les passes

## Conseils & Meilleures Pratiques

### Choix du Pas

- Une couverture plus dense (petit pas) signifie plus de passes et des
  temps de travail plus longs
- Une couverture éparse (grand pas) est plus rapide mais laisse plus de
  matériau entre les passes
- Équilibrez la densité et le temps de travail pour votre application

### Fabrication de Tampons et Matrices

Le front d'onde est bien adapté à la fabrication de tampons. Les anneaux
concentriques en expansion défrichent naturellement la poche
d'arrière-plan tout en naviguant autour des caractéristiques en relief
traitées comme des îles intérieures.

### Combinaison avec le Contour

Un flux de travail courant consiste à défricher l'intérieur de la poche
avec le front d'onde, puis à finir la limite avec une passe de
[Contour](contour) pour un bord net. Réglez le décalage pour laisser
suffisamment de marge pour la coupe de contour.

## Sujets Connexes

- **[Contour](contour)** — Coupe le long des contours vectoriels
- **[Gravure](engrave)** — Remplissage de zones avec des motifs de
  gravure raster
- **[Enveloppe Rétractable](shrink-wrap)** — Coupe de limite autour des
  objets
- **[Lissage de Parcours](../smooth)** — Affinage des bords de
  trajectoire d'outil
