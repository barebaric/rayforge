# Remplissage par Décalage

Le remplissage par décalage remplit les formes vectorielles fermées avec des anneaux parallèles à
leur contour. Le premier anneau suit le contour lui-même ; chaque anneau suivant se trouve un
intervalle de ligne plus à l'intérieur, jusqu'à ce que la forme soit remplie. Les trous d'une forme
restent libres : les anneaux suivent aussi leurs bords.

## Aperçu

Là où la [Gravure](engrave) balaie d'avant en arrière en lignes droites et où le
[Front d'Onde](wavefront) s'étend depuis le centre de la poche, le remplissage par décalage
travaille du contour vers l'intérieur. Chaque anneau est un chemin fermé coupé le long de sa ligne
médiane, de sorte que le remplissage épouse la forme du dessin.

Les opérations de remplissage par décalage :

- Remplissent les formes vectorielles fermées avec des anneaux parallèles au contour
- Laissent libres les trous à l'intérieur d'une forme
- Ignorent les chemins ouverts
- Coupent les anneaux de l'extérieur vers l'intérieur ou de l'intérieur vers l'extérieur

## Quand Utiliser le Remplissage par Décalage

Utilisez le remplissage par décalage pour :

- Remplir du texte, des logos et d'autres formes vectorielles avec un remplissage qui suit leurs
  contours
- Évider des poches où les anneaux doivent épouser le bord de la forme
- Les dessins où le motif de remplissage est visible sur la pièce finie

**N'utilisez pas le remplissage par décalage pour :**

- Couper uniquement le long des contours (utilisez plutôt [Contour](contour))
- Remplir des images bitmap (utilisez plutôt [Gravure](engrave))
- Les chemins ouverts, qui n'entourent aucune zone à remplir

## Créer une Opération de Remplissage par Décalage

### Étape 1 : Préparer les Formes

1. Importez ou dessinez des formes vectorielles fermées sur le canevas
2. Assurez-vous que les formes sont des chemins fermés ; les chemins ouverts sont ignorés

### Étape 2 : Ajouter une Étape de Remplissage par Décalage

1. Ouvrez le flux de travail du calque dans le panneau de droite.
2. Cliquez sur le bouton **Ajouter une étape** et choisissez **Remplissage par Décalage**.

### Étape 3 : Configurer les Paramètres

Ajustez l'intervalle de ligne, le décalage et la direction de remplissage selon votre matériau et la
finition souhaitée.

## Paramètres Clés

La boîte de dialogue des paramètres de l'étape comporte trois onglets : **Paramètres de l'étape**,
**Laser** et **Post-traitement**. Les paramètres sont décrits ci-dessous dans l'ordre des onglets.

### Paramètres de Remplissage par Décalage

Le groupe **Remplissage par Décalage** de l'onglet _Paramètres de l'étape_ contrôle le motif de
remplissage.

#### Intervalle de Ligne (Line Interval)

La distance entre deux anneaux de remplissage voisins. Des valeurs plus petites donnent un
remplissage plus dense avec plus d'anneaux et des travaux plus longs ; des valeurs plus grandes sont
plus rapides mais laissent plus d'espace entre les anneaux.

**L'intervalle de ligne vaut par défaut la taille du point laser** et va de 0,01 à 50,0 mm.

#### Décalage (Offset)

La distance du premier anneau à l'intérieur du contour. Avec la valeur par défaut de 0,0, le premier
anneau se trouve sur le contour ; une valeur plus grande laisse une marge entre le contour et le
remplissage. C'est utile lorsqu'une passe de [Contour](contour) séparée finira le bord.

Plage : 0,0–20,0 mm. La valeur par défaut est 0,0.

#### Direction de Remplissage (Fill Direction)

L'ordre dans lequel les anneaux sont coupés :

- **Extérieur-Intérieur** (par défaut) : l'anneau le plus extérieur est coupé en premier et le
  remplissage progresse vers le centre
- **Intérieur-Extérieur** : l'anneau le plus intérieur est coupé en premier et l'anneau du contour
  en dernier

### Paramètres Laser

La puissance, la vitesse et la sélection de la tête laser se trouvent sur la page **Laser** de la
boîte de dialogue des paramètres de l'étape. Une nouvelle étape de remplissage par décalage démarre
à 80 % de puissance.

**Puissance (%) :**

- Intensité du laser pour le remplissage
- Adaptez-la aux exigences de votre matériau

**Vitesse (mm/min) :**

- Vitesse de déplacement du laser
- Adaptez-la à la vitesse de gravure de votre matériau

## Comment Fonctionne le Remplissage par Décalage

1. **Formes fermées** — Les chemins ouverts sont écartés ; chaque forme fermée est traitée avec ses
   trous
2. **Premier anneau** — Le contour, déplacé vers l'intérieur du décalage configuré
3. **Anneaux suivants** — Chaque anneau suivant se trouve un intervalle de ligne à l'intérieur du
   précédent ; les anneaux autour des trous grandissent en même temps vers l'extérieur à partir des
   trous
4. **Achèvement** — Des anneaux sont ajoutés jusqu'à ce que la forme soit remplie
5. **Coupe** — Les anneaux sont coupés le long de leur ligne médiane dans la direction de
   remplissage choisie

## Post-Traitement

Les opérations de remplissage par décalage prennent en charge :

- **[Lissage de Parcours](../smooth.md)** — Réduire les bords irréguliers dans les trajectoires
  (désactivé par défaut)
- **[Rognage au Stock](../crop-to-stock.md)** — Limiter les trajectoires à la limite du matériau
  (désactivé par défaut)
- **[Optimisation de Parcours](../path-optimization.md)** — Minimiser la distance de déplacement
  entre les anneaux
- **[Passe Multiple](../multi-pass.md)** — Répéter le remplissage plusieurs fois

## Conseils & Meilleures Pratiques

### Choix de l'Intervalle de Ligne

- Partez de la valeur par défaut (la taille du point laser) pour un remplissage fermé et régulier
- Augmentez l'intervalle pour un motif d'anneaux visible ou un travail plus rapide

### Combinaison avec le Contour

Évidez l'intérieur d'une forme avec le remplissage par décalage, puis finissez le bord avec une
passe de [Contour](contour). Réglez le décalage pour que le remplissage laisse assez de place à la
coupe de contour.

## Sujets Connexes

- **[Front d'Onde](wavefront)** — Remplissage concentrique s'étendant depuis le centre de la poche
- **[Contour](contour)** — Coupe le long des contours vectoriels
- **[Gravure](engrave)** — Remplissage de zones avec des motifs de gravure raster
- **[Lissage de Parcours](../smooth.md)** — Affinage des bords de trajectoire d'outil
