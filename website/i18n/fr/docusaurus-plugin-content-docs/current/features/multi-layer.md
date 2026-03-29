# Flux de travail multi-couches

Le système multi-couches de Rayforge vous permet d'organiser des travaux
complexes en étapes de traitement séparées, chacune avec ses propres opérations
et paramètres. Ceci est essentiel pour combiner différents processus comme la
gravure et la découpe, ou pour travailler avec plusieurs matériaux.

## Que sont les couches ?

Une **couche** dans Rayforge est :

- **Un conteneur** pour les pièces (formes importées, images, texte)
- **Un flux de travail** définissant comment ces pièces sont traitées
- **Une étape** traitée séquentiellement pendant les travaux

**Concept clé :** Les couches sont traitées dans l'ordre, l'une après l'autre,
vous permettant de contrôler la séquence des opérations.

:::note Couches et pièces
Une couche contient une ou plusieurs pièces. Lors de l'importation de fichiers
SVG avec des couches, chaque couche de votre conception devient une couche
séparée dans Rayforge. Cela vous permet de garder votre conception organisée
exactement comme vous l'avez créée.
:::

---

## Pourquoi utiliser plusieurs couches ?

### Cas d'utilisation courants

**1. Graver puis découper**

Le flux de travail multi-couches le plus courant :

- **Couche 1 :** Gravure tramée du motif
- **Couche 2 :** Découpe de contour du profil

**Pourquoi des couches séparées ?**

- La gravure en premier garantit que la pièce ne bouge pas pendant la gravure
- La découpe en dernier empêche les pièces de tomber avant la fin de la gravure
- Différents réglages de puissance/vitesse pour chaque opération

**2. Découpe en passes multiples**

Pour les matériaux épais :

- **Couche 1 :** Première passe à puissance modérée
- **Couche 2 :** Deuxième passe à pleine puissance (même géométrie)
- **Couche 3 :** Troisième passe optionnelle si nécessaire

**Avantages :**

- Réduit la carbonisation par rapport à une seule passe à haute puissance
- Chaque couche peut avoir différents réglages vitesse/puissance

**3. Projets multi-matériaux**

Différents matériaux dans un seul travail :

- **Couche 1 :** Découper les pièces en acrylique
- **Couche 2 :** Graver les pièces en bois
- **Couche 3 :** Marquer les pièces en métal

**Exigences :**

- Chaque couche cible différentes zones du plateau
- Différentes vitesse/puissance/mise au point pour chaque matériau

**4. Importation de couches SVG**

Importer des fichiers SVG avec une structure de couches existante :

- **Couche 1 :** Éléments de gravure du SVG
- **Couche 2 :** Éléments de découpe du SVG
- **Couche 3 :** Éléments de marquage du SVG

**Flux de travail :**

- Importer un fichier SVG contenant des couches
- Activer « Utiliser les vecteurs originaux » dans la boîte de dialogue
  d'importation
- Sélectionner les couches à importer dans la liste des couches détectées
- Chaque couche devient une couche séparée dans Rayforge

**Exigences :**

- Votre fichier SVG doit utiliser des couches (créées dans Inkscape ou logiciel
  similaire)
- Activer « Utiliser les vecteurs originaux » lors de l'importation
- Les noms des couches sont préservés depuis votre logiciel de conception

---

## Créer et gérer les couches

### Ajouter une nouvelle couche

1. **Cliquez sur le bouton « + »** dans le panneau Couches
2. **Nommez la couche** de manière descriptive (ex. : « Couche gravure »,
   « Couche découpe »)
3. **La couche apparaît** dans la liste des couches

**Par défaut :** Les nouveaux documents commencent avec une couche.

### Propriétés des couches

Chaque couche possède :

| Propriété           | Description                                          |
| ------------------- | ---------------------------------------------------- |
| **Nom**             | Le nom affiché dans la liste des couches             |
| **Visible**         | Basculer la visibilité sur le canevas et l'aperçu    |
| **Flux de travail** | Les opérations appliquées aux pièces de cette couche |
| **Rotatif**         | Indique si cette couche s'exécute en mode rotatif    |
| **Pièces**          | Les formes/images contenues dans cette couche        |

### Mode rotatif par couche

Si vous avez un [accessoire rotatif](../machine/rotary) configuré, vous pouvez
activer le mode rotatif pour des couches individuelles. Cela vous permet de
combiner un travail sur surface plane et un travail cylindrique dans le même
projet — par exemple, graver un motif sur le couvercle plat d'une boîte sur une
couche et enrouler du texte autour du corps cylindrique sur une autre.

Les couches avec le mode rotatif actif affichent une petite icône rotative dans
la liste des couches. Chaque couche mémorise son propre réglage rotatif, vous
permettant de les mélanger librement.

:::note Couches comme conteneurs
Les couches sont des conteneurs pour vos pièces. Lors de l'importation de
fichiers SVG avec des couches, chaque couche de votre conception devient une
couche séparée dans Rayforge.
:::

### Réorganiser les couches

**L'ordre d'exécution = l'ordre des couches dans la liste (de haut en bas)**

Pour réorganiser :

1. **Glissez-déposez** les couches dans le panneau Couches
2. **L'ordre compte** - les couches s'exécutent de haut en bas

**Exemple :**

```
Panneau Couches :
1. Couche de gravure     S'exécute en premier
2. Couche de marquage    S'exécute en deuxième
3. Couche de découpe     S'exécute en dernier (recommandé)
```

### Supprimer des couches

1. **Sélectionnez la couche** dans le panneau Couches
2. **Cliquez sur le bouton de suppression** ou appuyez sur Supprimer
3. **Confirmez la suppression** (toutes les pièces de la couche sont
   supprimées)

:::warning La suppression est définitive
Supprimer une couche retire toutes ses pièces et ses paramètres de flux de
travail. Utilisez Annuler en cas de suppression accidentelle.
:::

---

## Assigner des pièces aux couches

### Assignation manuelle

1. **Importez ou créez** une pièce
2. **Glissez la pièce** vers la couche souhaitée dans le panneau Couches
3. **Ou utilisez le panneau de propriétés** pour changer la couche de la pièce

### Importation de couches SVG

Lors de l'importation de fichiers SVG avec « Utiliser les vecteurs originaux »
activé :

1. **Activez « Utiliser les vecteurs originaux »** dans la boîte de dialogue
   d'importation
2. **Rayforge détecte les couches** de votre fichier SVG
3. **Sélectionnez les couches** à importer avec les interrupteurs de couche
4. **Chaque couche sélectionnée** devient une couche séparée avec sa propre
   pièce

:::note Détection des couches
Rayforge détecte automatiquement les couches de votre fichier SVG. Chaque couche
que vous avez créée dans votre logiciel de conception apparaîtra comme une couche
séparée dans Rayforge.
:::

:::note Importation vectorielle uniquement
La sélection de couches n'est disponible que lors de l'utilisation de
l'importation directe de vecteurs. En mode tracé, le SVG entier est traité
comme une seule pièce.
:::

### Déplacer des pièces entre les couches

**Glisser-déposer :**

- Sélectionnez la ou les pièces sur le canevas ou le panneau Document
- Glissez vers la couche cible dans le panneau Couches

**Couper et coller :**

- Coupez la pièce de la couche actuelle (Ctrl+X)
- Sélectionnez la couche cible
- Collez (Ctrl+V)

### Boîte de dialogue d'importation SVG

Lors de l'importation de fichiers SVG, la boîte de dialogue d'importation
fournit des options qui affectent la gestion des couches :

**Mode d'importation :**

- **Utiliser les vecteurs originaux :** Préserve vos tracés vectoriels et la
  structure des couches. Lorsqu'activé, une section « Couches » apparaît
  montrant toutes les couches de votre fichier.
- **Mode tracé :** Convertit le SVG en image bitmap et trace les contours. La
  sélection de couches est désactivée dans ce mode.

**Section Couches (importation vectorielle uniquement) :**

- Affiche toutes les couches de votre fichier SVG
- Chaque couche possède un interrupteur pour activer/désactiver l'importation
- Les noms des couches de votre logiciel de conception sont préservés
- Seules les couches sélectionnées sont importées comme couches séparées

:::tip Préparer les fichiers SVG pour l'importation de couches
Pour utiliser l'importation de couches SVG, créez votre conception avec des
couches dans un logiciel comme Inkscape. Utilisez le panneau Couches pour
organiser votre conception, et Rayforge préservera cette structure.
:::

---

## Flux de travail des couches

Chaque couche possède un **flux de travail** qui définit comment ses pièces sont
traitées.

### Configurer les flux de travail des couches

Pour chaque couche, vous choisissez un type d'opération et configurez ses
paramètres :

**Types d'opération :**

- **Contour** - Suit les contours (pour la découpe ou le marquage)
- **Gravure tramée** - Grave des images et remplit des zones
- **Gravure en profondeur** - Crée des gravures à profondeur variable

**Améliorations optionnelles :**

- **Ponts** - Petits ponts pour maintenir les pièces en place pendant la
  découpe
- **Overscan** - Prolonge les coupes au-delà de la forme pour des bords plus
  propres
- **Ajustement du kerf** - Compense la largeur de coupe du laser

### Configurations courantes de couches

**Couche de gravure :**

- Opération : Gravure tramée
- Paramètres : 300-500 DPI, vitesse modérée
- Généralement aucune option supplémentaire nécessaire

**Couche de découpe :**

- Opération : Découpe de contour
- Options : Ponts (pour maintenir les pièces), Overscan (pour des bords
  propres)
- Paramètres : Vitesse plus lente, puissance plus élevée

**Couche de marquage :**

- Opération : Contour (puissance légère, ne coupe pas à travers)
- Paramètres : Faible puissance, vitesse rapide
- Objectif : Lignes de pliage, lignes décoratives

---

## Visibilité des couches

Contrôlez quelles couches sont affichées sur le canevas et dans les aperçus :

### Visibilité sur le canevas

- **L'icône œil** dans le panneau Couches bascule la visibilité
- **Couches masquées :**
  - Non affichées sur le canevas 2D
  - Non affichées dans l'aperçu 3D
  - **Toujours incluses dans le G-code généré**

**Cas d'utilisation :**

- Masquer les couches de gravure complexes tout en positionnant les couches de
  découpe
- Dégager le canevas lors du travail sur des couches spécifiques
- Se concentrer sur une couche à la fois

### Visibilité vs. Activé

| État                     | Canevas | Aperçu | G-code |
| ------------------------ | ------- | ------ | ------ |
| **Visible et activé**    | Oui     | Oui    | Oui    |
| **Masqué et activé**     | Non     | Non    | Oui    |
| **Visible et désactivé** | Oui     | Oui    | Non    |
| **Masqué et désactivé**  | Non     | Non    | Non    |

:::note Désactiver les couches
:::

Pour exclure temporairement une couche des travaux sans la supprimer,
désactivez l'opération de la couche ou désactivez-la dans les paramètres de la
couche.

---

## Ordre d'exécution des couches

### Comment les couches sont traitées

Pendant l'exécution du travail, Rayforge traite chaque couche dans l'ordre de
haut en bas. Au sein de chaque couche, toutes les pièces sont traitées avant de
passer à la couche suivante.

### L'ordre compte

**Mauvais ordre :**

```
1. Couche de découpe
2. Couche de gravure
```

**Problème :** Les pièces découpées peuvent tomber ou bouger avant la gravure !

**Ordre correct :**

```
1. Couche de gravure
2. Couche de découpe
```

**Pourquoi :** La gravure se fait pendant que la pièce est encore solidaire,
ensuite la découpe la libère.

### Passes multiples

Pour les matériaux épais, créez plusieurs couches de découpe :

```
1. Couche de gravure
2. Couche de découpe (Passe 1) - 50% puissance
3. Couche de découpe (Passe 2) - 75% puissance
4. Couche de découpe (Passe 3) - 100% puissance
```

**Astuce :** Utilisez la même géométrie pour toutes les passes de découpe
(dupliquez la couche).

---

## Techniques avancées

### Regroupement des couches par matériau

Utilisez les couches pour organiser par matériau lors de travaux mixtes :

```
Matériau 1 (Acrylique 3 mm) :
  - Couche gravure acrylique
  - Couche découpe acrylique

Matériau 2 (Contreplaqué 3 mm) :
  - Couche gravure bois
  - Couche découpe bois
```

**Flux de travail :**

1. Traiter toutes les couches du Matériau 1
2. Changer les matériaux
3. Traiter toutes les couches du Matériau 2

**Alternative :** Utilisez des documents séparés pour différents matériaux.

### Pause entre les couches

Vous pouvez configurer Rayforge pour faire une pause entre les couches. Ceci est
utile lorsque vous devez :

- Changer de matériau en cours de travail
- Inspecter l'avancement avant de continuer
- Ajuster la mise au point pour différentes opérations

Pour configurer les pauses entre couches, utilisez la fonctionnalité de hooks
dans les paramètres de votre machine.

### Paramètres spécifiques par couche

Le flux de travail de chaque couche peut avoir des paramètres uniques :

| Couche   | Opération | Vitesse    | Puissance | Passes |
| -------- | --------- | ---------- | --------- | ------ |
| Gravure  | Tramée    | 300 mm/min | 20%       | 1      |
| Marquage | Contour   | 500 mm/min | 10%       | 1      |
| Découpe  | Contour   | 100 mm/min | 90%       | 2      |

---

## Bonnes pratiques

### Conventions de nommage

**Bons noms de couches :**

- « Gravure - Logo »
- « Découpe - Contour extérieur »
- « Marquage - Lignes de pliage »
- « Passe 1 - Découpe ébauche »
- « Passe 2 - Découpe finale »

**Mauvais noms de couches :**

- « Couche 1 », « Couche 2 » (non descriptifs)
- Descriptions longues (restez concis)

### Organisation des couches

1. **De haut en bas = ordre d'exécution**
2. **Gravure avant découpe** (règle générale)
3. **Regrouper les opérations connexes** (toute la découpe, toute la gravure)
4. **Utiliser la visibilité** pour se concentrer sur le travail en cours
5. **Supprimer les couches inutilisées** pour garder les projets propres

### Préparer les fichiers SVG pour l'importation de couches

**Pour de meilleurs résultats lors de l'importation de couches SVG :**

1. **Utilisez le panneau Couches** dans votre logiciel de conception pour
   organiser votre motif
2. **Attribuez des noms significatifs** à chaque couche (ex. : « Gravure »,
   « Découpe »)
3. **Gardez les couches à plat** - évitez de placer des couches dans d'autres
   couches
4. **Enregistrez votre fichier** et importez-le dans Rayforge
5. **Vérifiez la détection des couches** en consultant la boîte de dialogue
   d'importation

Rayforge fonctionne mieux avec les fichiers SVG créés dans Inkscape ou un
logiciel similaire de conception vectorielle qui prend en charge les couches.

### Performances

**Nombreuses couches :**

- Pas d'impact significatif sur les performances
- 10 à 20 couches sont courantes pour les travaux complexes
- Organisez logiquement, pas pour minimiser le nombre de couches

**Simplifiez si nécessaire :**

- Combinez les opérations similaires en une seule couche si possible
- Utilisez moins de gravures tramées (les plus gourmandes en ressources)

---

## Dépannage

### La couche ne génère pas de G-code

**Problème :** La couche apparaît dans le document mais pas dans le G-code
généré.

**Solutions :**

1. **Vérifiez que la couche contient des pièces** - Les couches vides sont
   ignorées
2. **Vérifiez que le flux de travail est configuré** - La couche nécessite une
   opération
3. **Vérifiez les paramètres de l'opération** - Puissance > 0, vitesse valide,
   etc.
4. **Vérifiez la visibilité des pièces** - Les pièces masquées peuvent ne pas
   être traitées
5. **Régénérez le G-code** - Faites un petit changement pour forcer la
   régénération

### Ordre incorrect des couches

**Problème :** Les opérations s'exécutent dans un ordre inattendu.

**Solution :** Réorganisez les couches dans le panneau Couches. Souvenez-vous :
haut = premier.

### Couches superposées dans l'aperçu

**Problème :** Plusieurs couches affichent un contenu superposé dans l'aperçu.

**Précision :** C'est normal si les couches partagent la même zone XY.

**Solutions :**

- Utilisez la visibilité des couches pour masquer temporairement les autres
  couches
- Vérifiez l'aperçu 3D pour voir la profondeur/l'ordre
- Vérifiez que c'est intentionnel (ex. : graver puis découper la même forme)

### Pièce dans la mauvaise couche

**Problème :** La pièce a été assignée à une couche incorrecte.

**Solution :** Glissez la pièce vers la bonne couche dans le panneau Couches ou
l'arborescence du Document.

### Couches SVG non détectées

**Problème :** Importation d'un fichier SVG mais aucune couche n'apparaît dans la
boîte de dialogue d'importation.

**Solutions :**

1. **Vérifiez la structure du SVG** - Ouvrez votre fichier dans Inkscape ou un
   logiciel similaire pour vérifier qu'il contient des couches
2. **Activez « Utiliser les vecteurs originaux »** - La sélection de couches
   n'est disponible que dans ce mode d'importation
3. **Vérifiez que votre conception comporte des couches** - Assurez-vous d'avoir
   créé des couches dans votre logiciel de conception, pas seulement des groupes
4. **Vérifiez les couches imbriquées** - Les couches à l'intérieur d'autres
   couches peuvent ne pas être détectées correctement
5. **Réenregistrez votre fichier** - Parfois, réenregistrer avec une version
   récente de votre logiciel de conception aide

### L'importation de couches SVG affiche un contenu incorrect

**Problème :** La couche importée affiche le contenu d'autres couches ou est
vide.

**Solutions :**

1. **Vérifiez la sélection des couches** - Vérifiez que les bonnes couches sont
   activées dans la boîte de dialogue d'importation
2. **Vérifiez votre conception** - Ouvrez le fichier original dans votre
   logiciel de conception pour confirmer que chaque couche contient le bon
   contenu
3. **Vérifiez les éléments partagés** - Les éléments apparaissant dans plusieurs
   couches peuvent causer de la confusion
4. **Essayez le mode tracé** - Utilisez le mode tracé comme alternative si
   l'importation vectorielle pose problème

---

## Pages associées

- [Opérations](./operations/contour) - Types d'opérations pour les flux de
  travail de couches
- [Mode simulation](./simulation-mode) - Aperçu de l'exécution multi-couches
- [Macros et Hooks](../machine/hooks-macros) - Hooks au niveau des couches pour
  l'automatisation
- [Aperçu 3D](../ui/3d-preview) - Visualiser l'empilement des couches
