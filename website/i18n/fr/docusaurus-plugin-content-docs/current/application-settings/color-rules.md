# Règles de couleur

Les règles de couleur vous permettent d'associer un type d'étape à une couleur spécifique afin que
la bonne opération soit choisie automatiquement lorsque vous importez un SVG, un PDF ou un autre
fichier vectoriel. Au lieu de créer manuellement des étapes pour chaque calque importé, Rayforge lit
la couleur de chaque forme et applique la règle correspondante.

## Fonctionnement

Lorsque vous importez un fichier vectoriel, Rayforge peut regrouper les formes entrantes par
couleur. Chaque couleur distincte devient un calque. Si une règle de couleur existe pour cette
couleur, le calque se voit attribuer automatiquement le type d'étape de la règle. Les couleurs sans
règle reçoivent le comportement par défaut (Contour pour les contours, plus Gravure si les formes
ont des remplissages).

Une fois le type d'étape attribué, le système normal de [correspondance de recettes](recipes)
s'applique par-dessus — les règles de couleur déterminent donc _quelle_ opération est exécutée, et
les recettes déterminent _comment_ elle est exécutée (puissance, vitesse, passes, etc.).

## Créer des règles de couleur

### 1. Ouvrir la page des règles de couleur

Menu : **Édition → Paramètres**, puis sélectionnez **Règles de couleur** dans la barre latérale.

### 2. Ajouter une règle

Cliquez sur **Ajouter une règle de couleur** pour ouvrir la boîte de dialogue de l'éditeur :

- **Couleur** — Choisissez la couleur SVG qui doit déclencher cette règle. Utilisez le sélecteur de
  couleur pour correspondre à la couleur de contour ou de remplissage de votre logiciel de
  conception.
- **Étiquette** _(optionnel)_ — Un nom convivial affiché dans la liste des règles (ex. « Couper le
  rouge », « Graver le bleu »). Si laissé vide, la valeur hexadécimale est utilisée.
- **Type d'étape** — L'opération à créer lorsque cette couleur est importée. Tout type d'étape
  enregistré est disponible, y compris ceux fournis par les [addons](addons) (ex. Shrink Wrap,
  Material Test Grid).

### 3. Enregistrer

Cliquez sur **Ajouter** pour enregistrer la règle. Elle prend effet immédiatement à la prochaine
importation. Les règles sont stockées dans votre configuration utilisateur et persistent entre les
sessions.

<!-- prettier-ignore-start -->
:::tip[Correspondance exacte des couleurs]
Les règles de couleur correspondent par valeur
hexadécimale exacte. Lorsque vous choisissez une couleur dans votre logiciel de conception
(Inkscape, Illustrator, etc.), notez le code hexadécimal exact et saisissez la même valeur dans
Rayforge. Par exemple, `#e34c4c` dans votre SVG doit être `#e34c4c` dans la règle — même une
différence d'un seul chiffre empêchera la correspondance.
:::
<!-- prettier-ignore-end -->

## Gérer les règles

Chaque règle de la liste affiche un échantillon de couleur, l'étiquette, le type d'étape et des
boutons de modification/suppression.

- **Modifier** — Changez la couleur, l'étiquette ou le type d'étape. Changer la couleur d'une règle
  existante la remplace (l'ancienne couleur est supprimée).
- **Supprimer** — Supprime définitivement la règle.
- **Types d'étape indisponibles** — Si l'addon du type d'étape a été désinstallé, une icône
  d'avertissement apparaît à côté de la règle. La règle est conservée afin que vous puissiez la
  corriger ou réinstaller l'addon. Lors de l'importation, les calques correspondant à une règle dont
  le type d'étape est indisponible reviennent au comportement par défaut.

## Comportement d'importation

### Regroupement automatique par couleur

Lorsque des règles de couleur existent, la boîte de dialogue d'importation bascule automatiquement
sur **Couleurs** comme source de calques pour les fichiers contenant des couleurs distinctes. Cela
garantit que chaque couleur devient son propre calque afin que les règles puissent s'appliquer. Vous
pouvez toujours revenir à **Calques SVG** ou à d'autres sources dans la boîte de dialogue si vous
préférez.

### Qu'est-ce qui déclenche une règle

Une règle de couleur s'applique lorsque :

1. Le fichier est importé avec **Couleurs** comme source de calques.
2. La couleur de contour ou de remplissage d'une forme correspond exactement à la couleur de la
   règle.
3. Le type d'étape de la règle est actuellement enregistré.

Les règles ne s'appliquent **pas** aux fichiers importés avec les sources de calques **Calques SVG**
ou **Aplatir**, car ces sources ne regroupent pas par couleur.

## Exemple de flux de travail

Une configuration courante pour les designs SVG multicolores :

1. **Dans votre logiciel de conception**, attribuez des couleurs distinctes à différentes opérations
   :
   - Rouge (`#ff0000`) pour les contours de découpe
   - Bleu (`#0000ff`) pour la gravure
   - Vert (`#00ff00`) pour le marquage

2. **Dans Rayforge**, créez trois règles de couleur :
   - `#ff0000` → Contour
   - `#0000ff` → Gravure
   - `#00ff00` → Contour (avec des paramètres de recette différents)

3. **Importez le SVG.** La boîte de dialogue d'importation sélectionne automatiquement Couleurs, et
   chaque groupe de couleurs reçoit son type d'étape automatiquement.

4. **Ajustez** avec les [recettes](recipes) pour définir la puissance, la vitesse et d'autres
   paramètres par type d'étape.

## Règles de couleur et recettes

Les règles de couleur et les recettes sont complémentaires :

| Fonctionnalité    | Ce qu'elle définit                      | Quand elle s'applique          |
| ----------------- | --------------------------------------- | ------------------------------ |
| Règles de couleur | Type d'étape (Contour, etc.)            | Lors de l'importation          |
| Recettes          | Paramètres de l'étape (puissance, etc.) | Lors de la création de l'étape |

Une configuration typique consiste à utiliser les règles de couleur pour choisir l'opération et les
recettes pour configurer les paramètres. Par exemple, une règle de couleur rouge correspond à
Contour, et une recette ciblant le type d'étape Contour sur votre matériau actuel applique la bonne
vitesse de coupe et la bonne puissance.

---

**Sujets connexes** :

- [Recettes](recipes) - Appliquer des préréglages de puissance, vitesse et paramètres
- [Importer des fichiers](../files/importing.md) - Options d'importation SVG et vectorielle
- [Flux de travail multi-calques](../features/multi-layer.md) - Organisation des calques
- [Opérations](../features/operations/contour.md) - Référence des types d'étape
