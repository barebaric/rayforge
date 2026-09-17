# Paramètres Laser

La page Laser dans les Paramètres Machine configure tes têtes laser et leurs propriétés.

![Paramètres Laser](/screenshots/machine-settings-laser.webp)

## Têtes Laser

Rayforge supporte les machines avec plusieurs têtes laser. Chaque tête laser a sa propre
configuration.

### Ajouter une Tête Laser

Clique sur le bouton **Ajouter Laser** pour créer une nouvelle configuration de tête laser.

### Propriétés de la Tête Laser

Chaque tête laser a les paramètres suivants :

#### Nom

Un nom descriptif pour cette tête laser.

Exemples :

- "Diode 10W"
- "Tube CO2"
- "Laser Infrarouge"

#### Numéro d'Outil

L'index d'outil pour cette tête laser. Utilisé dans le G-code avec la commande T.

- Machines mono-tête : Utilise 0
- Machines multi-têtes : Assigne des numéros uniques (0, 1, 2, etc.)

#### Puissance Maximum

La valeur de puissance maximum pour ton laser.

- **GRBL typique** : 1000 (plage S0-S1000)
- **Certains contrôleurs** : 255 (plage S0-S255)
- **Mode pourcentage** : 100 (plage S0-S100)

Cette valeur doit correspondre au paramètre $30 de ton firmware.

#### Puissance de Cadrage

Le niveau de puissance utilisé pour les opérations de cadrage (traçage du contour sans couper).

- Définis à 0 pour désactiver le cadrage
- Ajuste selon ton laser et matériau

#### Vitesse de Cadrage

La vitesse à laquelle la tête laser se déplace pendant le cadrage. Elle est définie par tête laser,
ainsi si ta machine possède plusieurs lasers aux caractéristiques différentes, tu peux choisir une
vitesse appropriée pour chacun. Des vitesses plus lentes rendent le trajet de cadrage plus facile à
suivre visuellement.

#### Puissance de Focus

Le niveau de puissance utilisé lorsque le mode focus est activé. Le mode focus allume le laser à
faible puissance pour agir comme un « pointeur laser » pour le positionnement.

- Définis à 0 pour désactiver la fonction de mode focus
- Utilise pour l'alignement visuel et le positionnement

<!-- prettier-ignore-start -->
:::tip[Utiliser le Mode Focus]
Clique sur le bouton focus (icône laser) dans la barre d'outils pour
activer le mode focus. Le laser s'allumera à ce niveau de puissance, t'aidant à voir exactement où
le laser est positionné. Consulte [Positionnement de la Pièce](../features/workpiece-positioning.md)
pour plus d'informations.
:::
<!-- prettier-ignore-end -->

#### Taille du Spot

La taille physique de ton faisceau laser focalisé en millimètres.

- Entre les dimensions X et Y
- La plupart des lasers ont un spot circulaire (ex : 0.1 x 0.1)
- Affecte les calculs de qualité de gravure

<!-- prettier-ignore-start -->
:::tip[Mesurer la Taille du Spot]
Pour mesurer la taille de ton spot :

1. Tire une impulsion courte à faible puissance sur un matériau de test
2. Mesure la marque résultante avec un pied à coulisse
3. Utilise la moyenne de plusieurs mesures
:::
<!-- prettier-ignore-end -->

#### Couleur

La couleur utilisée pour afficher les opérations de ce laser (coupes et gravure) dans le canevas et
la prévisualisation 3D. Cela t'aide à distinguer visuellement quel laser effectuera chaque opération
lorsque tu travailles avec plusieurs têtes laser.

- Clique sur l'échantillon de couleur pour ouvrir un sélecteur de couleur
- Choisis une couleur qui contraste bien avec l'aperçu de ton matériau
- Les couleurs par défaut sont attribuées automatiquement

<!-- prettier-ignore-start -->
:::tip[Workflows Multi-Laser]
Lors de l'utilisation de plusieurs têtes laser, l'attribution de
couleurs différentes à chaque laser facilite la visualisation des opérations effectuées par chaque
laser. Par exemple, utilise le rouge pour ton laser de coupe principal et le bleu pour un laser de
gravure secondaire.
:::
<!-- prettier-ignore-end -->

#### Type de Laser

Choisis le type de tête laser dans le menu déroulant :

- **Diode** : Lasers diode standards (les plus courants pour les machines de loisir)
- **CO2** : Lasers à tube CO2
- **Fiber** : Lasers fibrés

Lorsque CO2 ou Fiber est sélectionné, des **paramètres PWM** supplémentaires apparaissent (voir
ci-dessous). Pour les lasers diode, la section PWM est masquée car elle ne s'applique pas.

Le type de laser définit également une **longueur d'onde** par défaut (utilisée par le modèle de
brûlure physique) lorsqu'aucune valeur explicite n'est saisie ci-dessous.

#### Longueur d'onde (nm)

La longueur d'onde d'émission de ton laser, en nanomètres. Alimente le
[modèle de brûlure physique](../ui/3d-preview.md#physical-burn-model) dans l'aperçu 3D : avec les
données d'[absorption](../application-settings/materials.md#absorption) du matériau, elle détermine
quelle partie de l'énergie laser le brut absorbe.

Lorsque définie sur 0, Rayforge utilise la longueur d'onde typique pour le type de laser sélectionné
(par ex. 445 nm pour diode, 1064 nm pour fibré, 10600 nm pour CO2).

#### Puissance Optique Max (W)

La puissance de sortie optique de ton laser à pleine puissance, en watts. Il s'agit de la sortie
lumineuse réelle, pas de l'entrée électrique. Avec la taille du spot et la vitesse de balayage, elle
détermine la fluence (J/cm²) utilisée par le
[modèle de brûlure physique](../ui/3d-preview.md#physical-burn-model).

Lorsque définie sur 0, une valeur par défaut de milieu de gamme de bureau est utilisée.

#### Paramètres PWM

Lorsqu'un type de laser CO2 ou Fiber est sélectionné, les contrôles PWM suivants apparaissent :

- **Fréquence PWM** : La fréquence PWM par défaut en Hz pour cette tête laser. Les valeurs typiques
  vont de 500 Hz à plusieurs kHz selon ton contrôleur et ton alimentation.
- **Fréquence PWM max** : La limite supérieure du réglage de fréquence. Cela empêche d'entrer des
  valeurs que ton matériel ne peut pas gérer.
- **Largeur d'impulsion** : La largeur d'impulsion par défaut en microsecondes. Cela contrôle la
  durée d'activation de chaque impulsion pendant un cycle.
- **Largeur d'impulsion min/max** : Les limites pour le réglage de la largeur d'impulsion.

Ces valeurs par défaut sont transmises à tes étapes d'opération, où elles peuvent être remplacées
par étape si nécessaire.

#### Décalage du Pointeur

Si votre machine dispose d'un laser pointeur séparé (un petit laser à point rouge) monté à une
distance fixe du faisceau de coupe, vous pouvez indiquer cette distance à Rayforge pour qu'il la
compense.

- **Utiliser le décalage du pointeur** : active la compensation. Désactivé par défaut.
- **Décalage du pointeur X / Y** : la distance en millimètres entre le point du faisceau de coupe et
  le point du pointeur, le long des axes X et Y de la machine.

Lorsqu'il est activé, trois choses changent :

1. **Définir l'origine à la position actuelle** (et les boutons Zéro X / Zéro Y) place l'origine de
   travail à l'endroit où le _point du pointeur_ marque la pièce, pas là où se trouve le faisceau de
   coupe (invisible).
2. Le canevas affiche un point jaune du pointeur à côté du point rouge du faisceau, indiquant où se
   trouve le point du pointeur sur votre matériau.
3. Un interrupteur **Alignement du pointeur** devient disponible dans le popover de déplacement
   (voir ci-dessous).

#### Alignement du Pointeur

L'alignement du pointeur est un interrupteur de session dans le popover de déplacement (l'icône de
boussole à côté de l'affichage de la position). Lorsqu'il est actif, toutes les opérations de visée
absolues — Déplacer vers, les raccourcis de coin, l'aller à l'origine du SCF, Cliquer pour déplacer,
Déplacer la tête ici et le cadrage — sont décalées pour que le _point du pointeur_ se pose sur la
position visée. Le point du pointeur sur le canevas est dessiné plein lorsque l'alignement est actif
et creux lorsqu'il est inactif.

Le flux de travail typique :

1. Déplacez la machine jusqu'à ce que le point du pointeur marque votre point de repère sur la
   pièce.
2. **Définissez l'origine de travail** là — avec le décalage du pointeur activé, l'origine tombe
   exactement là où le pointeur a visé.
3. Activez **Alignement du pointeur** dans le popover de déplacement.
4. Cadrez et déplacez avec le point du pointeur : tout ce que vous visez est marqué par le pointeur.
5. Lorsque vous appuyez sur **Envoyer**, un avertissement vous rappelle que le job brûle avec le
   faisceau aux positions du SCF — vous pouvez désactiver l'alignement et graver, graver quand même
   ou annuler.

Deux choses ne sont jamais décalées : le **jog** (un mouvement relatif ne nécessite aucune
compensation) et les **jobs** — la coupe se fait toujours avec le faisceau aux positions du SCF,
votre sortie G-code est donc identique que l'alignement soit activé ou non. Combiné au zérage par le
point du pointeur, tout reste cohérent : l'origine se situe à `faisceau + décalage`, la visée décale
chaque cible de `-décalage`, et la gravure n'est pas décalée.

L'alignement du pointeur est un réglage de session : il n'est pas enregistré dans le profil de la
machine et se réinitialise lorsque vous changez de machine.

<!-- prettier-ignore-start -->
:::tip[Mesurer le Décalage]
1. Déplacez la machine jusqu'à ce que le point du pointeur marque un point
   visible sur la pièce.
2. Activez le mode focus et déplacez jusqu'à ce que le *faisceau de coupe* brûle
   une marque exactement au même endroit (ou déplacez prudemment le faisceau à
   la puissance focus).
3. Le décalage est la position du pointeur moins celle du faisceau. Par exemple,
   si le pointeur a marqué X=100 et que vous avez dû déplacer le faisceau à
   X=88 pour toucher le même endroit, saisissez X = 12.0 — le point du pointeur
   se trouve 12 mm devant le faisceau.

Si une coupe de test sort décalée, inversez le signe de l'axe concerné.
:::
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
:::note[Mode Rotatif]
Lorsque l'accessoire rotatif est actif, l'axe Y est remplacé par le rouleau
rotatif, de sorte que la composante Y du décalage du pointeur ne s'applique pas
de manière significative. Mettez-la à 0 pour les jobs rotatifs.
:::
<!-- prettier-ignore-end -->

#### Modèle 3D

Chaque tête laser peut avoir un modèle 3D attribué. Ce modèle est affiché dans la
[vue 3D](../ui/3d-preview.md) et suit le trajet d'outil pendant la simulation.

Clique sur la ligne de sélection du modèle pour parcourir les modèles disponibles. Une fois un
modèle sélectionné, tu peux ajuster son échelle, sa rotation (X/Y/Z) et sa distance focale pour
correspondre à ta tête laser physique.

## Voir Aussi

- [Paramètres de l'Appareil](device) - Paramètres du mode laser GRBL
- [Positionnement de la Pièce](../features/workpiece-positioning.md) - Utilisation du mode focus et
  autres méthodes de positionnement
