# Guide de Positionnement de la Pièce

Ce guide couvre toutes les méthodes disponibles dans Rayforge pour positionner avec précision votre
pièce et aligner vos conceptions avant la découpe ou la gravure.

## Aperçu

Un positionnement précis de la pièce est essentiel pour :

- **Prévenir le gaspillage** : Éviter de couper au mauvais endroit
- **Alignement précis** : Positionner les conceptions sur des matériaux pré-imprimés
- **Résultats reproductibles** : Exécuter le même travail plusieurs fois de manière cohérente
- **Travaux multipièces** : Aligner plusieurs pièces sur une seule feuille

Rayforge fournit plusieurs outils complémentaires pour le positionnement :

| Méthode                  | Objectif                  | Idéal Pour                                        |
| ------------------------ | ------------------------- | ------------------------------------------------- |
| **Mode Focus**           | Voir la position du laser | Alignement visuel rapide                          |
| **Cadrage**              | Prévisualiser les limites | Vérifier que la conception tient sur le matériau  |
| **Zéro SCF**             | Définir l'origine         | Positionnement reproductible                      |
| **Surimposition Caméra** | Placement visuel          | Alignement précis sur caractéristiques existantes |

---

## Mode Focus (Pointeur Laser)

Le mode focus allume le laser à un faible niveau de puissance, agissant comme un "pointeur laser"
pour vous aider à voir exactement où la tête du laser est positionnée.

### Activer le Mode Focus

1. **Connecter à votre machine**
2. **Cliquer sur le bouton Focus** dans la barre d'outils (icône laser)
3. Le laser s'allume au niveau de puissance focus configuré
4. **Déplacer la tête du laser** pour voir la position du faisceau sur votre matériau
5. **Cliquer à nouveau sur le bouton Focus** pour éteindre une fois terminé

<!-- prettier-ignore-start -->
:::warning[Sécurité]
Même à faible puissance, le laser peut endommager les yeux. Ne regardez jamais
directement le faisceau et ne le pointez pas vers des surfaces réfléchissantes. Portez une
protection oculaire appropriée.
:::
<!-- prettier-ignore-end -->

### Configurer la Puissance Focus

La puissance focus détermine la luminosité du point laser :

1. Allez dans **Paramètres → Machine → Laser**
2. Trouvez le paramètre **Puissance Focus**
3. Définissez une valeur qui rend le point visible sans marquer votre matériau
   - Valeurs typiques : 1-5% pour la plupart des matériaux
   - Définissez sur 0 pour désactiver la fonction

<!-- prettier-ignore-start -->
:::tip[Trouver la Bonne Puissance]
Commencez avec 1% et augmentez progressivement. Le point doit être
visible mais ne laisser aucune marque sur votre matériau. Les matériaux plus foncés peuvent
nécessiter une puissance plus élevée pour voir le point clairement.
:::
<!-- prettier-ignore-end -->

### Quand Utiliser le Mode Focus

- **Vérifications rapides d'alignement** : Voir si le laser est approximativement où vous l'attendez
- **Trouver les bords du matériau** : Déplacer vers les coins pour vérifier le placement du matériau
- **Définir l'origine SCF** : Positionner le laser au point zéro souhaité avant de définir le SCF
- **Vérifier la position d'origine** : Vérifier que le référencement a fonctionné correctement

---

## Laser Pointeur avec Décalage

Certaines machines ont un laser pointeur dédié (un petit laser à point rouge) monté à une distance
fixe du faisceau de coupe. Lorsque vous alignez la pièce à l'aide du point du pointeur, le faisceau
de coupe atterrirait en réalité décalé de ce point — sauf si Rayforge compense.

Le réglage laser [Décalage du Pointeur](../machine/laser.md) vous permet de saisir cette distance et
d'activer ou de désactiver la compensation. Lorsqu'il est activé :

- **Définir l'origine à la position actuelle** (et Zéro X / Zéro Y) place l'origine de travail à la
  position du point du pointeur.
- Le canevas affiche un point jaune du pointeur à côté du point rouge du faisceau, indiquant où se
  trouve le point du pointeur sur votre matériau.
- Un interrupteur **Alignement du pointeur** apparaît dans le popover de déplacement (l'icône de
  boussole à côté de l'affichage de la position).

### Alignement du Pointeur

Tant que l'**alignement du pointeur** est actif, chaque opération de visée absolue — Déplacer vers,
les raccourcis de coin, l'aller à l'origine du SCF, Cliquer pour déplacer, Déplacer la tête ici et
le cadrage — fait se poser le _point du pointeur_ sur la position visée, ce qui vous permet de
positionner la pièce entièrement à l'aide du point visible. La saisie de coordonnées du popover est
pré-remplie avec la position du point du pointeur, et le point du pointeur du canevas est dessiné
plein lorsque l'alignement est actif et creux lorsqu'il est inactif.

Le flux de travail se combine proprement avec le zérage par le point du pointeur : l'origine se
situe là où le pointeur a marqué, la visée décale chaque cible du décalage, et la gravure n'est pas
décalée — aligner avec le point et couper avec le faisceau restent donc cohérents.

Lorsque vous appuyez sur **Envoyer** tandis que l'alignement est actif, un avertissement apparaît à
chaque envoi (pas d'option « ne plus demander ») : le job coupe avec le faisceau aux positions du
SCF et n'est jamais décalé. Vous pouvez choisir _Désactiver et graver_, _Graver quand même_ ou
annuler.

Les mouvements de jog et les jobs ne sont jamais décalés par l'interrupteur : le jog est relatif, et
la sortie G-code est identique que l'alignement soit activé ou non. L'interrupteur est propre à la
session — il n'est pas enregistré dans le profil de la machine et se réinitialise lorsque vous
changez de machine.

Le décalage du pointeur est particulièrement utile pour les pièces plus grandes que le plateau laser
(travail en passage continu), où vous alignez à plusieurs reprises le dessin sur une marque de
référence de la pièce en mouvement.

---

## Cadrage

Le cadrage trace le rectangle englobant de votre travail à faible (ou zéro) puissance, montrant
exactement où votre conception sera découpée ou gravée.

### Comment Cadrer

1. **Charger et positionner votre conception** dans Rayforge
2. **Cliquer sur Machine → Cadrer** ou appuyer sur `Ctrl+F`
3. La tête du laser trace le cadre englobant de votre travail
4. **Vérifier le contour** qu'il tient dans votre matériau

### Paramètres de Cadrage

Configurer le comportement du cadrage dans **Paramètres → Machine → Laser** :

- **Vitesse de Cadrage** : La vitesse à laquelle la tête se déplace pendant le cadrage (plus lent =
  plus facile à voir)
- **Puissance de Cadrage** : Puissance du laser pendant le cadrage
  - Définissez sur 0 pour le cadrage à l'air (laser éteint, mouvement uniquement)
  - Définissez sur 1-5% pour une trace visible sur le matériau

<!-- prettier-ignore-start -->
:::tip[Cadrage à l'Air vs. Faible Puissance]
- **Cadrage à l'air (0% puissance)** : Sûr pour tout matériau, mais vous ne voyez que le mouvement
  de la tête
- **Cadrage à faible puissance** : Laisse une marque visible faible, utile pour un alignement précis
  sur les matériaux foncés
:::
<!-- prettier-ignore-end -->

### Quand Cadrer

- **Avant chaque travail** : Vérification rapide que la conception tient
- **Après des changements de position** : Confirmer que le nouveau placement est correct
- **Matériaux coûteux** : Vérifier deux fois avant de s'engager
- **Travaux multipièces** : Vérifier que toutes les pièces tiennent sur le matériau

Voir [Cadrer Votre Travail](framing-your-job) pour plus de détails.

---

## Définir le Zéro SCF (Système de Coordonnées de Travail)

Les Systèmes de Coordonnées de Travail (SCF) vous permettent de définir des "points zéro"
personnalisés pour vos travaux. Cela facilite l'alignement des travaux avec la position de votre
matériau.

### Configuration Rapide SCF

1. **Déplacer la tête du laser** vers le coin de votre matériau (ou point d'origine souhaité)
2. **Ouvrir le Panneau de Contrôle** (`Ctrl+L`)
3. **Sélectionner un SCF** (G54 est le système de coordonnées de travail par défaut)
4. **Cliquer sur Zéro X et Zéro Y** pour définir la position actuelle comme origine
5. Le point (0,0) de votre conception sera maintenant aligné avec cette position

### Comprendre les Systèmes de Coordonnées

Rayforge utilise plusieurs systèmes de coordonnées :

| Système     | Description                                                |
| ----------- | ---------------------------------------------------------- |
| **G53**     | Coordonnées machine (fixes, ne peuvent pas être modifiées) |
| **G54**     | Système de coordonnées de travail 1 (par défaut)           |
| **G55-G59** | Systèmes de coordonnées de travail supplémentaires         |

<!-- prettier-ignore-start -->
:::tip[Zones de Travail Multiples]
Utilisez différents emplacements SCF pour différentes positions de
fixation. Par exemple :

- G54 pour le côté gauche de votre lit
- G55 pour le côté droit
- G56 pour un accessoire rotatif
:::
<!-- prettier-ignore-end -->

### Quand Définir le Zéro SCF

- **Nouveau placement de matériau** : Aligner l'origine au coin du matériau
- **Travail avec fixation** : Définir l'origine au point de référence de la fixation
- **Travaux reproductibles** : Même travail, différentes positions
- **Lots de production** : Positionnement cohérent à travers plusieurs pièces

Voir [Systèmes de Coordonnées de Travail](../general-info/coordinate-systems.md) pour la
documentation complète.

---

## Positionnement par Caméra

La surimposition de caméra montre une vue en direct de votre matériau avec votre conception
superposée, permettant un alignement visuel précis.

### Configurer la Caméra

1. **Connecter une caméra USB** au-dessus de votre zone de travail
2. Allez dans **Paramètres → Caméra** et ajoutez votre dispositif de caméra
3. **Activer la caméra** pour voir la surimposition sur votre canevas
4. **Aligner la caméra** en utilisant la procédure d'alignement (requis pour un positionnement
   précis)

### Alignement de la Caméra

L'alignement de la caméra mappe les pixels de la caméra aux coordonnées du monde réel :

1. Ouvrir **Caméra → Aligner la Caméra**
2. Placer des marqueurs d'alignement à des positions connues (au moins 4 points)
3. Entrer les coordonnées X/Y du monde réel pour chaque point
4. Cliquer sur **Appliquer** pour calculer la transformation

<!-- prettier-ignore-start -->
:::tip[Précision de l'Alignement]
- Utilisez des points répartis sur toute votre zone de travail
- Mesurez les coordonnées du monde soigneusement avec une règle
- Utilisez les positions machine (déplacer vers des coordonnées connues) pour une meilleure
  précision
:::
<!-- prettier-ignore-end -->

### Positionnement avec Surimposition Caméra

1. **Activer la surimposition caméra** pour voir votre matériau
2. **Importer votre conception**
3. **Glisser la conception** pour l'aligner avec les caractéristiques visibles dans la caméra
4. **Ajustement fin** en utilisant les touches fléchées pour un placement parfait au pixel près
5. **Cadrer pour vérifier** avant d'exécuter le travail

### Quand Utiliser le Positionnement par Caméra

- **Matériaux pré-imprimés** : Aligner les coupes avec les impressions existantes
- **Matériaux irréguliers** : Positionner sur des pièces non rectangulaires
- **Placement précis** : Exigences de précision sub-millimétrique
- **Dispositions complexes** : Plusieurs éléments avec un espacement spécifique

Voir [Intégration Caméra](../machine/camera.md) pour la documentation complète.

---

## Flux de Travail Recommandés

### Flux de Travail de Positionnement de Base

Pour des travaux simples sur des matériaux rectangulaires :

1. **Placer le matériau** sur le lit du laser
2. **Activer le mode focus** et déplacer pour vérifier la position du matériau
3. **Définir le zéro SCF** au coin du matériau
4. **Positionner votre conception** sur le canevas
5. **Cadrer le travail** pour vérifier le placement
6. **Exécuter le travail**

### Flux de Travail d'Alignement de Précision

Pour un placement précis sur des matériaux pré-imprimés ou marqués :

1. **Configurer et aligner la caméra** (configuration unique)
2. **Placer le matériau** sur le lit du laser
3. **Activer la surimposition caméra** pour voir le matériau
4. **Importer et positionner la conception** visuellement sur l'image de la caméra
5. **Désactiver la caméra** et cadrer pour vérifier
6. **Exécuter le travail**

### Flux de Travail en Passage Continu

Pour les pièces plus longues que le plateau laser (alimentées à travers la machine en segments), le
**décalage du pointeur** supprime les calculs manuels :

1. **Coupez le segment 1** de votre dessin normalement.
2. **Faites avancer la pièce** dans le passage continu pour que le segment suivant soit sur le
   plateau.
3. **Déplacez la machine** jusqu'à ce que le point du pointeur marque un point de repère sur la
   pièce (par exemple un coin d'un élément déjà coupé).
4. **Définissez le zéro du SCF** (ou Zéro X / Zéro Y) — avec le décalage du pointeur activé,
   l'origine tombe exactement là où le pointeur a visé.
5. **Positionnez le segment suivant** du dessin par rapport à cette origine dans le canevas.
6. **Cadrez pour vérifier**, puis **lancez le job**.
7. Répétez à partir de l'étape 2 pour chaque segment restant.

### Flux de Travail de Production

Pour exécuter plusieurs travaux identiques :

1. **Configurer la fixation** sur le lit du laser
2. **Définir le zéro SCF** aligné avec la fixation (ex. G54)
3. **Charger et configurer** votre conception
4. **Cadrer pour vérifier** l'alignement avec la fixation
5. **Exécuter le travail**
6. **Remplacer le matériau** et répéter (le SCF reste le même)

### Flux de Travail Multi-Positions

Pour exécuter le même travail à différents emplacements :

1. **Configurer plusieurs positions SCF** :
   - Déplacer à la position 1, définir le zéro G54
   - Déplacer à la position 2, définir le zéro G55
   - Déplacer à la position 3, définir le zéro G56
2. **Charger votre conception** (même conception pour toutes les positions)
3. **Sélectionner G54**, cadrer et exécuter
4. **Sélectionner G55**, cadrer et exécuter
5. **Sélectionner G56**, cadrer et exécuter

---

## Dépannage

### Point laser non visible en mode focus

- **Augmenter la puissance focus** dans les paramètres du laser
- **Matériaux foncés** peuvent nécessiter une puissance plus élevée (5-10%)
- **Vérifier la connexion du laser** et s'assurer que la machine répond
- **Vérifier que la puissance focus** n'est pas définie sur 0

### Surimposition caméra désalignée

- **Réexécuter l'alignement de la caméra** avec plus de points de référence
- **Vérifier le montage de la caméra** - elle a peut-être bougé
- **Vérifier les coordonnées du monde** ont été mesurées avec précision
- **Voir le dépannage de la caméra** dans la documentation d'Intégration Caméra

---

## Sujets Liés

- [Cadrer Votre Travail](framing-your-job) - Documentation détaillée du cadrage
- [Systèmes de Coordonnées de Travail](../general-info/coordinate-systems.md) - Référence SCF
- [Intégration Caméra](../machine/camera.md) - Configuration et alignement de la caméra
- [Panneau de Contrôle](../ui/bottom-panel.md) - Contrôles de déplacement et gestion SCF
- [Guide de Démarrage Rapide](../getting-started/quick-start.md) - Flux de travail de base
