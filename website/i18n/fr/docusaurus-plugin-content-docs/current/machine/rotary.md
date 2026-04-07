# Configuration de l'axe rotatif

Rayforge prend en charge les accessoires rotatifs pour la gravure et la découpe
d'objets cylindriques comme les gobelets, verres, stylos et matériaux ronds.
Lorsqu'un module rotatif est connecté, Rayforge enroule le travail autour du
cylindre et affiche un aperçu 3D du résultat.

![Paramètres du module rotatif](/screenshots/machine-rotary-module.png)

## Quand utiliser le mode rotatif

Utilise le mode rotatif chaque fois que ta pièce est cylindrique. Voici des
exemples courants :

- Graver des logos ou du texte sur des articles de boisson
- Découper des motifs sur des tubes ou tuyaux
- Marquer des objets cylindriques comme des stylos ou des manches d'outils

Sans le mode rotatif, l'axe Y déplace la tête laser d'avant en arrière sur un
lit plat. Avec le mode rotatif activé, l'axe Y contrôle la rotation du cylindre
à la place, de sorte que le dessin s'enroule autour de la surface.

## Configurer un module rotatif

Avant de commencer, fixe physiquement ton module rotatif à la machine selon
les instructions du fabricant. En général, cela signifie le brancher sur le port
du pilote de moteur pas à pas de l'axe Y à la place du moteur Y normal.

Dans Rayforge, ouvre **Paramètres → Machine** et accède à la page **Rotatif**
pour configurer ton module :

- **Circonférence** : Mesure la distance autour de l'objet que tu souhaites
  graver. Tu peux enrouler un morceau de papier ou de ficelle autour du
  cylindre et mesurer sa longueur. Cela indique à Rayforge la taille de la surface
  cylindrique pour que le dessin soit mis à l'échelle correctement.
- **Micropas par rotation** : Il s'agit du nombre de pas nécessaires au moteur
  rotatif pour une rotation complète. Consulte la documentation de ton module
  rotatif pour trouver cette valeur.

## Mode rotatif par couche

Si ton document comporte plusieurs couches, tu peux activer ou désactiver
le mode rotatif indépendamment pour chaque couche. C'est utile lorsque tu
souhaites combiner un travail plat et cylindrique dans un seul projet, ou lorsque
tu as des paramètres rotatifs différents pour différentes parties du travail.

Lorsque le mode rotatif est actif sur une couche, une petite icône rotative
apparaît à côté de cette couche dans la liste des couches, pour que tu puisses
voir d'un coup d'œil quelles couches s'exécuteront en mode rotatif.

## Aperçu 3D en mode rotatif

Lorsque le mode rotatif est actif, la [vue 3D](../ui/3d-preview) affiche ton
parcours d'outil enroulé autour d'un cylindre au lieu d'une surface plane.

![Aperçu 3D en mode rotatif](/screenshots/main-3d-rotary.png)

Cela te donne un aperçu réaliste de l'apparence du dessin sur l'objet réel,
facilitant la détection des problèmes de taille ou de placement avant de commencer
la découpe.

### Modèle 3D du module rotatif

Chaque module rotatif peut avoir un modèle 3D attribué. Le modèle apparaît dans
la [vue 3D](../ui/3d-preview) à côté de ton parcours d'outil, te donnant une
meilleure idée de l'apparence de l'installation physique. Tu peux ajuster l'échelle,
la position et la rotation du modèle pour correspondre à ton matériel réel.

### Avance en Z

Le mode rotatif prend en charge l'avance en Z, te permettant de couper
progressivement plus profondément sur des pièces cylindriques sur plusieurs passes.

## Conseils pour de bons résultats

- **Mesure la circonférence avec soin** — même une petite erreur ici étirera
  ou comprimera ton dessin autour du cylindre.
- **Fixe la pièce solidement** — assure-toi que l'objet est bien positionné
  sur les rouleaux et ne vacille ni ne glisse pendant le travail.
- **Teste d'abord à faible puissance** — effectue une passe de gravure légère
  pour vérifier l'alignement avant de t'engager dans une découpe à pleine
  puissance.
- **Garde la surface propre** — la poussière ou les résidus sur le cylindre
  peuvent affecter la qualité de la gravure.

## Pages associées

- [Flux de travail multicouche](../features/multi-layer) - Paramètres par couche incluant le rotatif
- [Vue 3D](../ui/3d-preview) - Aperçu des parcours d'outil en 3D
- [Paramètres machine](general) - Configuration générale de la machine
