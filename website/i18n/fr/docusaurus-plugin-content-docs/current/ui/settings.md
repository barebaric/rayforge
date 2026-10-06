# Paramètres

![Paramètres généraux](/screenshots/app-settings-general.webp)

Personnalise Rayforge pour correspondre à ton flux de travail et tes préférences. Ouvre les
paramètres via **Édition → Paramètres** ou appuie sur <kbd>ctrl+virgule</kbd>.

## Général

La page Général contient les paramètres à l'échelle de l'application.

### Apparence

Choisis entre le thème **Système**, **Clair** ou **Sombre** pour correspondre à ton environnement de
bureau ou ta préférence personnelle. Tu peux aussi configurer les **Couleurs d'opération** pour
utiliser soit la couleur du laser, soit la couleur du calque comme distinction visuelle sur le
canevas.

### Unités

Configure les unités d'affichage utilisées dans l'application. Tu peux définir des unités séparées
pour la **longueur** (millimètres, pouces, etc.), la **vitesse** (mm/min, mm/sec, pouces/min, etc.)
et l'**accélération** (mm/s², etc.).

### Comportement

Par défaut, les opérations sont recalculées automatiquement après chaque modification. Si tu
travailles sur une machine plus lente ou avec des documents très complexes, tu peux désactiver la
**Mise à jour automatique des opérations** et déclencher le recalcul manuellement via le bouton de
la barre d'outils.

Rayforge peut **Vérifier les mises à jour** automatiquement au démarrage. Lorsqu'activé, tu seras
informé lorsqu'une nouvelle version est disponible.

Tu peux aussi configurer le **Comportement au démarrage** — démarrer avec un espace de travail vide,
rouvrir le dernier projet ou toujours ouvrir un fichier de projet spécifique. Note que les fichiers
spécifiés en ligne de commande remplaceront toujours ces paramètres.

### Confidentialité

Rayforge peut envoyer des données d'utilisation anonymes pour aider à améliorer l'application.
Aucune information personnelle n'est collectée. Tu peux activer ou désactiver **Rapporter
l'utilisation anonyme** à tout moment. Consulte la page
[suivi d'utilisation](https://rayforge.org/docs/general-info/usage-tracking) pour en savoir plus sur
les données collectées et leur utilisation.

## Gestes de la souris

![Réglages des gestes de la souris](/screenshots/app-settings-gestures.webp)

La page Gestes de la souris te permet de réattribuer les gestes de navigation du canevas 2D et du
canevas 3D ; l'éditeur de croquis utilise les mêmes gestes que le canevas 2D. Chaque ligne propose
un menu déroulant avec les combinaisons de boutons de souris disponibles :

- **Canevas 2D** — _Déplacer la vue_ (par défaut, glisser avec le bouton du milieu ; sur le bouton
  gauche, le glissement est réservé à la sélection et n'est donc pas proposé).
- **Canevas 3D** — _Orbiter la caméra_ (glisser avec le bouton du milieu), _Déplacer la caméra_
  (Maj + bouton du milieu) et _Tourner autour de l'axe Z_ (glisser avec le bouton gauche).

Le zoom (molette de la souris), le menu contextuel (clic droit) et la réinitialisation de la vue (la
touche `1` sur le canevas 2D) sont fixes et ne peuvent pas être modifiés. Un simple clic sur un
bouton attribué conserve son action fixe : lorsque le déplacement est attribué au bouton droit de la
souris, glisser avec le bouton droit déplace la vue et un simple clic droit ouvre toujours le menu
contextuel. Choisir **Non attribué** désactive un geste, et les combinaisons déjà utilisées par une
autre action du même canevas ne sont pas proposées. Les addons peuvent apporter des configurations
de gestes supplémentaires, qui apparaissent comme des sections supplémentaires sur cette page.

## Autres paramètres

La boîte de dialogue des paramètres inclut également des pages pour gérer d'autres parties de
l'application. Chacune possède sa propre documentation :

- [Machines](../application-settings/machines.md) — ajouter, supprimer et configurer tes découpeurs
  laser
- [Matériaux](../application-settings/materials.md) — gérer tes bibliothèques de matériaux
- [Recettes](../application-settings/recipes.md) — gérer les recettes d'opérations enregistrées
- [Règles de couleur](../application-settings/color-rules.md) — faire correspondre les couleurs SVG
  aux types d'étape
- [Fournisseurs IA](../application-settings/ai-provider.md) — configurer les fournisseurs IA pour
  les addons
- [Addons](../application-settings/addons.md) — installer, mettre à jour et supprimer les addons
  d'extension
