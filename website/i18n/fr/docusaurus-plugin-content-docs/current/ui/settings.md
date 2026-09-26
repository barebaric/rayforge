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

La page Gestes de la souris te permet de réattribuer les gestes de navigation du canevas 2D, du
canevas 3D et de l'éditeur de croquis. Chaque entrée affiche la combinaison de boutons de souris
actuellement attribuée. Clique sur une entrée pour capturer une nouvelle attribution : appuie sur le
bouton de souris souhaité (avec ou sans touches modificatrices) ou utilise la molette de la souris,
et l'attribution est appliquée immédiatement.

- **Déplacer la vue** — maintiens le bouton de souris attribué enfoncé et déplace la souris pour
  déplacer la vue.
- **Zoomer la vue** — utilise la molette de la souris pour zoomer.
- **Orbite / rotation (canevas 3D)** — fais glisser pour orbiter autour de la scène ou tourner
  autour de l'axe Z.
- **Ouvrir le menu contextuel** — le menu contextuel du canevas 2D ou le menu d'outils de l'éditeur
  de croquis.
- **Réinitialiser la vue** — replace la vue. Non attribué par défaut.

Une attribution peut être supprimée avec **Désattribuer** ou restaurée avec **Rétablir les valeurs
par défaut**. L'attribution d'un geste déjà utilisé pour une autre action dans la même vue est
refusée, afin qu'une combinaison de boutons de souris ne déclenche jamais deux actions à la fois.
Les addons peuvent apporter des configurations de gestes supplémentaires, qui apparaissent comme des
sections supplémentaires sur cette page.

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
