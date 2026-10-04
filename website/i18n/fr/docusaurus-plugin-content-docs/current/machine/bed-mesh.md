---
description:
  "Sondez la surface de travail de votre laser sur une grille et compensez automatiquement un plan
  de travail irrégulier. Garde le point focal sur le matériau, même sur les grands plateaux."
---

# Maillage du plateau

Les grandes surfaces de travail sont rarement parfaitement planes. Lorsque la hauteur du plateau
varie plus que la profondeur de champ de votre laser, les découpes deviennent incohérentes sur
l'ensemble du matériau. La fonction de maillage du plateau sonde la hauteur de la surface sur une
grille et compense l'axe Z du parcours pour que le point de focalisation suive la surface réelle.

![Maillage du plateau](/screenshots/machine-settings-bed-mesh.webp)

## Prérequis

Le maillage du plateau nécessite une machine avec un axe Z et un pilote prenant en charge la sonde
(GRBL, Marlin, Smoothie et OctoPrint actuellement). La page **Maillage du plateau** n'apparaît dans
**Paramètres → Machine** que si les deux conditions sont remplies. Elle se trouve après la page
Appareil.

## Sonder le plateau

Ouvrez **Paramètres → Machine** et accédez à la page **Maillage du plateau**.

1. **Grille de sonde** : Définissez la zone à sonder (origine X/Y, largeur, hauteur) ainsi que la
   densité de la grille (colonnes et lignes). La page affiche le nombre de points de sonde et estime
   la durée. Les grilles plus denses suivent la surface plus précisément, mais prennent plus de
   temps.
2. **Sondage** : Configurez la vitesse d'avance de la sonde, la distance maximale dont la tête peut
   descendre pour chercher la surface à chaque point (Course maximale) et la hauteur Z de sécurité
   utilisée entre les points.
3. Cliquez sur **Démarrer le sondage**. La machine parcourt chaque point de la grille en motif
   serpentin et touche la surface à chacun ; la vue 3D se remplit en direct au fil des résultats.
   Vous pouvez arrêter l'exécution à tout moment ; le maillage n'est enregistré qu'une fois la
   grille complète terminée.

Le maillage est enregistré avec le profil de la machine et affiché sous forme de surface 3D colorée
: les zones bleues sont plus basses, les rouges plus hautes. Utilisez le bouton **Supprimer le
maillage** pour le retirer si vous ne souhaitez plus de compensation de hauteur.

:::note Pendant le sondage, la tête se déplace sur toute la zone de la grille. Dégagez le plateau de
tout objet susceptible de gêner la sonde et assurez-vous que votre pointe de sonde (ou le réticule
du laser) peut atteindre la surface en chaque point de la grille. :::

---

## Pages associées

- [Paramètres matériels](hardware) - Dimensions de la machine et configuration des axes
- [Paramètres de l'appareil](device) - Connexion et options du contrôleur
- [Vue 3D](../ui/3d-preview.md) - Visualisation 3D du parcours
