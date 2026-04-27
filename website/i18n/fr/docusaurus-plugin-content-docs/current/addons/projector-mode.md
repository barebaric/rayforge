# Mode Projecteur

Le Mode Projecteur affiche votre zone de coupe dans une fenêtre séparée,
conçue pour être affichée sur un projecteur externe ou un moniteur secondaire.
Cela vous permet de voir exactement où le laser va couper en projetant les
trajectoires directement sur votre matériau, facilitant ainsi l'alignement.

La fenêtre du projecteur affiche vos workpieces en vert brillant sur un fond
noir. Elle montre le cadre d'étendue des axes de la machine et l'origine de
travail afin que vous puissiez voir toute la zone de coupe et où se trouve le
point d'origine. La vue se met à jour en temps réel lorsque vous déplacez ou
modifiez des workpieces sur le canevas principal.

## Ouvrir la fenêtre du projecteur

Ouvrez la fenêtre du projecteur depuis **Affichage - Afficher le dialogue du
projecteur**. La fenêtre s'ouvre comme une fenêtre séparée et indépendante que
vous pouvez déplacer vers n'importe quel écran connecté à votre système.

Un basculeur contrôle la fenêtre du projecteur — le même élément de menu la
ferme, et appuyer sur Échap lorsque la fenêtre du projecteur est active la
ferme également.

## Mode plein écran

Cliquez sur le bouton **Plein écran** dans la barre de titre de la fenêtre du
projecteur pour passer en mode plein écran. Cela masque les décorations de la
fenêtre et remplit tout l'écran. Cliquez sur **Quitter le plein écran** (le
même bouton) pour revenir au mode fenêtré.

Le plein écran est le mode prévu lors de la projection sur le matériau, car il
supprime les bordures de fenêtre distrayantes et utilise toute la surface de
l'écran.

## Opacité

Le bouton d'opacité dans la barre de titre parcourt quatre niveaux : 100 %,
80 %, 60 % et 40 %. Réduire l'opacité rend la fenêtre du projecteur
semi-transparente, ce qui peut être utile sur un moniteur de bureau pour voir
les fenêtres en arrière-plan. Chaque clic passe au niveau d'opacité suivant et
revient au début.

![Mode Projecteur](/screenshots/addon-projector-mode.png)

## Ce que le projecteur affiche

L'affichage du projecteur rend une vue simplifiée de votre document. Les
workpieces apparaissent sous forme de contours vert brillant montrant les
trajectoires calculées — les mêmes chemins qui seront envoyés au laser. Les
images de base de vos workpieces ne sont pas affichées, gardant l'affichage
concentré sur les trajectoires de coupe.

Le cadre d'étendue de la machine apparaît comme une bordure représentant toute
la zone de déplacement des axes de votre machine. Le réticule de l'origine de
travail indique où se trouve l'origine du système de coordonnées dans cette
zone. Les deux se mettent à jour automatiquement si vous modifiez le décalage
du système de coordonnées de travail sur votre machine.

## Sujets associés

- [Systèmes de coordonnées](../general-info/coordinate-systems) - Comprendre les coordonnées machine et les décalages de travail
- [Positionnement des workpieces](../features/workpiece-positioning) - Positionner les workpieces sur le canevas
