# Smart Stock

Smart Stock utilise la vision par ordinateur pour détecter le matériel placé sur
le lit de votre laser et créer les éléments de matériel correspondants dans votre
document. En comparant une image de référence du lit vide avec la vue actuelle de
la caméra, l'addon identifie les contours du matériel physique et génère des
éléments de matériel correctement positionnés avec la bonne forme et la bonne
taille.

## Prérequis

Vous avez besoin d'une caméra configurée et calibrée connectée à votre machine.
La caméra doit être configurée avec une correction de perspective pour que
l'image capturée s'aligne sur le système de coordonnées physiques de la machine.
Vous avez également besoin d'une machine configurée pour que l'addon connaisse
les dimensions de la zone de travail.

## Ouvrir la boîte de dialogue de détection

Ouvrez la boîte de dialogue depuis **Outils - Détecter le matériel depuis la
caméra**. La fenêtre affiche un aperçu en direct de la caméra à gauche et les
paramètres de détection à droite.

## Capturer une image de référence

Avant de détecter le matériel, vous avez besoin d'une image de référence du lit
du laser vide. Sans matériel sur le lit, cliquez sur le bouton **Capturer** à
côté de **Capturer la référence**. L'addon stocke cette image et la compare au
flux vidéo de la caméra pour détecter les nouveaux objets.

Les images de référence sont enregistrées par caméra. Lorsque vous rouvrez la
boîte de dialogue avec la même caméra, la référence précédemment capturée est
chargée automatiquement et la détection s'exécute immédiatement si du matériel
est déjà présent sur le lit.

## Détecter le matériel

Placez votre matériel sur le lit du laser, puis cliquez sur **Détecter le
matériel** en bas du panneau des paramètres. L'addon compare l'image actuelle de
la caméra avec l'image de référence et trace les contours de tout nouvel objet.
Les formes détectées apparaissent dans l'aperçu sous forme de contours magenta
avec un remplissage vert.

La ligne d'état en bas du panneau des paramètres indique le nombre d'éléments
trouvés. Si aucun matériel n'est détecté, ajustez le placement ou l'éclairage et
réessayez.

## Paramètres de détection

**Caméra** affiche la caméra actuellement sélectionnée. Cliquez sur **Modifier**
pour passer à une autre caméra configurée.

**Sensibilité** contrôle l'importance du changement visuel nécessaire pour être
enregistré comme matériel. Avec des valeurs plus élevées, des différences plus
faibles ou subtiles entre la référence et l'image actuelle sont détectées. Avec
des valeurs plus basses, seuls les changements importants sont pris en compte.
Si l'addon ne détecte pas du matériel présent, augmentez la sensibilité. S'il
détecte des ombres ou des reflets comme du matériel, diminuez-la.

**Lissage** contrôle la douceur des contours détectés. Des valeurs plus élevées
produisent des contours plus arrondis et plus simples en filtrant les petits
bords irréguliers de l'image de la caméra. Des valeurs plus basses préservent
davantage de détails de la forme réelle du matériel.

## Créer des éléments de matériel

Une fois que l'aperçu affiche les contours détectés correspondant à votre
matériel, cliquez sur **Créer des éléments de matériel** dans la barre de titre.
L'addon ajoute un actif de matériel et un élément de matériel à votre document
pour chaque forme détectée, positionnés aux coordonnées physiques correctes sur
le canevas. La boîte de dialogue se ferme après la création des éléments.

## Sujets associés

- [Configuration de la caméra](../machine/camera) - Configurer et calibrer votre caméra
- [Gestion du matériel](../features/stock-handling) - Travailler avec les éléments de matériel dans votre document
