# Print & Cut

Alignez les découpes laser sur du matériel pré-imprimé en enregistrant des
points de référence sur votre design et en les associant à leurs positions
physiques sur le matériau. Ceci est utile pour découper des autocollants, des
étiquettes ou tout élément devant correspondre à une impression existante.

## Prérequis

L'addon nécessite une machine configurée. Votre machine doit être connectée
pour l'étape de déplacement. Vous devez également avoir sélectionné un
workpiece ou un groupe sur le canevas.

## Ouvrir l'assistant

Sélectionnez un seul workpiece ou groupe sur le canevas, puis ouvrez
**Outils - Aligner à la position physique**. L'assistant s'ouvre sous forme
de dialogue en trois étapes avec un aperçu de votre workpiece à gauche et
les commandes à droite.

## Étape 1 : Sélectionner les points de design

![Sélectionner les points de design](/screenshots/addon-print-and-cut-pick.png)

Le panneau de gauche affiche un rendu de votre workpiece sélectionné. Cliquez
directement sur l'image pour placer le premier point d'alignement, marqué en
vert, puis cliquez à nouveau pour placer le deuxième point, marqué en bleu.
Une ligne en pointillés relie les deux points.

Choisissez deux points correspondant à des repères identifiables sur votre
matériau physique — par exemple, des marques de repérage imprimées ou des
coins distincts. Les points doivent être suffisamment éloignés pour un
alignement précis. Vous pouvez faire glisser l'un ou l'autre point après
placement pour ajuster la position.

Utilisez la molette de défilement pour zoomer dans l'aperçu et le clic
central avec glissement pour vous déplacer. Le bouton **Réinitialiser** en
bas efface les deux points et vous permet de recommencer.

Une fois les deux points placés, cliquez sur **Suivant** pour continuer.

## Étape 2 : Enregistrer les positions physiques

![Enregistrer les positions physiques](/screenshots/addon-print-and-cut-jog.png)

Sur cette page, vous déplacez le laser vers les positions physiques
correspondant aux deux points de design que vous avez sélectionnés. Le panneau
de droite affiche un pavé directionnel pour le déplacement et un contrôle de
distance qui définit la distance parcourue par le laser à chaque étape.

Déplacez le laser vers la position physique correspondant à votre premier
point de design, puis cliquez sur **Enregistrer** à côté de Position 1. Les
coordonnées enregistrées apparaissent dans la ligne. Répétez le processus
pour Position 2. Vous pouvez revenir à une position enregistrée à tout moment
en cliquant sur le bouton **Aller à** à côté de celle-ci.

Le basculeur **Laser de mise au point** allume le laser à la puissance de mise
au point configurée, créant un point visible sur le matériau pour vous aider
à localiser les positions avec précision. Ce basculeur nécessite une valeur de
puissance de mise au point supérieure à zéro dans vos paramètres laser.

La position actuelle du laser est affichée en bas du panneau. Lorsque les deux
positions sont enregistrées, cliquez sur **Suivant** pour continuer.

## Étape 3 : Vérifier et appliquer la transformation

![Vérifier et appliquer la transformation](/screenshots/addon-print-and-cut-apply.png)

La dernière page affiche l'alignement calculé sous forme de décalage de
translation et d'angle de rotation. Ces valeurs sont dérivées de la différence
entre vos points de design et les positions physiques enregistrées.

Par défaut, la mise à l'échelle est verrouillée à 1.0. Si votre matériau
physique diffère en taille du design — par exemple en raison de la mise à
l'échelle de l'imprimante — activez le basculeur **Autoriser la mise à
l'échelle**. Le facteur d'échelle est alors calculé à partir du rapport entre
la distance physique et la distance de design entre vos deux points. Une note
apparaît lorsque la mise à l'échelle est verrouillée mais que les distances ne
correspondent pas, indiquant que le deuxième point peut ne pas être aligné
exactement.

Cliquez sur **Appliquer** pour déplacer et faire pivoter le workpiece sur le
canevas afin qu'il corresponde aux positions physiques. La transformation est
appliquée comme une action annulable.

## Sujets associés

- [Positionnement des workpieces](../features/workpiece-positioning) - Positionner et transformer les workpieces manuellement
- [Paramètres du laser](../machine/laser) - Configurer la puissance de mise au point du laser
