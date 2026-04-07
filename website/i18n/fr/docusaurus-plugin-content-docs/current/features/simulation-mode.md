# Mode Simulation

![Mode Simulation](/screenshots/main-simulation.png)

Le mode simulation montre comment ton travail laser sera exécuté avant de le lancer sur la machine. Tu peux parcourir le G-code pas à pas et voir exactement ce qui se passera.

## Activer le Mode Simulation

- **Clavier** : Appuie sur <kbd>F11</kbd>
- **Menu** : Va dans **Affichage → Simuler l'exécution**
- **Barre d'outils** : Clique sur le bouton de simulation

## Visualisation

### Carte de chaleur de vitesse

Les opérations sont colorées selon leur vitesse :

| Vitesse        | Couleur |
| -------------- | ------- |
| La plus lente  | Bleu    |
| Lente          | Cyan    |
| Moyenne        | Vert    |
| Rapide         | Jaune   |
| La plus rapide | Rouge   |

Les couleurs sont relatives à la plage de vitesse de ton travail - le bleu est le minimum, le rouge est le maximum.

### Transparence de puissance

L'opacité des lignes indique la puissance du laser :

- **Lignes faintes** = Faible puissance (déplacements, gravure légère)
- **Lignes solides** = Forte puissance (découpe)

## Contrôles de Lecture

Utilise les contrôles en bas du canevas :

- **Lecture/Pause** (<kbd>Espace</kbd>) : Démarrer ou arrêter la lecture automatique
- **Curseur de progression** : Glisser pour naviguer dans le travail
- **Touches fléchées** : Parcourir les instructions une par une

La simulation et la vue du G-code restent synchronisées — avancer dans la simulation met en surbrillance la
ligne de G-code correspondante, et cliquer sur une ligne de G-code fait sauter la simulation à ce point.

La vue 3D possède aussi sa propre simulation avec lecture synchronisée. Avancer dans le simulateur
3D met en surbrillance la ligne correspondante dans le visualiseur G-code, et vice versa.

## Modifier pendant la simulation

Tu peux modifier les pièces pendant la simulation. Déplace, mets à l'échelle ou fais pivoter des objets, et la simulation se met à jour automatiquement.

## Sujets Connexes

- **[Vue 3D](../ui/3d-preview)** - Visualisation du trajet d'outil 3D
- **[Grille de Test de Matériau](operations/material-test-grid)** - Utilise la simulation pour valider les tests
