# Mode Simulation

![Mode Simulation](/screenshots/main-simulation.png)

Le mode simulation montre comment votre travail laser sera exécuté avant de le lancer sur la machine. Vous pouvez parcourir le code G pas à pas et voir exactement ce qui se passera.

## Activer le Mode Simulation

- **Clavier**: Appuyez sur <kbd>F11</kbd>
- **Menu**: Allez dans **Affichage → Simuler l'exécution**
- **Barre d'outils**: Cliquez sur le bouton de simulation

## Visualisation

### Carte de Chaleur de Vitesse

Les opérations sont colorées selon leur vitesse :

| Vitesse        | Couleur |
| -------------- | ------- |
| La plus lente  | Bleu    |
| Lente          | Cyan    |
| Moyenne        | Vert    |
| Rapide         | Jaune   |
| La plus rapide | Rouge   |

Les couleurs sont relatives à la plage de vitesse de votre travail - le bleu est le minimum, le rouge est le maximum.

### Transparence de Puissance

L'opacité des lignes indique la puissance du laser :

- **Lignes faintes** = Faible puissance (déplacements, gravure légère)
- **Lignes solides** = Forte puissance (découpe)

## Contrôles de Lecture

Utilisez les contrôles en bas du canevas :

- **Lecture/Pause** (<kbd>Espace</kbd>): Démarrer ou arrêter la lecture automatique
- **Curseur de progression**: Glissez pour naviguer dans le travail
- **Touches fléchées**: Parcourir les instructions une par une

La simulation et la vue du code G restent synchronisées - parcourir la simulation met en surbrillance le code G correspondant, et cliquer sur les lignes de code G saute à ce point dans la simulation.

## Modifier Pendant la Simulation

Vous pouvez modifier les pièces pendant la simulation. Déplacez, mettez à l'échelle ou faites pivoter des objets, et la simulation se met à jour automatiquement.

## Sujets Connexes

- **[Aperçu 3D](../ui/3d-preview)** - Visualisation du parcours d'outil 3D
- **[Grille de Test de Matériau](operations/material-test-grid)** - Utilisez la simulation pour valider les tests
