# Guide de Démarrage Rapide

Maintenant que Rayforge est installé et votre machine configurée, lançons votre premier travail laser ! Ce guide vous accompagnera dans l'importation d'un design, la configuration des opérations et l'envoi du G-code à votre machine.

## Étape 1 : Importer un Design

Rayforge prend en charge divers formats de fichiers, notamment SVG, DXF, PDF et les images raster (JPEG, PNG, BMP).

1. **Cliquez** sur **Fichier → Ouvrir** ou appuyez sur <kbd>ctrl+o</kbd>
2. Naviguez vers votre fichier de design et sélectionnez-le
3. Le design apparaîtra sur le canevas

![Canevas avec design importé](/screenshots/main-standard.png)

:::tip Vous n'avez pas encore de design ?
Vous pouvez créer des formes simples en utilisant les outils du canevas ou télécharger des fichiers SVG gratuits depuis des sites comme [Flaticon](https://www.flaticon.com/) ou [SVG Repo](https://www.svgrepo.com/).
:::


## Étape 2 : Positionner Votre Design

Utilisez les outils du canevas pour positionner et ajuster votre design :

- **Panoramique** : Clic molette et glisser, ou maintenez <kbd>espace</kbd> et glissez
- **Zoom** : Molette de défilement, ou <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Déplacer** : Cliquez et glissez votre design
- **Rotation** : Sélectionnez le design et utilisez les poignées de rotation
- **Mise à l'échelle** : Sélectionnez le design et glissez les poignées d'angle

## Étape 3 : Assigner une Opération

Les opérations définissent comment Rayforge traitera votre design. Les opérations courantes incluent :

- **Contour** : Couper le long du contour des formes
- **Gravure Raster** : Remplir les formes avec des lignes aller-retour (pour la gravure)
- **Gravure en Profondeur** : Créer des effets de profondeur 3D à partir d'images

### Ajouter une Opération

1. Sélectionnez votre design sur le canevas
2. Cliquez sur **Opérations → Ajouter une Opération** ou appuyez sur <kbd>ctrl+shift+a</kbd>
3. Choisissez le type d'opération (ex : "Contour" pour la coupe)
4. Configurez les paramètres de l'opération :
   - **Puissance** : Pourcentage de puissance laser (commencez bas et testez !)
   - **Vitesse** : Vitesse de déplacement en mm/min
   - **Passes** : Nombre de fois pour répéter l'opération (utile pour couper des matériaux épais)

![Paramètres d'Opération](/screenshots/step-settings-contour-general.png)

:::warning Commencez avec une Faible Puissance
Lorsque vous travaillez avec de nouveaux matériaux, commencez toujours avec des paramètres de puissance plus bas et effectuez des tests de coupe. Augmentez progressivement la puissance jusqu'à obtenir le résultat souhaité. Utilisez la fonction [Grille de Test de Matériau](../features/operations/material-test-grid) pour trouver systématiquement les paramètres optimaux.
:::


## Étape 4 : Aperçu

Avant d'envoyer à votre machine, prévisualisez le parcours d'outil en 3D :

1. Cliquez sur **Affichage → Aperçu 3D** ou appuyez sur <kbd>ctrl+3</kbd>
2. La fenêtre d'aperçu 3D affiche le parcours d'outil complet
3. Utilisez votre souris pour faire pivoter et zoomer sur l'aperçu
4. Vérifiez que le parcours semble correct

![Aperçu 3D](/screenshots/main-3d.png)

:::tip Détectez les Erreurs Tôt
L'aperçu 3D vous aide à repérer les problèmes comme :

- Parcours manquants
- Ordre incorrect
- Opérations appliquées aux mauvais objets
- Parcours qui dépassent votre zone de travail
:::


## Étape 5 : Envoyer à la Machine

:::danger Sécurité Avant Tout
- Assurez-vous que la zone de travail est dégagée
- Ne laissez jamais la machine sans surveillance pendant le fonctionnement
- Ayez un équipement de sécurité incendie à proximité
- Portez une protection oculaire appropriée
:::


### Préparer Votre Matériau

1. Placez votre matériau sur le lit laser
2. Faites la mise au point du laser selon les instructions de votre machine
3. Si vous utilisez la caméra, alignez votre design en utilisant la [superposition caméra](../machine/camera)

### Démarrer le Travail

1. **Positionnez le laser** : Utilisez les commandes de déplacement pour déplacer le laser à la position de départ
   - Cliquez sur **Affichage → Panneau de Contrôle** ou appuyez sur <kbd>ctrl+l</kbd>
   - Utilisez les boutons fléchés ou les flèches du clavier pour déplacer le laser
   - Appuyez sur <kbd>home</kbd> pour mettre la machine à l'origine

2. **Cadrer le design** : Exécutez la fonction de cadrage pour vérifier le placement
   - Cliquez sur **Machine → Cadrer** ou appuyez sur <kbd>ctrl+f</kbd>
   - Le laser tracera le rectangle englobant de votre design à faible/sans puissance
   - Vérifiez qu'il tient dans votre matériau

3. **Démarrer le travail** : Cliquez sur **Machine → Démarrer le Travail** ou appuyez sur <kbd>ctrl+r</kbd>
4. Surveillez la progression dans la barre d'état

### Pendant le Travail

- La section droite de la barre d'état affiche la progression actuelle et l'estimation du temps d'exécution total
- Vous pouvez mettre le travail en pause avec <kbd>ctrl+p</kbd> ou cliquer sur le bouton Pause
- Appuyez sur <kbd>esc</kbd> ou cliquez sur Arrêter pour annuler le travail (arrêt d'urgence)

## Étape 6 : Terminer

Une fois le travail terminé :

1. Attendez que le ventilateur d'extraction évacue les fumées
2. Retirez soigneusement votre pièce finie
3. Nettoyez le lit laser si nécessaire

:::success Félicitations !
Vous avez terminé votre premier travail Rayforge ! Vous pouvez maintenant explorer des fonctionnalités plus avancées.
:::


## Prochaines Étapes

Maintenant que vous avez terminé votre premier travail, explorez ces fonctionnalités :

- **[Opérations Multi-Couches](../features/multi-layer)** : Assigner différentes opérations aux calques
- **[Ponts de Maintien](../features/holding-tabs)** : Maintenir les pièces coupées en place pendant la coupe
- **[Intégration Caméra](../machine/camera)** : Utiliser une caméra pour un alignement précis
- **[Hooks & Macros](../machine/hooks-macros)** : Automatiser les tâches répétitives

## Conseils pour Réussir

1. **Sauvegardez votre travail** : Utilisez <kbd>ctrl+s</kbd> pour sauvegarder votre projet fréquemment
2. **Tests de coupe** : Effectuez toujours un test de coupe sur du matériau de récupération d'abord
3. **Base de données matériaux** : Gardez des notes sur les paramètres puissance/vitesse réussis pour différents matériaux
4. **Maintenance** : Gardez votre lentille laser propre et vérifiez régulièrement la tension des courroies
5. **Assistance air** : Si votre machine a une assistance air, utilisez-la pour éviter le brunissage et améliorer la qualité de coupe

---

**Besoin d'Aide ?** Consultez la section [Dépannage](../troubleshooting/connection) ou visitez la page [GitHub Issues](https://github.com/barebaric/rayforge/issues).
