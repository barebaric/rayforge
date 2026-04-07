# Zones Interdites

Les zones interdites définissent des zones restreintes sur la surface de travail que le laser ne
doit pas traverser. Avant d'exécuter ou d'exporter un travail, Rayforge vérifie si des
trajets d'outil entrent dans une zone interdite activée et t'avertit si une collision est détectée.

![Zones Interdites](/screenshots/machine-nogo-zones.png)

## Ajouter une Zone Interdite

Ouvre **Paramètres → Machine** et accède à la page **Zones Interdites**. Clique
sur le bouton d'ajout pour créer une nouvelle zone, puis choisis sa forme et sa position.

Chaque zone a les paramètres suivants :

- **Nom** : Une étiquette descriptive pour la zone
- **Activé** : Activer ou désactiver la zone sans la supprimer
- **Forme** : Rectangle, Boîte ou Cylindre
- **Position (X, Y, Z)** : Où la zone est placée sur la surface de travail
- **Dimensions** : Largeur, hauteur et profondeur (ou rayon pour les cylindres)

## Avertissements de Collision

Lorsque tu exécutes ou exportes un travail, Rayforge vérifie tous les trajets d'outil par rapport
aux zones interdites activées. Si un trajet d'outil traverse une zone, une boîte de dialogue
d'avertissement apparaît avec la possibilité d'annuler ou de continuer à tes propres risques.

## Visibilité

Les zones interdites sont affichées sur le canevas 2D et 3D comme des superpositions
semi-transparentes. Utilise le bouton de bascule des zones interdites dans la superposition
du canevas pour les afficher ou les masquer. Le paramètre de visibilité est mémorisé entre
les sessions.

---

## Pages Connexes

- [Paramètres Matériel](hardware) - Dimensions de la machine et configuration des axes
- [Vue 3D](../ui/3d-preview) - Visualisation du trajet d'outil 3D
