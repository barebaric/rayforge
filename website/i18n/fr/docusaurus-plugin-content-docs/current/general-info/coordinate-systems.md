# Systèmes de Coordonnées

Comprendre comment Rayforge gère les systèmes de coordonnées est essentiel pour positionner correctement votre travail.

## Système de Coordonnées de Travail (WCS) vs Coordonnées Machine

Rayforge utilise deux systèmes de coordonnées principaux :

### Système de Coordonnées de Travail (WCS)

Le WCS est le système de coordonnées de votre travail. Quand vous positionnez un design à (50, 100) sur le canevas, ce sont des coordonnées WCS.

- **Origine** : Définie par vous (défaut est G54)
- **Objectif** : Design et positionnement du travail
- **Systèmes multiples** : G54-G59 disponibles pour différentes configurations

### Coordonnées Machine

Les coordonnées machine sont des positions absolues relatives à la position d'origine de la machine.

- **Origine** : Origine machine (0,0,0) - fixée par le matériel
- **Objectif** : Positionnement physique sur le lit
- **Fixe** : Ne peut pas être changé par logiciel

**Relation** : Les décalages WCS définissent comment vos coordonnées de travail sont mappées aux coordonnées machine. Si le décalage G54 est (100, 50, 0), alors votre design à WCS (0, 0) coupe à la position machine (100, 50).

## Configurer les Coordonnées dans Rayforge

### Définir l'Origine WCS

Pour positionner votre travail sur la machine :

1. **Mettez la machine à l'origine** d'abord (commande `$H` ou bouton Home)
2. **Déplacez la tête laser** vers votre origine de travail souhaitée
3. **Définissez le zéro WCS** en utilisant le Panneau de Contrôle :
   - Cliquez sur "Zéro X" pour définir le X actuel comme origine
   - Cliquez sur "Zéro Y" pour définir le Y actuel comme origine
4. Votre travail démarrera maintenant de cette position

### Sélectionner un WCS

Rayforge supporte les systèmes de coordonnées de travail G54-G59 :

| Système | Cas d'Utilisation |
| -------- | ----------------- |
| G54 | Défaut, zone de travail principale |
| G55-G59 | Positions de montage supplémentaires |

Sélectionnez le WCS actif dans le Panneau de Contrôle. Chaque système stocke son propre décalage depuis l'origine machine.

### Direction de l'Axe Y

Certaines machines ont Y augmentant vers le bas au lieu de vers le haut. Configurez cela dans :

**Paramètres → Machine → Matériel → Axes**

Si vos travaux sortent en miroir verticalement, basculez le paramètre de direction de l'axe Y.

## Problèmes Courants

### Travail à la Mauvaise Position

- **Vérifiez le décalage WCS** : Envoyez `G10 L20 P1` pour voir le décalage G54
- **Vérifiez le homing** : La machine doit être à l'origine pour un positionnement cohérent
- **Vérifiez la direction de l'axe Y** : Peut être inversée

### Dérive des Coordonnées Entre les Travaux

- **Toujours mettre à l'origine avant les travaux** : Établit une référence cohérente
- **Vérifiez les décalages G92** : Effacez avec la commande `G92.1`

---

## Pages Connexes

- [Systèmes de Coordonnées de Travail (WCS)](work-coordinate-systems) - Gérer le WCS dans Rayforge
- [Panneau de Contrôle](../ui/control-panel) - Contrôles de déplacement et boutons WCS
- [Exporter le G-code](../files/exporting) - Options de positionnement du travail
