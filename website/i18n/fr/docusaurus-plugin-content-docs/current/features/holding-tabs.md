# Ponts de Maintien

Les ponts de maintien (aussi appelés ponts ou onglets) sont de petites sections non coupées laissées le long des parcours de coupe qui maintiennent les pièces attachées au matériau environnant. Cela empêche les pièces coupées de bouger pendant le travail, ce qui pourrait causer un mauvais alignement, des dommages ou des risques d'incendie.

## Pourquoi Utiliser des Ponts de Maintien ?

Lors de la coupe à travers le matériau, la pièce coupée peut :

- **Changer de position** en milieu de travail, provoquant un mauvais alignement des opérations suivantes
- **Tomber à travers** la grille du lit ou basculer si supportée seulement aux bords
- **Entrer en collision avec** la tête laser lors de ses déplacements
- **Prendre feu** si elle tombe sur des déchets chauds en dessous
- **Être endommagée** par la chute ou les vibrations

Les ponts de maintien résolvent ces problèmes en gardant la pièce attachée jusqu'à ce que vous soyez prêt à la retirer.

---

## Comment Fonctionnent les Ponts de Maintien

Rayforge implémente les ponts en créant de **petits espaces dans le parcours de coupe** :

1. Vous marquez les positions le long du parcours de coupe où les ponts devraient être
2. Pendant la génération du G-code, Rayforge interrompt la coupe à chaque pont
3. Le laser se soulève (ou s'éteint), saute la largeur du pont, puis reprend la coupe
4. Après l'achèvement du travail, vous cassez ou coupez manuellement les ponts pour libérer la pièce

---

## Ajouter des Ponts de Maintien

### Ajout Rapide

1. **Sélectionnez la pièce** à laquelle vous voulez ajouter des ponts (doit être une opération de coupe/contour)
2. **Cliquez sur l'outil pont** dans la barre d'outils ou appuyez sur le raccourci pont
3. **Cliquez sur le parcours** où vous voulez des ponts :
   - Les ponts apparaissent comme de petites poignées sur le contour du parcours
   - Cliquez plusieurs fois pour ajouter plus de ponts
   - Typiquement 3-4 ponts pour les petites pièces, plus pour les pièces plus grandes
4. **Activez les ponts** si ce n'est pas déjà fait (basculez dans le panneau des propriétés)

### Utiliser le Popover Ajouter des Ponts

Pour plus de contrôle :

1. **Clic droit** sur la pièce ou utilisez **Édition → Ajouter des Ponts**
2. **Choisissez la méthode de placement des ponts :**
   - **Manuel :** Cliquez sur les emplacements individuels
   - **Équidistants :** Espace automatiquement les ponts uniformément autour du parcours
3. **Configurez les paramètres des ponts :**
   - **Nombre de ponts :** Combien de ponts créer (pour équidistant)
   - **Largeur du pont :** Longueur de chaque section non coupée (typiquement 2-5mm)
4. **Cliquez sur Appliquer**

---

## Propriétés des Ponts

### Largeur du Pont

La **largeur** est la longueur de la section non coupée le long du parcours.

**Largeurs recommandées :**

| Matériau | Épaisseur | Largeur du Pont |
| -------- | --------- | --------------- |
| **Carton** | 1-3mm | 2-3mm |
| **Contreplaqué** | 3mm | 3-4mm |
| **Contreplaqué** | 6mm | 4-6mm |
| **Acrylique** | 3mm | 2-3mm |
| **Acrylique** | 6mm | 3-5mm |
| **MDF** | 3mm | 3-4mm |
| **MDF** | 6mm | 5-7mm |

**Directives :**
- **Matériaux plus épais** ont besoin de ponts plus larges pour la solidité
- **Pièces plus lourdes** ont besoin de plus et/ou de ponts plus larges
- **Matériaux fragiles** (acrylique) peuvent utiliser des ponts plus petits (plus faciles à casser)
- **Matériaux fibreux** (bois) peuvent nécessiter des ponts plus larges

:::warning Largeur du Pont vs Épaisseur du Matériau
Les ponts doivent être assez larges pour supporter la pièce mais assez petits pour être retirés proprement. Trop étroit = la pièce peut se détacher ; trop large = difficile à retirer ou endommage la pièce.
:::

### Position du Pont

Les ponts sont positionnés en utilisant deux paramètres :

- **Index de segment :** Quel segment de ligne/arc du parcours
- **Position (0.0 - 1.0) :** Où le long de ce segment (0 = début, 1 = fin)

**Conseils de placement manuel :**
- Placez les ponts sur des **sections droites** quand c'est possible (plus facile à retirer)
- Évitez les ponts sur des **courbes serrées** (concentration de contrainte)
- Répartissez les ponts **uniformément** autour de la pièce
- Placez les ponts sur des **coins** pour un support maximum si nécessaire

### Ponts Équidistants

La fonction **équidistante** place automatiquement des ponts à intervalles égaux :

**Avantages :**
- Distribution uniforme du poids
- Motif de rupture prévisible
- Configuration rapide pour les formes régulières

---

## Travailler avec les Ponts

### Éditer les Ponts

**Déplacer un pont :**
1. Sélectionnez la pièce
2. Glissez la poignée du pont le long du parcours
3. Relâchez pour définir la nouvelle position

**Redimensionner un pont :**
- Utilisez le panneau des propriétés pour ajuster la largeur
- Tous les ponts sur une pièce partagent la même largeur

**Supprimer un pont :**
1. Cliquez sur la poignée du pont pour la sélectionner
2. Appuyez sur Supprimer ou utilisez le menu contextuel
3. Ou effacez tous les ponts et recommencez

### Activer/Désactiver les Ponts

Basculer les ponts on/off sans les supprimer :

- **Panneau des propriétés de la pièce :** Case à cocher "Activer les Ponts"
- **Barre d'outils :** Icône de basculement de visibilité des ponts

**Lorsque désactivé :**
- Les ponts ne sont pas générés dans le G-code
- Les poignées de pont sont masquées dans le canevas
- Le parcours coupe complètement à travers

**Cas d'utilisation :** Désactivez temporairement les ponts pour tester la coupe, puis réactivez pour la production.

---

## Retirer les Ponts Après la Coupe

**Outils :**
- Cutter ou couteau à lame rétractable
- Pinces coupantes
- Ciseau (pour le bois)
- Scie fine pour les matériaux épais

**Technique :**

1. **Scorez le pont** des deux côtés si accessible
2. **Pliez doucement** la pièce pour stresser le pont
3. **Coupez à travers** le matériau restant
4. **Poncez ou limez** le reste du pont au ras du bord

**Pour les matériaux fragiles (acrylique) :**
- Utilisez des ponts minimaux (ils se cassent facilement)
- Scorez profondément avant de casser
- Supportez la pièce pendant la rupture des ponts pour éviter les fissures

**Pour le bois :**
- Les ponts peuvent nécessiter une coupe (ne se cassent pas proprement)
- Utilisez un couteau bien aiguisé ou un ciseau
- Coupez au ras, puis poncez lisse

---

## Pages Connexes

- [Coupe de Contour](operations/contour) - Opération principale qui utilise des ponts
- [Flux de Travail Multi-Couches](multi-layer) - Gérer les ponts sur plusieurs calques
- [Aperçu 3D](../ui/3d-preview) - Visualiser les ponts dans l'aperçu
- [Mode Simulation](simulation-mode) - Prévisualiser les coupes avec des espaces de pont
