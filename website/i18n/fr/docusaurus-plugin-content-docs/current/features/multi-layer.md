# Flux de Travail Multi-Couches

Le système multi-couches de Rayforge vous permet d'organiser des travaux complexes en étapes de traitement séparées, chacune avec ses propres opérations et paramètres. C'est essentiel pour combiner différents processus comme la gravure et la coupe, ou travailler avec plusieurs matériaux.

## Que Sont les Calques ?

Un **calque** dans Rayforge est :

- **Un conteneur** pour les pièces (formes importées, images, texte)
- **Un flux de travail** définissant comment ces pièces sont traitées
- **Une étape** traitée séquentiellement pendant les travaux

**Concept clé :** Les calques sont traités dans l'ordre, un après l'autre, vous permettant de contrôler la séquence des opérations.

:::note Calques et Pièces
Un calque contient une ou plusieurs pièces. Lors de l'importation de fichiers SVG avec des calques, chaque calque de votre design devient un calque séparé dans Rayforge.
Cela vous permet de garder votre design organisé exactement comme vous l'avez créé.
:::


---

## Pourquoi Utiliser Plusieurs Calques ?

### Cas d'Utilisation Courants

**1. Graver puis Couper**

Le flux de travail multi-couches le plus courant :

- **Calque 1 :** Gravure raster du design
- **Calque 2 :** Coupe de contour du contour

**Pourquoi des calques séparés ?**

- La gravure d'abord assure que la pièce ne bouge pas pendant la gravure
- La coupe à la fin empêche les pièces de tomber avant la fin de la gravure
- Différents paramètres puissance/vitesse pour chaque opération

**2. Coupe Multi-Passes**

Pour les matériaux épais :

- **Calque 1 :** Première passe à puissance modérée
- **Calque 2 :** Deuxième passe à pleine puissance (même géométrie)
- **Calque 3 :** Troisième passe optionnelle si nécessaire

**Avantages :**

- Réduit le brunissement comparé à une seule passe à haute puissance
- Chaque calque peut avoir des paramètres vitesse/puissance différents

**3. Projets Multi-Matériaux**

Différents matériaux dans un travail :

- **Calque 1 :** Couper les pièces en acrylique
- **Calque 2 :** Graver les pièces en bois
- **Calque 3 :** Marquer les pièces en métal

**Prérequis :**

- Chaque calque cible différentes zones du lit
- Différents vitesse/puissance/mise au point pour chaque matériau

**4. Import de Calques SVG**

Importer des fichiers SVG avec une structure de calques existante :

- **Calque 1 :** Éléments de gravure du SVG
- **Calque 2 :** Éléments de coupe du SVG
- **Calque 3 :** Éléments de marquage du SVG

**Flux de travail :**

- Importez un fichier SVG qui a des calques
- Activez "Utiliser les Vecteurs Originaux" dans la boîte de dialogue d'importation
- Sélectionnez quels calques importer depuis la liste des calques détectés
- Chaque calque devient un calque séparé dans Rayforge

**Prérequis :**

- Votre fichier SVG doit utiliser des calques (créés dans Inkscape ou logiciel similaire)
- Activez "Utiliser les Vecteurs Originaux" lors de l'importation
- Les noms de calques sont préservés depuis votre logiciel de design

---

## Créer et Gérer les Calques

### Ajouter un Nouveau Calque

1. **Cliquez sur le bouton "+"** dans le panneau Calques
2. **Nommez le calque** de manière descriptive (ex : "Calque Gravure", "Calque Coupe")
3. **Le calque apparaît** dans la liste des calques

**Par défaut :** Les nouveaux documents commencent avec un calque.

### Propriétés du Calque

Chaque calque a :

| Propriété | Description |
| --------- | ----------- |
| **Nom** | Le nom affiché dans la liste des calques |
| **Visible** | Basculer la visibilité dans le canevas et l'aperçu |
| **Élément de Matériau** | Association de matériau optionnelle |
| **Flux de Travail** | Les opérations appliquées aux pièces de ce calque |
| **Pièces** | Les formes/images contenues dans ce calque |

:::note Calques comme Conteneurs
Les calques sont des conteneurs pour vos pièces. Lors de l'importation de fichiers SVG avec des calques, chaque calque de votre design devient un calque séparé dans Rayforge.
:::


### Réorganiser les Calques

**L'ordre d'exécution = ordre des calques dans la liste (de haut en bas)**

Pour réorganiser :

1. **Glissez et déposez** les calques dans le panneau Calques
2. **L'ordre compte** - les calques s'exécutent de haut en bas

**Exemple :**

```
Panneau Calques :
1. Calque Gravure     S'exécute en premier
2. Calque Marquage    S'exécute en deuxième
3. Calque Coupe       S'exécute en dernier (recommandé)
```

### Supprimer des Calques

1. **Sélectionnez le calque** dans le panneau Calques
2. **Cliquez sur le bouton supprimer** ou appuyez sur Supprimer
3. **Confirmez la suppression** (toutes les pièces du calque sont supprimées)

:::warning La Suppression est Permanente
Supprimer un calque supprime toutes ses pièces et ses paramètres de flux de travail. Utilisez Annuler si vous supprimez accidentellement.
:::


---

## Assigner des Pièces aux Calques

### Assignation Manuelle

1. **Importez ou créez** une pièce
2. **Glissez la pièce** vers le calque souhaité dans le panneau Calques
3. **Ou utilisez le panneau des propriétés** pour changer le calque de la pièce

### Import de Calques SVG

Lors de l'importation de fichiers SVG avec "Utiliser les Vecteurs Originaux" activé :

1. **Activez "Utiliser les Vecteurs Originaux"** dans la boîte de dialogue d'importation
2. **Rayforge détecte les calques** de votre fichier SVG
3. **Sélectionnez quels calques** importer en utilisant les interrupteurs de calque
4. **Chaque calque sélectionné** devient un calque séparé avec sa propre pièce

:::note Détection de Calques
Rayforge détecte automatiquement les calques de votre fichier SVG. Chaque calque que vous avez créé dans votre logiciel de design apparaîtra comme un calque séparé dans Rayforge.
:::


:::note Import Vectoriel Uniquement
La sélection de calques n'est disponible que lors de l'utilisation de l'import vectoriel direct.
Lors de l'utilisation du mode trace, le SVG entier est traité comme une seule pièce.
:::


### Déplacer des Pièces Entre les Calques

**Glisser et déposer :**

- Sélectionnez la/les pièce(s) dans le canevas ou le panneau Document
- Glissez vers le calque cible dans le panneau Calques

**Couper et coller :**

- Coupez la pièce du calque actuel (Ctrl+X)
- Sélectionnez le calque cible
- Collez (Ctrl+V)

### Boîte de Dialogue d'Import SVG

Lors de l'importation de fichiers SVG, la boîte de dialogue d'importation fournit des options qui affectent la gestion des calques :

**Mode d'Importation :**

- **Utiliser les Vecteurs Originaux :** Préserve vos parcours vectoriels et la structure des calques.
  Lorsqu'activé, une section "Calques" apparaît montrant tous les calques de votre fichier.
- **Mode Trace :** Convertit le SVG en bitmap et trace les contours.
  La sélection de calques est désactivée dans ce mode.

**Section Calques (Import Vectoriel Uniquement) :**

- Affiche tous les calques de votre fichier SVG
- Chaque calque a un interrupteur pour activer/désactiver l'importation
- Les noms de calques de votre logiciel de design sont préservés
- Seuls les calques sélectionnés sont importés comme calques séparés

:::tip Préparer les Fichiers SVG pour l'Import de Calques
Pour utiliser l'import de calques SVG, créez votre design avec des calques dans un logiciel comme Inkscape. Utilisez le panneau Calques pour organiser votre design, et Rayforge
préservera cette structure.
:::


---

## Flux de Travail des Calques

Chaque calque a un **Flux de Travail** qui définit comment ses pièces sont traitées.

### Configurer les Flux de Travail des Calques

Pour chaque calque, vous choisissez un type d'opération et configurez ses paramètres :

**Types d'Opérations :**

- **Contour** - Suit les contours (pour la coupe ou le marquage)
- **Gravure Raster** - Grave les images et remplit les zones
- **Gravure en Profondeur** - Crée des gravures de profondeur variable

**Améliorations Optionnelles :**

- **Onglets** - Petits ponts pour maintenir les pièces en place pendant la coupe
- **Overscan** - Étend les coupes au-delà de la forme pour des bords plus propres
- **Ajustement de Kerf** - Compense la largeur de coupe du laser

### Configurations de Calques Courantes

**Calque de Gravure :**

- Opération : Gravure Raster
- Paramètres : 300-500 DPI, vitesse modérée
- Typiquement aucune option supplémentaire nécessaire

**Calque de Coupe :**

- Opération : Coupe de Contour
- Options : Onglets (pour maintenir les pièces), Overscan (pour des bords propres)
- Paramètres : Vitesse plus lente, puissance plus élevée

**Calque de Marquage :**

- Opération : Contour (puissance légère, ne coupe pas à travers)
- Paramètres : Faible puissance, vitesse rapide
- Objectif : Lignes de pliage, lignes décoratives

---

## Visibilité des Calques

Contrôlez quels calques sont affichés dans le canevas et les aperçus :

### Visibilité du Canevas

- **Icône œil** dans le panneau Calques bascule la visibilité
- **Calques masqués :**
  - Non affichés dans le canevas 2D
  - Non affichés dans l'aperçu 3D
  - **Toujours inclus dans le G-code généré**

**Cas d'utilisation :**

- Masquer les calques de gravure complexes pendant le positionnement des calques de coupe
- Désencombrer le canevas lors du travail sur des calques spécifiques
- Se concentrer sur un calque à la fois

### Visibilité vs Activé

| État | Canevas | Aperçu | G-code |
| ---- | ------- | ------ | ------ |
| **Visible & Activé** | Oui | Oui | Oui |
| **Masqué & Activé** | Non | Non | Oui |
| **Visible & Désactivé** | Oui | Oui | Non |
| **Masqué & Désactivé** | Non | Non | Non |

:::note Désactiver les Calques
:::

Pour exclure temporairement un calque des travaux sans le supprimer, désactivez l'opération du calque ou désactivez-le dans les paramètres du calque.

---

## Ordre d'Exécution des Calques

### Comment les Calques Sont Traités

Pendant l'exécution du travail, Rayforge traite chaque calque dans l'ordre de haut en bas. Dans chaque calque, toutes les pièces sont traitées avant de passer au calque suivant.

### L'Ordre Compte

**Mauvais ordre :**

```
1. Calque Coupe
2. Calque Gravure
```

**Problème :** Les pièces coupées peuvent tomber ou bouger avant la gravure !

**Ordre correct :**

```
1. Calque Gravure
2. Calque Coupe
```

**Pourquoi :** La gravure se produit pendant que la pièce est encore attachée, puis la coupe la libère.

### Passes Multiples

Pour les matériaux épais, créez plusieurs calques de coupe :

```
1. Calque Gravure
2. Calque Coupe (Passe 1) - 50% puissance
3. Calque Coupe (Passe 2) - 75% puissance
4. Calque Coupe (Passe 3) - 100% puissance
```

**Conseil :** Utilisez la même géométrie pour toutes les passes de coupe (dupliquez le calque).

---

## Techniques Avancées

### Groupement de Calques par Matériau

Utilisez les calques pour organiser par matériau lors de l'exécution de travaux mixtes :

```
Matériau 1 (Acrylique 3mm) :
  - Calque Gravure Acrylique
  - Calque Coupe Acrylique

Matériau 2 (Contreplaqué 3mm) :
  - Calque Gravure Bois
  - Calque Coupe Bois
```

**Flux de travail :**

1. Traitez tous les calques du Matériau 1
2. Changez de matériau
3. Traitez tous les calques du Matériau 2

**Alternative :** Utilisez des documents séparés pour différents matériaux.

### Pause Entre les Calques

Vous pouvez configurer Rayforge pour faire une pause entre les calques. C'est utile quand vous avez besoin de :

- Changer de matériau en milieu de travail
- Inspecter la progression avant de continuer
- Ajuster la mise au point pour différentes opérations

Pour configurer les pauses de calque, utilisez la fonction hooks dans vos paramètres machine.

### Paramètres Spécifiques au Calque

Chaque flux de travail de calque peut avoir des paramètres uniques :

| Calque | Opération | Vitesse | Puissance | Passes |
| ------ | --------- | ------- | --------- | ------ |
| Gravure | Raster | 300 mm/min | 20% | 1 |
| Marquage | Contour | 500 mm/min | 10% | 1 |
| Coupe | Contour | 100 mm/min | 90% | 2 |

---

## Meilleures Pratiques

### Conventions de Nommage

**Bons noms de calques :**

- "Gravure - Logo"
- "Coupe - Contour Extérieur"
- "Marquage - Lignes de Pliage"
- "Passe 1 - Coupe Ébauche"
- "Passe 2 - Coupe Finale"

**Mauvais noms de calques :**

- "Calque 1", "Calque 2" (pas descriptif)
- Longues descriptions (restez concis)

### Organisation des Calques

1. **De haut en bas = ordre d'exécution**
2. **Gravure avant coupe** (règle générale)
3. **Groupez les opérations liées** (toute coupe, toute gravure)
4. **Utilisez la visibilité** pour vous concentrer sur le travail actuel
5. **Supprimez les calques inutilisés** pour garder les projets propres

### Préparer les Fichiers SVG pour l'Import de Calques

**Pour de meilleurs résultats lors de l'importation de calques SVG :**

1. **Utilisez le panneau Calques** dans votre logiciel de design pour organiser votre design
2. **Assignez des noms significatifs** à chaque calque (ex : "Gravure", "Coupe")
3. **Gardez les calques plats** - évitez de mettre des calques dans d'autres calques
4. **Sauvegardez votre fichier** et importez dans Rayforge
5. **Vérifiez la détection des calques** en vérifiant la boîte de dialogue d'importation

Rayforge fonctionne mieux avec les fichiers SVG créés dans Inkscape ou un logiciel de design vectoriel similaire qui supporte les calques.

### Performance

**Beaucoup de calques :**

- Pas d'impact significatif sur les performances
- 10-20 calques est courant pour les travaux complexes
- Organisez logiquement, pas pour minimiser le nombre de calques

**Simplifiez si nécessaire :**

- Combinez des opérations similaires dans un calque quand c'est possible
- Utilisez moins de gravures raster (les plus gourmandes en ressources)

---

## Dépannage

### Le Calque Ne Génère Pas de G-code

**Problème :** Le calque apparaît dans le document mais pas dans le G-code généré.

**Solutions :**

1. **Vérifiez que le calque a des pièces** - Les calques vides sont ignorés
2. **Vérifiez que le flux de travail est configuré** - Le calque a besoin d'une opération
3. **Vérifiez les paramètres de l'opération** - Puissance > 0, vitesse valide, etc.
4. **Vérifiez la visibilité de la pièce** - Les pièces masquées peuvent ne pas être traitées
5. **Régénérez le G-code** - Faites un petit changement pour forcer la régénération

### Mauvais Ordre de Calques

**Problème :** Les opérations s'exécutent dans un ordre inattendu.

**Solution :** Réorganisez les calques dans le panneau Calques. Rappelez-vous : haut = premier.

### Calques Qui Se Chevauchent dans l'Aperçu

**Problème :** Plusieurs calques affichent un contenu qui se chevauche dans l'aperçu.

**Clarification :** C'est normal si les calques partagent la même zone XY.

**Solutions :**

- Utilisez la visibilité des calques pour masquer temporairement les autres calques
- Vérifiez l'aperçu 3D pour voir la profondeur/ordre
- Vérifiez que c'est intentionnel (ex : graver puis couper la même forme)

### Pièce dans le Mauvais Calque

**Problème :** La pièce a été assignée au mauvais calque.

**Solution :** Glissez la pièce vers le bon calque dans le panneau Calques ou l'arborescence Document.

### Calques SVG Non Détectés

**Problème :** Importation d'un fichier SVG mais aucun calque n'apparaît dans la boîte de dialogue d'importation.

**Solutions :**

1. **Vérifiez la structure SVG** - Ouvrez votre fichier dans Inkscape ou logiciel similaire pour vérifier qu'il a des calques
2. **Activez "Utiliser les Vecteurs Originaux"** - La sélection de calques n'est disponible que dans ce mode d'importation
3. **Vérifiez que votre design a des calques** - Assurez-vous d'avoir créé des calques dans votre logiciel de design, pas juste des groupes
4. **Vérifiez les calques imbriqués** - Les calques dans d'autres calques peuvent ne pas être détectés correctement
5. **Resauvegardez votre fichier** - Parfois resauvegarder avec une version actuelle de votre logiciel de design aide

### L'Import de Calques SVG Affiche le Mauvais Contenu

**Problème :** Le calque importé affiche le contenu d'autres calques ou est vide.

**Solutions :**

1. **Vérifiez la sélection de calque** - Vérifiez que les bons calques sont activés dans la boîte de dialogue d'importation
2. **Vérifiez votre design** - Ouvrez le fichier original dans votre logiciel de design pour confirmer que chaque calque contient le bon contenu
3. **Vérifiez les éléments partagés** - Les éléments qui apparaissent dans plusieurs calques peuvent causer de la confusion
4. **Essayez le mode trace** - Utilisez le mode trace comme solution de repli si l'import vectoriel a des problèmes

---

## Pages Connexes

- [Opérations](./operations/contour) - Types d'opérations pour les flux de travail de calque
- [Mode Simulation](./simulation-mode) - Prévisualiser l'exécution multi-couches
- [Macros & Hooks](../machine/hooks-macros) - Hooks au niveau du calque pour l'automatisation
- [Aperçu 3D](../ui/3d-preview) - Visualiser la pile de calques
