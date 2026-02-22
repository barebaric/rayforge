# Macros & Hooks

Rayforge fournit deux puissantes fonctionnalités d'automatisation pour personnaliser votre flux de travail : **Macros** et **Hooks**. Les deux vous permettent d'injecter du G-code personnalisé dans vos travaux, mais ils servent des objectifs différents.

![Paramètres Hooks & Macros](/screenshots/machine-hooks-macros.png)

---

## Aperçu

| Fonctionnalité | Objectif | Déclencheur | Cas d'Utilisation |
| -------------- | -------- | ----------- | ----------------- |
| **Macros** | Extraits de G-code réutilisables | Exécution manuelle | Commandes rapides, motifs de test, routines personnalisées |
| **Hooks** | Injection automatique de G-code | Événements du cycle de vie du travail | Séquences de démarrage, changements de calque, nettoyage |

---

## Macros

Les macros sont des **scripts G-code nommés et réutilisables** que vous pouvez exécuter manuellement à tout moment.

### À quoi Servent les Macros ?

Cas d'utilisation courants des macros :

- **Mettre la machine à l'origine** - Envoyer `$H` rapidement
- **Définir les décalages de travail** - Stocker et rappeler les positions G54/G55
- **Contrôle de l'assistance air** - Activer/désactiver l'assistance air
- **Test de mise au point** - Exécuter un motif de test de mise au point rapide
- **Changements d'outil personnalisés** - Pour les configurations multi-lasers
- **Routines d'urgence** - Arrêt rapide ou effacement d'alarme
- **Sondage de matériau** - Autofocus ou mesure de hauteur

### Créer une Macro

1. **Ouvrir les Paramètres Machine :**
   - Naviguez vers **Paramètres Machine Macros**

2. **Ajouter une nouvelle macro :**
   - Cliquez sur le bouton **"+"**
   - Entrez un nom descriptif (ex : "Mettre Machine à l'Origine", "Activer Assistance Air")

3. **Écrivez votre G-code :**
   - Chaque ligne est une commande G-code séparée
   - Les commentaires commencent par `;` ou `(`
   - Des variables peuvent être utilisées (voir Substitution de Variables ci-dessous)

4. **Sauvegardez la macro**

5. **Exécutez la macro :**
   - Depuis la liste des macros, cliquez sur la macro
   - Ou assignez un raccourci clavier (si supporté)

### Exemples de Macros

#### Simple : Mettre la Machine à l'Origine

**Nom :** Mettre Machine à l'Origine
**Code :**

```gcode
$H
; Attend que le homing soit terminé
```

**Utilisation :** Mettre rapidement la machine à l'origine avant de commencer le travail.

---

#### Moyen : Définir le Décalage de Travail

**Nom :** Définir G54 à la Position Actuelle
**Code :**

```gcode
G10 L20 P1 X0 Y0
; Définit l'origine du système de coordonnées de travail G54 à la position actuelle
```

**Utilisation :** Marquer la position actuelle du laser comme origine du travail.

---

#### Avancé : Grille de Test de Mise au Point

**Nom :** Test de Mise au Point 9 Points
**Code :**

```gcode
; Grille 9 points pour trouver la mise au point optimale
G21  ; Millimètres
G90  ; Positionnement absolu
G0 X10 Y10
M3 S1000
G4 P0.1
M5
G0 X20 Y10
M3 S1000
G4 P0.1
M5
; ... (répéter pour les points restants)
```

**Utilisation :** Tester rapidement la mise au point à différentes positions sur le lit.

---

---

### Exemples de Macros

Les hooks sont des **injections automatiques de G-code** déclenchées par des événements spécifiques pendant l'exécution du travail.

### Déclencheurs de Hooks

Rayforge supporte ces déclencheurs de hooks :

| Déclencheur | Quand il S'exécute | Utilisations Courantes |
| ----------- | ------------------ | ---------------------- |
| **Début de Travail** | Tout au début du travail | Homing, décalage de travail, assistance air activée, préchauffage |
| **Fin de Travail** | Tout à la fin du travail | Retour à l'origine, assistance air désactivée, bip, refroidissement |
| **Début de Calque** | Avant le traitement de chaque calque | Changement d'outil, ajustement de puissance, commentaires |
| **Fin de Calque** | Après le traitement de chaque calque | Notification de progression, pause |
| **Début de Pièce** | Avant le traitement de chaque pièce | Numérotation des pièces, marques d'alignement |
| **Fin de Pièce** | Après le traitement de chaque pièce | Refroidissement, pause d'inspection |

### Créer un Hook

1. **Ouvrir les Paramètres Machine :**
   - Naviguez vers **Paramètres Machine Hooks**

2. **Sélectionner un déclencheur :**
   - Choisissez l'événement quand ce hook doit s'exécuter

3. **Écrivez votre G-code :**
   - Le code du hook est injecté au point de déclenchement
   - Utilisez des variables pour des valeurs dynamiques (voir ci-dessous)

4. **Activer/désactiver :**
   - Activez/désactivez les hooks sans les supprimer

### Exemples de Hooks

#### Début de Travail : Initialiser la Machine

**Déclencheur :** Début de Travail
**Code :**

```gcode
G21         ; Millimètres
G90         ; Positionnement absolu
$H          ; Mettre la machine à l'origine
G0 X0 Y0    ; Déplacer à l'origine
M3 S0       ; Laser activé mais puissance 0 (certains contrôleurs ont besoin de cela)
M8          ; Assistance air ACTIVÉE
```

**Objectif :** S'assure que la machine est dans un état connu avant chaque travail.

---

#### Fin de Travail : Retour à l'Origine et Bip

**Déclencheur :** Fin de Travail
**Code :**

```gcode
M5          ; Laser ÉTEINT
M9          ; Assistance air ÉTEINTE
G0 X0 Y0    ; Retour à l'origine
M300 S800 P200  ; Bip (si supporté)
```

**Objectif :** Termine le travail en toute sécurité et signale l'achèvement.

---

#### Début de Calque : Ajouter un Commentaire

**Déclencheur :** Début de Calque
**Code :**

```gcode
; Démarrage du calque : {layer_name}
; Index du calque : {layer_index}
```

**Objectif :** Rend le G-code plus lisible pour le débogage.

---

#### Début de Pièce : Numérotation des Pièces

**Déclencheur :** Début de Pièce
**Code :**

```gcode
; Pièce : {workpiece_name}
; Pièce {workpiece_index} sur {total_workpieces}
```

**Objectif :** Suivre la progression dans les travaux multi-pièces.

---

### Ordre d'Exécution des Hooks

Pour un travail avec 2 calques, chacun avec 2 pièces :

```
[Hook Début de Travail]
  [Hook Début de Calque] (Calque 1)
    [Hook Début de Pièce] (Pièce 1)
      ... G-code de la pièce 1 ...
    [Hook Fin de Pièce] (Pièce 1)
    [Hook Début de Pièce] (Pièce 2)
      ... G-code de la pièce 2 ...
    [Hook Fin de Pièce] (Pièce 2)
  [Hook Fin de Calque] (Calque 1)
  [Hook Début de Calque] (Calque 2)
    [Hook Début de Pièce] (Pièce 3)
      ... G-code de la pièce 3 ...
    [Hook Fin de Pièce] (Pièce 3)
    [Hook Début de Pièce] (Pièce 4)
      ... G-code de la pièce 4 ...
    [Hook Fin de Pièce] (Pièce 4)
  [Hook Fin de Calque] (Calque 2)
[Hook Fin de Travail]
```

---

## Substitution de Variables

Les macros et hooks supportent la **substitution de variables** pour injecter des valeurs dynamiques.

### Variables Disponibles

Les variables utilisent la syntaxe `{nom_variable}` et sont remplacées pendant la génération du G-code.

**Variables au niveau du travail :**

| Variable | Description | Exemple de Valeur |
| -------- | ----------- | ----------------- |
| `{job_name}` | Nom du travail/document actuel | "test-job" |
| `{date}` | Date actuelle | "2025-10-03" |
| `{time}` | Heure actuelle | "14:30:25" |

**Variables au niveau du calque :**

| Variable | Description | Exemple de Valeur |
| -------- | ----------- | ----------------- |
| `{layer_name}` | Nom du calque actuel | "Calque de Coupe" |
| `{layer_index}` | Index base 0 du calque actuel | 0, 1, 2... |
| `{total_layers}` | Nombre total de calques dans le travail | 3 |

**Variables au niveau de la pièce :**

| Variable | Description | Exemple de Valeur |
| -------- | ----------- | ----------------- |
| `{workpiece_name}` | Nom de la pièce | "Cercle 1" |
| `{workpiece_index}` | Index base 0 de la pièce actuelle | 0, 1, 2... |
| `{total_workpieces}` | Nombre total de pièces | 5 |

**Variables machine :**

| Variable | Description | Exemple de Valeur |
| -------- | ----------- | ----------------- |
| `{machine_name}` | Nom du profil de machine | "Ma K40" |
| `{max_speed}` | Vitesse de coupe maximale (mm/min) | 1000 |
| `{work_width}` | Largeur de la zone de travail (mm) | 300 |
| `{work_height}` | Hauteur de la zone de travail (mm) | 200 |

### Exemple : Notification de Progression

**Hook :** Début de Calque
**Code :**

```gcode
; ========================================
; Calque {layer_index} sur {total_layers} : {layer_name}
; Travail : {job_name}
; Heure : {time}
; ========================================
```

**Résultat dans le G-code :**

```gcode
; ========================================
; Calque 0 sur 3 : Calque de Coupe
; Travail : test-project
; Heure : 14:30:25
; ========================================
```

---

## Cas d'Utilisation Avancés

### Configuration Multi-Outils

Pour les machines avec plusieurs lasers ou outils :

**Hook :** Début de Pièce
**Code :**

```gcode
; Sélectionner l'outil pour la pièce {workpiece_name}
T{tool_number}  ; Commande de changement d'outil (si supporté)
G4 P1           ; Attendre le changement d'outil
```

### Pauses Conditionnelles

Ajoutez des pauses optionnelles pour l'inspection :

**Hook :** Fin de Calque
**Code :**

```gcode
; M0  ; Décommenter pour暂停 après chaque calque pour inspection
```

### Assistance Air par Calque

Contrôlez l'assistance air calque par calque :

**Hook :** Début de Calque (pour calques de coupe)
**Code :**

```gcode
M8  ; Assistance air ACTIVÉE
```

**Hook :** Début de Calque (pour calques de gravure)
**Code :**

```gcode
M9  ; Assistance air ÉTEINTE (empêche la dispersion de poussière pour la gravure)
```

:::note Hooks Spécifiques au Calque
Rayforge ne supporte actuellement pas la personnalisation de hooks par calque. Pour y parvenir, utilisez du G-code conditionnel ou des profils de machine séparés.
:::

---

## Considérations de Sécurité

:::danger Testez Avant Production
Testez toujours les macros et hooks en **mode simulation** ou avec le laser **désactivé** avant de les exécuter sur de vrais travaux. Un G-code mal configuré peut :

- Faire planter la machine contre les limites
- Déclencher le laser de manière inattendue
- Endommager les matériaux ou équipements
:::

**Liste de contrôle de sécurité :**

- [ ] Les macros incluent des limites de vitesse d'avance (paramètre `F`)
- [ ] Les macros vérifient les limites de position
- [ ] Les hooks de début de travail incluent `M5` ou commande laser éteint
- [ ] Les hooks de fin de travail éteignent le laser (`M5`) et l'assistance air (`M9`)
- [ ] Pas de commandes destructives sans confirmation
- [ ] Testé en simulation ou avec laser désactivé

---

## Pages Connexes

- [Paramètres de l'Appareil](device) - Référence des commandes GRBL
- [Dialectes G-code](../reference/gcode-dialects) - Compatibilité G-code
- [Paramètres Généraux](general) - Configuration de la machine
- [Flux de Travail Multi-Couches](../features/multi-layer) - Utiliser les hooks avec les calques
