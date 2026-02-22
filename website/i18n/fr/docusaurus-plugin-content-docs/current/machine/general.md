# Paramètres Généraux

La page Général dans les Paramètres Machine contient les informations de base de la machine et les paramètres de vitesse.

![Paramètres Généraux](/screenshots/machine-general.png)

## Nom de la Machine

Donnez un nom descriptif à votre machine. Cela aide à identifier la machine dans le menu déroulant de sélection de machine lorsque vous avez plusieurs machines configurées.

Exemples :
- "K40 Atelier"
- "Laser Diode Garage"
- "Ortur LM2 Pro"

## Vitesses & Accélération

Ces paramètres contrôlent les vitesses maximums et l'accélération pour la planification de mouvement et l'estimation du temps.

### Vitesse de Déplacement Max

La vitesse maximum pour les mouvements rapides (non-coupe). C'est utilisé lorsque le laser est éteint et que la tête se déplace vers une nouvelle position.

- **Plage typique** : 2000-5000 mm/min
- **Objectif** : Planification de mouvement et estimation du temps
- **Note** : La vitesse réelle est aussi limitée par vos paramètres de firmware

### Vitesse de Coupe Max

La vitesse maximum autorisée pendant les opérations de coupe ou de gravure.

- **Plage typique** : 500-2000 mm/min
- **Objectif** : Limite les vitesses d'opération pour la sécurité
- **Note** : Les opérations individuelles peuvent utiliser des vitesses plus basses

### Accélération

Le taux auquel la machine accélère et décélère.

- **Plage typique** : 500-2000 mm/s²
- **Objectif** : Estimation du temps et planification de mouvement
- **Note** : Doit correspondre ou être inférieur aux paramètres d'accélération du firmware

:::tip
Commencez avec des valeurs de vitesse conservatrices et augmentez progressivement. Observez votre machine pour les sauts de courroie, les calages de moteur ou la perte de précision de positionnement.
:::

## Voir Aussi

- [Paramètres Matériels](hardware) - Dimensions de machine et configuration des axes
- [Paramètres de l'Appareil](device) - Connexion et paramètres GRBL
