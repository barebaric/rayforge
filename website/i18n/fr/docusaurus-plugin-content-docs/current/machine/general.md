---
description: "Configurez les paramètres généraux de la machine dans Rayforge — définissez le nom, sélectionnez un pilote et configurez les vitesses et l'accélération."
---

# Paramètres généraux

La page Général des Paramètres de la machine contient le nom de la machine,
la sélection du pilote et les paramètres de connexion, ainsi que les
paramètres de vitesse.

![Paramètres généraux](/screenshots/machine-general.png)

## Nom de la machine

Donnez un nom descriptif à votre machine. Cela permet de l'identifier dans
le menu déroulant de sélection lorsque vous avez plusieurs machines
configurées.

## Pilote

Sélectionnez le pilote correspondant au contrôleur de votre machine. Le
pilote gère la communication entre Rayforge et le matériel.

Après avoir sélectionné un pilote, des paramètres de connexion spécifiques
apparaissent sous le sélecteur (ex. : port série, baud rate). Ils varient
selon le pilote choisi.

:::tip
Une bannière d'erreur en haut de la page vous avertit si le pilote n'est
pas configuré ou rencontre un problème.
:::

## Vitesses et accélération

Ces paramètres contrôlent les vitesses maximales et l'accélération. Ils sont
utilisés pour l'estimation du temps de travail et l'optimisation des
trajectoires.

### Vitesse maximale de déplacement

La vitesse maximale pour les mouvements rapides (sans coupe) lorsque le
laser est éteint et que la tête se déplace vers une nouvelle position.

- **Plage typique** : 2000–5000 mm/min
- **Remarque** : La vitesse réelle est également limitée par les paramètres
  de votre firmware. Ce champ est désactivé si le dialecte G-code
  sélectionné ne prend pas en charge la spécification d'une vitesse de
  déplacement.

### Vitesse maximale de coupe

La vitesse maximale autorisée pendant les opérations de coupe ou de
gravure.

- **Plage typique** : 500–2000 mm/min
- **Remarque** : Certaines opérations peuvent utiliser des vitesses
  inférieures

### Accélération

Le taux d'accélération et de décélération de la machine, utilisé pour les
estimations de temps et le calcul de la distance d'overscan par défaut.

- **Plage typique** : 500–2000 mm/s²
- **Remarque** : Doit correspondre ou être inférieure aux paramètres
  d'accélération du firmware

:::tip
Commencez avec des valeurs de vitesse conservatrices et augmentez-les
progressivement. Observez votre machine pour détecter tout saut de courroie,
calage de moteur ou perte de précision de positionnement.
:::

## Exporter un profil de machine

Cliquez sur l'icône de partage dans la barre d'en-tête de la boîte de
dialogue des paramètres pour exporter la configuration actuelle de la
machine. Choisissez un dossier de destination. Un fichier ZIP est créé,
contenant les paramètres de la machine et son dialecte G-code, qui peut être
partagé avec d'autres utilisateurs ou importé sur un autre système.

## Voir aussi

- [Assistant de Configuration](config-wizard) - Détecter et configurer
  automatiquement un appareil connecté
- [Paramètres matériel](hardware) - Dimensions de la zone de travail et
  configuration des axes
- [Paramètres du périphérique](device) - Lire et écrire les paramètres du
  firmware sur le contrôleur
