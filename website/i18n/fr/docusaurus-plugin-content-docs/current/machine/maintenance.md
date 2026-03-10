# Maintenance

La page Maintenance dans les Paramètres Machine vous aide à suivre l'utilisation de la machine et à planifier les tâches de maintenance.

![Paramètres de Maintenance](/screenshots/machine-maintenance.png)

## Suivi d'Utilisation

Rayforge suit la durée d'utilisation de votre machine. Ces informations vous aident à planifier la maintenance préventive à des intervalles appropriés.

### Heures Totales

Le compteur d'heures totales suit tout le temps passé à exécuter des travaux sur la machine. Ce compteur cumulatif ne peut pas être réinitialisé et fournit un historique complet de l'utilisation de la machine.

Utilisez-le pour suivre l'âge global de la machine et planifier les intervalles de service majeurs.

## Compteurs de Maintenance Personnalisés

Vous pouvez créer des compteurs personnalisés pour suivre des intervalles de maintenance spécifiques. Chaque compteur a un nom, suit les heures et peut être configuré avec un seuil de notification.

### Créer un Compteur

1. Cliquez sur le bouton ajouter pour créer un nouveau compteur
2. Entrez un nom descriptif (ex : "Tube Laser", "Tension Courroie", "Nettoyage Miroir")
3. Définissez un seuil de notification en heures si souhaité

### Fonctionnalités du Compteur

- **Noms personnalisés** : Étiquetez les compteurs pour toute tâche de maintenance
- **Suivi des heures** : Accumule automatiquement le temps pendant l'exécution des travaux
- **Seuils de notification** : Recevez des rappels quand la maintenance est nécessaire
- **Capacité de réinitialisation** : Réinitialisez les compteurs après avoir effectué la maintenance

### Exemples de Compteurs

**Tube Laser** : Suivez les heures du tube CO2 pour planifier le remplacement (généralement 1000-3000 heures). Définissez une notification à 2500 heures pour planifier à l'avance.

**Tension Courroie** : Suivez les heures depuis la dernière tension de courroie. Réinitialisez après avoir effectué la maintenance.

**Nettoyage Miroir** : Suivez l'utilisation depuis le dernier nettoyage de miroir. Réinitialisez après le nettoyage.

**Lubrification Roulements** : Suivez les heures pour les intervalles de maintenance des roulements.

## Réinitialiser les Compteurs

Après avoir effectué la maintenance, vous pouvez réinitialiser le compteur pertinent :

1. Cliquez sur le bouton de réinitialisation à côté du compteur
2. Confirmez la réinitialisation dans la boîte de dialogue
3. Le compteur revient à zéro

:::tip Calendrier de Maintenance
Intervalles de maintenance courants :
- **Quotidien** : Nettoyer la lentille, vérifier l'alignement des miroirs
- **Hebdomadaire** : Nettoyer les rails, vérifier la tension des courroies
- **Mensuel** : Lubrifier les roulements, vérifier les connexions électriques
- **Annuel** : Inspection complète, remplacer les pièces usées

Ajustez les intervalles en fonction de vos modèles d'utilisation et des recommandations du fabricant.
:::

## Voir Aussi

- [Paramètres Laser](laser) - Configuration de la tête laser
- [Paramètres Matériels](hardware) - Dimensions de la machine
