---
description:
  "Connectez les découpeuses laser xTool D1 et D1 Pro et la xTool M2 en Wi-Fi avec les pilotes xTool
  expérimentaux de Rayforge."
---

# xTool en Wi-Fi

Rayforge inclut des pilotes expérimentaux pour les machines xTool connectées en Wi-Fi. Ils utilisent
l'API réseau propriétaire d'xTool plutôt que les interfaces GRBL Telnet ou FluidNC / ESP3D ; la
machine doit donc être configurée avec son propre pilote xTool.

## Machines compatibles

| Profil           | Pilote           | Fonctions                                                                                               |
| ---------------- | ---------------- | ------------------------------------------------------------------------------------------------------- |
| **xTool D1 Pro** | xTool (Wi-Fi)    | Envoyer et lancer des tâches, référencer, déplacer, mettre en pause, reprendre et annuler               |
| **xTool M2**     | xTool M2 (Wi-Fi) | Surveiller la machine, référencer, déplacer et mettre en pause, reprendre ou annuler une tâche préparée |

Les deux pilotes sont marqués **expérimental**. Testez-les sur une chute de matériau avant de les
utiliser en production, et signalez les problèmes avec un journal de débogage joint.

## Configurer une machine xTool

1. Connectez la machine au même réseau que votre ordinateur et relevez son adresse IP (par exemple
   dans l'application xTool ou dans la liste des appareils du routeur).
2. Ajoutez une machine et choisissez le profil **xTool D1 Pro** ou **xTool M2**. Le profil
   préconfigure le pilote et les dimensions de la machine.
3. Saisissez le **Nom d'hôte ou adresse IP** de la machine.
4. Laissez les ports à leurs valeurs par défaut, sauf si votre réseau impose autre chose :
   - D1 Pro : port HTTP `8080`, port WebSocket `8081`
   - M2 : port `28900`
5. Connectez.

Si Rayforge indique qu'aucune machine xTool n'a répondu, vérifiez l'adresse IP, que la machine est
allumée et que l'ordinateur et la machine sont sur le même réseau.

## D1 et D1 Pro

Le pilote **xTool (Wi-Fi)** envoie la tâche à la machine, la lance et suit sa progression via les
événements WebSocket. Le référencement, les déplacements manuels et absolus, la pause, la reprise et
l'annulation fonctionnent comme d'habitude.

L'API réseau de la machine ne rapporte pas sa position ; Rayforge déduit donc l'affichage de la
position de l'état de la tâche. Le référencement n'est disponible que pour tous les axes à la fois ;
le référencement par axe est désactivé. L'écriture des réglages firmware, des décalages de
coordonnées de travail et le palpage ne sont pas pris en charge par ce pilote.

## xTool M2

Le pilote **xTool M2 (Wi-Fi)** se connecte au service TLS du M2 sur le port `28900`. Il surveille
l'état de la machine et prend en charge le référencement, les déplacements manuels et la pause, la
reprise ou l'annulation d'une tâche déjà préparée sur la machine.

L'API réseau du M2 ne peut pas encore recevoir de G-code ; Rayforge ne peut donc pas lui envoyer ses
propres tâches. Préparez la tâche dans xTool Studio, lancez-la là-bas et utilisez Rayforge pour la
surveiller et la contrôler. Les mouvements Z, les changements d'outil, le contrôle à distance de la
puissance laser, les réglages firmware et les décalages de coordonnées de travail ne sont pas pris
en charge.

## Certificats TLS

Le pilote du M2 valide le certificat TLS de la machine. Si la machine utilise un certificat
auto-signé, activez **Autoriser les certificats TLS non valides** dans les réglages du pilote. Ne le
faites que sur un réseau de confiance.

## Voir aussi

- [Réglages généraux](general) - Sélection du pilote, nom d'hôte et ports
- [Référence firmware](../reference/firmware.md) - Contrôleurs et pilotes pris en charge
- [Appareils distants](remote) - Connexion à une machine via un daemon raydriver
