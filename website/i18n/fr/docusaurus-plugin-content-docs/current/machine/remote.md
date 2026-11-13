---
description:
  "Pilotez votre machine depuis un petit ordinateur placé près du matériel pendant que Rayforge
  tourne ailleurs. Les jobs survivent aux déconnexions et la reconnexion resynchronise la
  progression exactement."
---

# Appareils distants

Pilotez votre machine depuis un petit ordinateur placé près du matériel (par exemple un Raspberry Pi
dans l'atelier) pendant que Rayforge tourne sur votre portable ou votre ordinateur de bureau. Les
jobs s'exécutent sur l'hôte de l'appareil et continuent même si Rayforge se déconnecte ; la
reconnexion resynchronise la progression exactement.

## Fonctionnement

- Le **démon raydriver** héberge la connexion de la machine près du matériel.
- Rayforge se connecte à un _appareil sur le démon_ — il n'a pas besoin de savoir quel protocole
  parle l'appareil. Le démon le résout et indique à Rayforge ce que l'appareil peut faire.
- Les machines distantes utilisent un unique pilote **Remote machine (raydriver)**. Les pilotes
  spécifiques au protocole (GRBL, Smoothie, Marlin) sont destinés au matériel connecté directement à
  votre ordinateur.

## Configurer l'hôte de l'appareil

1. Installez raydriver sur l'hôte de l'appareil : `pip install raydriver` (ajoutez `[daemon]` pour
   la découverte réseau).
2. Exécutez `raydriver daemon add /dev/ttyUSB0` (utilisez le port série de votre machine ;
   `raydriver scan` les liste). Cela crée la configuration avec un jeton généré et enregistre
   l'appareil. Ajoutez `--protocol`/`--baudrate` pour un firmware non standard.
3. Exécutez `raydriver daemon install` pour installer et démarrer le service d'arrière-plan (systemd
   sous Linux, launchd sous macOS, une tâche de connexion sous Windows), ou exécutez
   `raydriver daemon` au premier plan.
4. Pour l'accès LAN, créez un certificat avec la commande `openssl` sur une ligne affichée par
   `raydriver daemon --setup` et définissez `tls_cert` et `tls_key` dans la configuration. Un démon
   sur une adresse autre que loopback refuse de démarrer sans jeton et TLS.

Consultez le
[guide du démon raydriver](https://github.com/barebaric/raydriver/blob/main/docs/daemon.md) pour la
référence complète.

## Connecter Rayforge

- L'assistant de machine découvre les démons sur le réseau et liste leurs appareils ; choisissez-en
  un pour l'ajouter.
- Ou ajoutez une machine manuellement avec le pilote **Remote machine (raydriver)** : saisissez le
  nom d'hôte, le port et le jeton du démon, puis choisissez l'appareil dans la liste lue depuis le
  démon.
- Le sondage fonctionne via la connexion distante, l'assistant peut donc identifier du matériel qui
  n'est pas connecté à votre ordinateur.

## Jobs et reconnexions

- Un job soumis s'exécute jusqu'au bout sur l'hôte de l'appareil même si Rayforge se ferme ou si le
  réseau tombe.
- Reconnectez-vous pour voir la progression exacte. Un client à la fois détient le contrôle ; les
  autres peuvent observer.

## Caméras sur l'hôte de l'appareil

La vidéo ne transite pas par la connexion de contrôle. Utilisez les [caméras réseau](camera.md) de
Rayforge avec un flux accessible sur l'hôte de l'appareil :

- **Caméra IP** : ajoutez directement son URL de flux.
- **Caméra USB sur l'hôte de l'appareil** : exécutez un streamer standard tel que `ustreamer` et
  ajoutez son URL MJPEG comme caméra de flux HTTP.

Le démon annonce les points de terminaison de caméra configurés afin que l'assistant puisse
préremplir l'URL.

## Notes de sécurité

- LAN (ou tunnel) uniquement — pas de cloud, pas de comptes.
- Le jeton et TLS sont obligatoires pour tout démon non loopback.
- Gardez les flux de caméra en LAN uniquement ou derrière un pare-feu ; ils sont séparés du canal de
  contrôle TLS.
