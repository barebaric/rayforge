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
- Rayforge conserve la configuration de votre machine et soumet les jobs via le LAN.
- Chaque pilote G-code (GRBL série, GRBL telnet, GRBL simple, Smoothie, Marlin) fonctionne en local
  ou à distance — le distant est un réglage de connexion, pas un pilote séparé.

## Configurer l'hôte de l'appareil

1. Installez raydriver sur l'hôte de l'appareil : `pip install raydriver` (ajoutez `[daemon]` pour
   la découverte réseau).
2. Exécutez `raydriver daemon --setup`. Cela écrit `~/.config/raydriver/daemon.json` avec un jeton
   généré et affiche la commande du certificat.
3. Pour l'accès LAN, créez un certificat avec la commande `openssl` affichée et renseignez
   `tls_cert` et `tls_key` dans la configuration. Un démon sur une adresse autre que loopback refuse
   de démarrer sans jeton et TLS.
4. Lancez `raydriver daemon`, ou exécutez-le comme service systemd (une unité d'exemple est fournie
   dans la documentation de raydriver).

## Connecter Rayforge

- L'assistant de machine découvre les démons sur le réseau et liste leurs appareils.
- Vous pouvez aussi activer **Appareil distant** dans les réglages de connexion du pilote et saisir
  le nom d'hôte, le port, le jeton et le certificat TLS du démon.
- La sonde fonctionne via la connexion distante : l'assistant peut donc identifier un matériel qui
  n'est pas branché sur votre ordinateur.

## Jobs et reconnexions

- Un job soumis s'exécute jusqu'au bout sur l'hôte de l'appareil, même si Rayforge est fermé ou si
  le réseau tombe.
- Reconnectez-vous pour voir la progression exacte. Un seul client à la fois détient le contrôle ;
  les autres peuvent observer.

## Caméras sur l'hôte de l'appareil

La vidéo ne transite pas par la connexion de contrôle. Utilisez les [caméras réseau](camera.md) de
Rayforge avec un flux accessible sur l'hôte de l'appareil :

- **Caméra IP** : ajoutez directement son URL de flux.
- **Caméra USB sur l'hôte de l'appareil** : lancez un streamer standard comme `ustreamer` et ajoutez
  son URL MJPEG comme caméra de flux HTTP.

Le démon annonce les points de caméra configurés afin que l'assistant puisse préremplir l'URL.

## Notes de sécurité

- LAN (ou tunnel) uniquement — pas de cloud, pas de compte.
- Le jeton et TLS sont obligatoires pour tout démon non loopback.
- Gardez les flux de caméra en LAN uniquement ou derrière un pare-feu ; ils sont séparés du canal de
  contrôle TLS.
