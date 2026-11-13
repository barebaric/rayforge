---
description:
  "Betreiben Sie Ihre Maschine über einen kleinen Computer direkt neben der Hardware, während
  Rayforge woanders läuft. Jobs überstehen Verbindungsabbrüche, und beim Wiederverbinden wird der
  Fortschritt exakt synchronisiert."
---

# Remote-Geräte

Betreiben Sie Ihre Maschine über einen kleinen Computer direkt neben der Hardware (zum Beispiel
einen Raspberry Pi in der Werkstatt), während Rayforge auf Ihrem Laptop oder Desktop läuft. Jobs
laufen auf dem Geräte-Host und laufen weiter, selbst wenn Rayforge die Verbindung trennt; beim
Wiederverbinden wird der Fortschritt exakt synchronisiert.

## So funktioniert es

- Der **raydriver-Daemon** hostet die Maschinenverbindung direkt neben der Hardware.
- Rayforge behält Ihre Maschinenkonfiguration und übermittelt Jobs über das LAN.
- Jeder G-Code-Treiber (GRBL seriell, GRBL Telnet, GRBL einfach, Smoothie, Marlin) funktioniert
  lokal oder remote — Remote ist eine Verbindungseinstellung, kein eigener Treiber.

## Geräte-Host einrichten

1. Installieren Sie raydriver auf dem Geräte-Host: `pip install raydriver` (mit `[daemon]` für die
   Netzwerk-Erkennung).
2. Führen Sie `raydriver daemon --setup` aus. Das schreibt `~/.config/raydriver/daemon.json` mit
   einem generierten Token und gibt den Befehl für das Zertifikat aus.
3. Erstellen Sie für den LAN-Zugriff ein Zertifikat mit dem ausgegebenen `openssl`-Einzeiler und
   setzen Sie `tls_cert` und `tls_key` in der Konfiguration. Ein Daemon auf einer
   Nicht-Loopback-Adresse startet ohne Token und TLS nicht.
4. Führen Sie `raydriver daemon install` aus, um den Hintergrunddienst zu installieren und zu
   starten (systemd unter Linux, launchd unter macOS, eine Anmelde-Aufgabe unter Windows), oder
   starten Sie `raydriver daemon` im Vordergrund.

## Rayforge verbinden

- Der Maschinen-Assistent erkennt Daemons im Netzwerk und listet ihre Geräte auf.
- Alternativ aktivieren Sie **Remote-Gerät** in den Verbindungseinstellungen des Treibers und geben
  Hostname, Port, Token und TLS-Zertifikat des Daemons ein.
- Das Sondieren funktioniert über die Remote-Verbindung, sodass der Assistent Hardware erkennen
  kann, die nicht an Ihrem Computer angeschlossen ist.

## Jobs und Wiederverbinden

- Ein übermittelter Job läuft auf dem Geräte-Host bis zum Ende, selbst wenn Rayforge geschlossen
  wird oder das Netzwerk abbricht.
- Verbinden Sie sich erneut, um den exakten Fortschritt zu sehen. Jeweils ein Client hält die
  Kontrolle; andere können zusehen.

## Kameras auf dem Geräte-Host

Video wird nicht über die Steuerungsverbindung übertragen. Verwenden Sie Rayforges
[Netzwerkkameras](camera.md) mit einem Stream, der auf dem Geräte-Host erreichbar ist:

- **IP-Kamera**: Fügen Sie ihre Stream-URL direkt hinzu.
- **USB-Kamera am Geräte-Host**: Betreiben Sie einen Standard-Streamer wie `ustreamer` und fügen Sie
  dessen MJPEG-URL als HTTP-Stream-Kamera hinzu.

Der Daemon bewirbt konfigurierte Kamera-Endpunkte, sodass der Assistent die URL vorausfüllen kann.

## Sicherheitshinweise

- Nur LAN (oder Tunnel) — keine Cloud, keine Konten.
- Token und TLS sind für jeden Nicht-Loopback-Daemon Pflicht.
- Halten Sie Kamera-Streams im LAN oder hinter einer Firewall; sie sind vom TLS-Steuerkanal
  getrennt.
