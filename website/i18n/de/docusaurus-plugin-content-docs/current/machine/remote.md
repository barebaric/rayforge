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
- Rayforge verbindet sich mit einem _Gerät am Daemon_ — es muss nicht wissen, welches Protokoll das
  Gerät spricht. Der Daemon ermittelt das und teilt Rayforge mit, was das Gerät kann.
- Remote-Maschinen verwenden einen einzigen Treiber **Remote machine (raydriver)**. Die
  protokollspezifischen Treiber (GRBL, Smoothie, Marlin) sind für Hardware gedacht, die direkt an
  Ihrem Computer angeschlossen ist.

## Geräte-Host einrichten

1. Installieren Sie raydriver auf dem Geräte-Host: `pip install raydriver`. Die Netzwerk-Erkennung
   ist enthalten.
2. Führen Sie `raydriver daemon add /dev/ttyUSB0` aus (verwenden Sie den seriellen Port Ihrer
   Maschine; `raydriver scan` listet sie auf). Dies legt die Konfiguration mit einem generierten
   **Token** an und registriert das Gerät. Für nicht standardmäßige Firmware
   `--protocol`/`--baudrate` ergänzen.
3. Führen Sie `raydriver daemon install` aus, um den Hintergrunddienst zu installieren und zu
   starten (systemd unter Linux, launchd unter macOS, eine Anmelde-Aufgabe unter Windows), oder
   starten Sie `raydriver daemon` im Vordergrund.
4. Für den LAN-Zugriff erstellen Sie ein Zertifikat mit dem von `raydriver daemon --setup`
   ausgegebenen `openssl`-Einzeiler und setzen `tls_cert` und `tls_key` in der Konfiguration. Ein
   Daemon auf einer Nicht-Loopback-Adresse startet nicht ohne Token und TLS.

Die vollständige Referenz finden Sie im
[raydriver-Daemon-Handbuch](https://github.com/barebaric/raydriver/blob/main/docs/daemon.md).

### Token abrufen

Das Token ist das gemeinsame Geheimnis, mit dem sich Rayforge verbindet. Es wird beim Hinzufügen des
ersten Geräts generiert und einmalig ausgegeben. Um es erneut auszugeben, führen Sie dies **auf dem
Geräte-Host** aus:

```bash
raydriver daemon status
```

Kopieren Sie den Wert hinter `token:` — Sie fügen ihn unten in Rayforge ein.

## Rayforge verbinden

Der Maschinen-Assistent erkennt Daemons im Netzwerk und listet ihre Geräte; wählen Sie eines aus, um
es hinzuzufügen. Alternativ fügen Sie eine Maschine manuell mit dem Treiber **Remote machine
(raydriver)** hinzu und geben Daemon-Hostname, Port und Token ein; wählen Sie dann das Gerät aus der
Liste, die vom Daemon gelesen wird:

![Assistent — Remote-Verbindung](/screenshots/remote-wizard-connect.webp)

- **Daemon-Hostname / Port** — der Computer, auf dem der Daemon läuft (Standardport `8765`).
- **Token** — der Wert aus `raydriver daemon status` auf dem Geräte-Host.
- **Gerät** — die Maschine am Daemon, aus dessen Geräteliste gelesen, sobald Hostname und Token
  gesetzt sind.
- Die TLS-Felder erscheinen nur für einen Daemon im Netzwerk; ein Daemon auf demselben Computer
  (`127.0.0.1`) benötigt kein Zertifikat.

Das Probing funktioniert über die Remote-Verbindung, sodass der Assistent Hardware identifizieren
kann, die nicht an Ihrem Computer angeschlossen ist.

## Maschinen-Einstellungen

Die Einstellungen einer Remote-Maschine zeigen den Daemon-Endpunkt und das Gerät statt eines lokalen
seriellen Ports:

![Maschinen-Einstellungen — Remote-Maschine](/screenshots/remote-settings.webp)

## Jobs und Wiederverbindungen

- Ein übermittelter Job läuft auf dem Geräte-Host bis zum Ende, selbst wenn Rayforge geschlossen
  wird oder das Netzwerk abbricht.
- Verbinden Sie sich erneut, um den exakten Fortschritt zu sehen. Jeweils ein Client hält die
  Kontrolle; andere können zusehen.

## Kameras auf dem Geräte-Host

Video wird nicht über die Steuerungsverbindung übertragen. Verwenden Sie Rayforges
[Netzwerkkameras](camera.md) mit einem Stream, der auf dem Geräte-Host erreichbar ist:

- **IP-Kamera**: Fügen Sie ihre Stream-URL direkt hinzu.
- **USB-Kamera am Geräte-Host**: Betreiben Sie einen Standard-Streamer wie `ustreamer` und fügen Sie
  seine MJPEG-URL als HTTP-Stream-Kamera hinzu.

Der Daemon bewirbt konfigurierte Kamera-Endpunkte, sodass der Assistent die URL vorausfüllen kann.

## Sicherheitshinweise

- Nur LAN (oder Tunnel) — keine Cloud, keine Konten.
- Token und TLS sind für jeden Nicht-Loopback-Daemon Pflicht.
- Halten Sie Kamera-Streams im LAN oder hinter einer Firewall; sie sind vom TLS-Steuerungskanal
  getrennt.
