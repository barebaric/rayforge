---
description:
  "Allgemeine Maschineneinstellungen in Rayforge konfigurieren — Maschinennamen festlegen, Treiber
  auswählen und Geschwindigkeiten sowie Beschleunigung einstellen."
---

# Allgemeine Einstellungen

Die Seite „Allgemein" in den Maschineneinstellungen enthält den Maschinennamen, die Treiberauswahl
und Verbindungseinstellungen sowie die Geschwindigkeitsparameter.

![Allgemeine Einstellungen](/screenshots/machine-settings-general.webp)

## Maschinenname

Gib deiner Maschine einen beschreibenden Namen. Das hilft, die Maschine im Auswahldropdown zu
erkennen, wenn du mehrere Maschinen konfiguriert hast.

## Treiber

Wähle den Treiber aus, der zum Controller deiner Maschine passt. Der Treiber übernimmt die
Kommunikation zwischen Rayforge und der Hardware.

GRBL-Geräte haben drei serielle Treiber-Optionen:

- **GRBL (Serial)** — Pufferzählender Treiber mit Deadlock-Erkennung und Stall-Wiederherstellung.
  Für die meisten GRBL-Geräte empfohlen
- **GRBL (Serial Simple)** — Ping-Pong-Protokoll-Treiber. Sendet eine Zeile, wartet auf "ok", sendet
  die nächste. Keine Pufferverwaltung, keine Deadlock-Erkennung. Nützlich, wenn der Standardtreiber
  falsche Alarme auslöst
- **GRBL (Rust)** — Experimenteller Treiber, dessen kompletter GRBL-Protokollstack (Flusssteuerung,
  Auftrags-Streaming, Stall-Erkennung, Deadlock-Recovery, Einstellungen und Probe) in Rust läuft.
  Kann als direkter Ersatz für GRBL (Serial) ausgewählt werden

Ruida-basierte Controller werden vom Treiber **Ruida RPA** unterstützt, der sich direkt über USB
oder UDP verbindet oder per TUI-RPC über den Ruida Protocol Analyzer.

### Serieller Port binden

Statt eines Gerätepfads (z. B. `/dev/ttyUSB0` oder `COM3`) akzeptiert das Feld für den seriellen
Port auch eine USB-`VID:PID`-Kennung wie `0403:6001`. Wenn eine Maschine per VID:PID gebunden ist,
folgt die automatische Wiederverbindung ihrem neuen Port, nachdem das Betriebssystem die USB-Geräte
neu aufzählt — z. B. nach einem Neustart oder beim Ab- und Wiederanstecken. Die VID:PID eines Geräts
findest du in der Ausgabe von `lsusb` (Linux) oder im Geräte-Manager unter Hardware-IDs (Windows).

Nach der Auswahl eines Treibers werden verbindungsspezifische Einstellungen unter der Auswahl
angezeigt (z. B. serieller Port, Baudrate). Diese variieren je nach gewähltem Treiber.

<!-- prettier-ignore-start -->
:::tip
Ein Fehlerbanner oben auf der Seite warnt dich, wenn der Treiber nicht konfiguriert ist oder
ein Problem auftritt.
:::
<!-- prettier-ignore-end -->

## Geschwindigkeiten & Beschleunigung

Diese Einstellungen steuern die maximalen Geschwindigkeiten und die Beschleunigung. Sie werden für
die Arbeitszeit­schätzung und die Pfadoptimierung verwendet.

### Maximale Eilganggeschwindigkeit

Die maximale Geschwindigkeit für schnelle (nicht schneidende) Bewegungen, wenn der Laser aus ist und
der Kopf zu einer neuen Position fährt.

- **Typischer Bereich**: 2000–5000 mm/min
- **Hinweis**: Die tatsächliche Geschwindigkeit wird auch durch deine Firmware-Einstellungen
  begrenzt. Dieses Feld ist deaktiviert, wenn der gewählte G-Code-Dialekt keine Angabe der
  Eilganggeschwindigkeit unterstützt.

### Maximale Schnittgeschwindigkeit

Die maximale Geschwindigkeit, die beim Schneiden oder Gravieren erlaubt ist.

- **Typischer Bereich**: 500–2000 mm/min
- **Hinweis**: Einzelne Operationen können niedrigere Geschwindigkeiten verwenden

### Beschleunigung

Die Rate, mit der die Maschine beschleunigt und abbremst. Wird für Zeitschätzungen und zur
Berechnung des Standard-Overscan-Abstands verwendet.

- **Typischer Bereich**: 500–2000 mm/s²
- **Hinweis**: Muss mit den Firmware-Beschleunigungseinstellungen übereinstimmen oder niedriger sein

<!-- prettier-ignore-start -->
:::tip
Beginne mit konservativen Geschwindigkeitswerten und steigere sie schrittweise. Beobachte
deine Maschine auf Zahnriemensprünge, Motorblockaden oder Positionsverlust.
:::
<!-- prettier-ignore-end -->

## Maschinenprofil exportieren

Klicke auf das Teilen-Symbol in der Kopfzeile des Einstellungsdialogs, um die aktuelle
Maschinenkonfiguration zu exportieren. Wähle einen Ordner zum Speichern. Es wird eine ZIP-Datei
erstellt, die die Maschineneinstellungen und den G-Code-Dialekt enthält. Diese kann mit anderen
Nutzern geteilt oder auf einem anderen System importiert werden.

## Siehe auch

- [Ersteinrichtung](../getting-started/first-time-setup.md) – Eine Maschine Schritt für Schritt mit
  dem Konfigurations-Assistenten erstellen
- [Hardware-Einstellungen](hardware) – Arbeitsflächenabmessungen und Achsenkonfiguration
- [Geräte-Einstellungen](device) – Firmware-Einstellungen auf dem Controller lesen und schreiben
