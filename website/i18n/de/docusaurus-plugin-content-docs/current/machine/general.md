---
description: "Allgemeine Maschineneinstellungen in Rayforge konfigurieren — Maschinennamen festlegen, Treiber auswählen und Geschwindigkeiten sowie Beschleunigung einstellen."
---

# Allgemeine Einstellungen

Die Seite „Allgemein" in den Maschineneinstellungen enthält den Maschinennamen,
die Treiberauswahl und Verbindungseinstellungen sowie die Geschwindigkeitsparameter.

![Allgemeine Einstellungen](/screenshots/machine-general.png)

## Maschinenname

Gib deiner Maschine einen beschreibenden Namen. Das hilft, die Maschine im
Auswahldropdown zu erkennen, wenn du mehrere Maschinen konfiguriert hast.

## Treiber

Wähle den Treiber aus, der zum Controller deiner Maschine passt. Der Treiber
übernimmt die Kommunikation zwischen Rayforge und der Hardware.

Nach der Auswahl eines Treibers werden verbindungsspezifische Einstellungen
unter der Auswahl angezeigt (z. B. serieller Port, Baudrate). Diese variieren
je nach gewähltem Treiber.

:::tip
Ein Fehlerbanner oben auf der Seite warnt dich, wenn der Treiber nicht
konfiguriert ist oder ein Problem auftritt.
:::

## Geschwindigkeiten & Beschleunigung

Diese Einstellungen steuern die maximalen Geschwindigkeiten und die
Beschleunigung. Sie werden für die Arbeitszeit­schätzung und die
Pfadoptimierung verwendet.

### Maximale Eilganggeschwindigkeit

Die maximale Geschwindigkeit für schnelle (nicht schneidende) Bewegungen,
wenn der Laser aus ist und der Kopf zu einer neuen Position fährt.

- **Typischer Bereich**: 2000–5000 mm/min
- **Hinweis**: Die tatsächliche Geschwindigkeit wird auch durch deine
  Firmware-Einstellungen begrenzt. Dieses Feld ist deaktiviert, wenn der
  gewählte G-Code-Dialekt keine Angabe der Eilganggeschwindigkeit
  unterstützt.

### Maximale Schnittgeschwindigkeit

Die maximale Geschwindigkeit, die beim Schneiden oder Gravieren erlaubt ist.

- **Typischer Bereich**: 500–2000 mm/min
- **Hinweis**: Einzelne Operationen können niedrigere Geschwindigkeiten
  verwenden

### Beschleunigung

Die Rate, mit der die Maschine beschleunigt und abbremst. Wird für
Zeitschätzungen und zur Berechnung des Standard-Overscan-Abstands verwendet.

- **Typischer Bereich**: 500–2000 mm/s²
- **Hinweis**: Muss mit den Firmware-Beschleunigungseinstellungen
  übereinstimmen oder niedriger sein

:::tip
Beginne mit konservativen Geschwindigkeitswerten und steigere sie schrittweise.
Beobachte deine Maschine auf Zahnriemensprünge, Motorblockaden oder
Positionsverlust.
:::

## Maschinenprofil exportieren

Klicke auf das Teilen-Symbol in der Kopfzeile des Einstellungsdialogs, um die
aktuelle Maschinenkonfiguration zu exportieren. Wähle einen Ordner zum Speichern.
Es wird eine ZIP-Datei erstellt, die die Maschineneinstellungen und den
G-Code-Dialekt enthält. Diese kann mit anderen Nutzern geteilt oder auf einem
anderen System importiert werden.

## Siehe auch

- [Hardware-Einstellungen](hardware) – Arbeitsflächenabmessungen und
  Achsenkonfiguration
- [Geräte-Einstellungen](device) – Firmware-Einstellungen auf dem Controller
  lesen und schreiben
