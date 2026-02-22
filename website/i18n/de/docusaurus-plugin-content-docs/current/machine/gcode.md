# G-Code-Einstellungen

Die G-Code-Seite in den Maschineneinstellungen konfiguriert, wie Rayforge G-Code für Ihre Maschine generiert.

![G-Code-Einstellungen](/screenshots/machine-gcode.png)

## G-Code-Dialekt

Wählen Sie den G-Code-Dialekt, der mit Ihrer Controller-Firmware übereinstimmt. Verschiedene Controller verwenden leicht unterschiedliche Befehle und Formate.

### Verfügbare Dialekte

- **GRBL**: Am häufigsten bei Hobby-Laserschneidern. Verwendet M3/M5 zur Lasersteuerung.
- **Smoothieware**: Für Smoothieboard und ähnliche Controller.
- **Marlin**: Für Marlin-basierte Controller.
- **GRBL-kompatibel**: Für Controller, die größtenteils der GRBL-Syntax folgen.

:::info
Der Dialekt beeinflusst, wie Laserleistung, Bewegungen und andere Befehle im Ausgabe-G-Code formatiert werden.
:::

## Benutzerdefinierter G-Code

Sie können den G-Code anpassen, den Rayforge an bestimmten Punkten im Job generiert.

### Programmstart

G-Code-Befehle, die am Anfang jedes Jobs ausgeführt werden, vor allen Schneideoperationen.

Häufige Verwendungen:
- Einheiten festlegen (G21 für mm)
- Positionierungsmodus festlegen (G90 für absolut)
- Maschinenzustand initialisieren

### Programmende

G-Code-Befehle, die am Ende jedes Jobs ausgeführt werden, nach allen Schneideoperationen.

Häufige Verwendungen:
- Laser ausschalten (M5)
- Zum Ursprung zurückkehren (G0 X0 Y0)
- Kopf parken

### Werkzeugwechsel

G-Code-Befehle, die beim Wechsel zwischen Laserköpfen ausgeführt werden (für Multi-Laser-Maschinen).

## Siehe auch

- [G-Code-Grundlagen](../general-info/gcode-basics) - G-Code verstehen
- [G-Code-Dialekte](../reference/gcode-dialects) - Detaillierte Dialekt-Unterschiede
- [Hooks & Makros](hooks-macros) - Benutzerdefinierte G-Code-Einschubpunkte
