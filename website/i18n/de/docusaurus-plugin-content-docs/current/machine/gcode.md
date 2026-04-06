# G-Code-Einstellungen

Die G-Code-Seite in den Maschineneinstellungen konfiguriert, wie Rayforge G-Code für deine Maschine generiert.

![G-Code-Einstellungen](/screenshots/machine-gcode.png)

## G-Code-Dialekt

Wähle den G-Code-Dialekt, der mit deiner Controller-Firmware übereinstimmt. Verschiedene Controller verwenden leicht unterschiedliche Befehle und Formate.

### Verfügbare Dialekte

- **Grbl (Compat)**: Standard-GRBL-Dialekt für Hobby-Laserschneider. Verwendet M3/M5 zur Lasersteuerung.
- **Grbl (Compat, no Z axis)**: Wie Grbl (Compat), aber ohne Z-Achsen-Befehle. Für reine 2D-Maschinen.
- **GRBL Dynamic**: Verwendet GRBLs dynamischen Laserleistungsmodus für variablen Leistungsgravur.
- **GRBL Dynamic (no Z axis)**: Dynamischer Modus ohne Z-Achsen-Befehle.
- **Smoothieware**: Für Smoothieboard und ähnliche Controller.
- **Marlin**: Für Marlin-basierte Controller.

:::info
Der Dialekt beeinflusst, wie Laserleistung, Bewegungen und andere Befehle im Ausgabe-G-Code formatiert werden.
:::

## Dialekt-Präambel und Postscript

Jeder Dialekt enthält anpassbare Präambel- und Postscript-G-Codes, die am Anfang und Ende von Jobs ausgeführt werden.

### Präambel

G-Code-Befehle, die am Anfang jedes Jobs ausgeführt werden, vor allen Schneideoperationen. Häufige Verwendungen umfassen das Festlegen von Einheiten (G21 für mm), Positionierungsmodus (G90 für absolut) und Initialisierung des Maschinenzustands.

### Postscript

G-Code-Befehle, die am Ende jedes Jobs ausgeführt werden, nach allen Schneideoperationen. Häufige Verwendungen umfassen das Ausschalten des Lasers (M5), Rückkehr zum Ursprung (G0 X0 Y0) und Parken des Kopfes.

## Siehe auch

- [G-Code-Grundlagen](../general-info/gcode-basics) - G-Code verstehen
- [G-Code-Dialekte](../reference/gcode-dialects) - Detaillierte Dialekt-Unterschiede
- [Hooks & Makros](hooks-macros) - Benutzerdefinierte G-Code-Einschubpunkte
