# Simulationsmodus

![Simulationsmodus](/screenshots/main-simulation.png)

Der Simulationsmodus zeigt, wie dein Laserjob ausgeführt wird, bevor du ihn auf der Maschine startest. Du kannst den G-Code schrittweise durchgehen und genau sehen, was passieren wird.

## Simulationsmodus aktivieren

- **Tastatur**: Drücke <kbd>F11</kbd>
- **Menü**: Gehe zu **Ansicht → Ausführung simulieren**
- **Symbolleiste**: Klicke auf den Simulations-Button

## Visualisierung

### Geschwindigkeits-Heatmap

Operationen werden nach Geschwindigkeit eingefärbt:

| Geschwindigkeit | Farbe |
| --------------- | ----- |
| Am langsamsten  | Blau  |
| Langsam         | Cyan  |
| Mittel          | Grün  |
| Schnell         | Gelb  |
| Am schnellsten  | Rot   |

Die Farben sind relativ zum Geschwindigkeitsbereich deines Jobs - Blau ist das Minimum, Rot das Maximum.

### Leistungstransparenz

Die Liniendeckkraft zeigt die Laserleistung:

- **Faint lines** = Niedrige Leistung (Fahrwege, leichtes Gravieren)
- **Fest gezeichnete Linien** = Hohe Leistung (Schneiden)

## Wiedergabesteuerung

Verwende die Steuerelemente unten auf der Leinwand:

- **Wiedergabe/Pause** (<kbd>Leertaste</kbd>): Automatische Wiedergabe starten oder stoppen
- **Fortschrittsregler**: Ziehen, um durch den Job zu scrollen
- **Pfeiltasten**: Anweisungen einzeln durchgehen

Die Simulation und die G-Code-Ansicht bleiben synchronisiert - das Durchgehen der
Simulation hebt die entsprechende G-Code-Zeile hervor, und das Klicken auf eine
G-Code-Zeile springt die Simulation zu diesem Punkt.

Die 3D-Ansicht hat ebenfalls eine eigene Simulation mit synchronisierter Wiedergabe.
Das Durchgehen des 3D-Simulators hebt die passende Zeile im G-Code-Viewer hervor
und umgekehrt.

## Bearbeiten während der Simulation

Du kannst Werkstücke während der Simulation bearbeiten. Verschiebe, skaliere oder drehe Objekte, und die Simulation wird automatisch aktualisiert.

## Verwandte Themen

- **[3D-Vorschau](../ui/3d-preview)** - 3D-Werkzeugweg-Visualisierung
- **[Materialtest-Raster](operations/material-test-grid)** - Verwende die Simulation zum Validieren von Tests
