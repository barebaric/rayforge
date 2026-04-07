# 3D-Ansicht

Die 3D-Ansicht ermöglicht dir, deine G-Code-Werkzeugwege zu visualisieren und
die Auftragsausführung zu simulieren, bevor du sie an deine Maschine sendest.

![3D-Vorschau](/screenshots/main-3d.png)

## 3D-Ansicht öffnen

Zugriff auf die 3D-Ansicht:

- **Menü**: Ansicht → 3D-Ansicht
- **Tastatur**: <kbd>F12</kbd>

## Navigation

### Maussteuerung

- **Drehen**: Linksklick und ziehen
- **Verschieben**: Rechtsklick und ziehen, oder Mittelklick und ziehen
- **Zoom**: Mausrad, oder <kbd>Strg</kbd> + Linksklick und ziehen

### Ansicht-Presets

Schnelle Kamerawinkel:

- **Draufsicht** (<kbd>1</kbd>): Vogelperspektive
- **Vorderansicht** (<kbd>2</kbd>): Frontale Ansicht
- **Rechte Ansicht** (<kbd>3</kbd>): Rechte Seitenansicht
- **Linke Ansicht** (<kbd>4</kbd>): Linke Seitenansicht
- **Rückansicht** (<kbd>5</kbd>): Hintere Ansicht
- **Isometrisch** (<kbd>7</kbd>): 3D-isometrische Ansicht

## Werkstückkoordinatensystem-Anzeige

Die 3D-Ansicht visualisiert das aktive Werkstückkoordinatensystem (WCS)
anders als die 2D-Canvas:

### Raster und Achsen

- **Isolierte Anzeige**: Das Raster und die Achsen erscheinen so, als wäre
  der WCS-Ursprung der Weltursprung
- **Offset angewendet**: Das gesamte Raster ist verschoben, um mit dem
  ausgewählten WCS-Offset ausgerichtet zu sein
- **Beschriftungen relativ zum WCS**: Koordinatenbeschriftungen zeigen
  Positionen relativ zum WCS-Ursprung, nicht zum Maschinenursprung

Diese "in Isolation"-Anzeige erleichtert es zu verstehen, wo dein Auftrag
relativ zum ausgewählten Werkstückkoordinatensystem laufen wird, ohne durch
die absolute Position der Maschine verwirrt zu werden.

### WCS ändern

Die 3D-Ansicht aktualisiert sich automatisch, wenn du das aktive WCS änderst:
- Ein anderes WCS aus der Symbolleisten-Dropdown auswählen
- Das Raster und die Achsen verschieben sich, um den neuen WCS-Ursprung
  widerzuspiegeln
- Beschriftungen aktualisieren sich, um Koordinaten relativ zum neuen WCS
  zu zeigen

:::tip WCS in der 3D-Ansicht
Die 3D-Ansicht zeigt deine Werkzeugwege relativ zum ausgewählten WCS. Wenn du
das WCS änderst, siehst du die Werkzeugwege scheinbar wandern, weil sich der
Referenzpunkt (das Raster) geändert hat, nicht weil sich die Werkzeugwege
selbst bewegt haben.
:::


## Anzeigeoptionen

Sichtbarkeits-Umschaltungen befinden sich als Overlay-Schaltflächen oben rechts
auf der 3D-Canvas:

- **Modell**: 3D-Maschinenmodell-Sichtbarkeit umschalten
- **Eilgänge**: Eilgang-Sichtbarkeit umschalten
- **No-Go-Zonen**: No-Go-Zonen-Sichtbarkeit umschalten

### Werkzeugweg-Visualisierung

Passe an, was du siehst:

- **Eilgänge anzeigen**: Positionierbewegungen anzeigen (gepunktete Linien)
- **Arbeitsbewegungen anzeigen**: Schneid-/Gravurbewegungen anzeigen
  (durchgezogene Linien)
- **Nach Operation färben**: Verschiedene Farben für jede Operation

:::tip Farben pro Laser
Bei Maschinen mit mehreren Laserköpfen kann jeder Laser eigene Schnitt- und
Rasterfarben haben, die in den [Lasereinstellungen](../machine/laser)
konfiguriert werden. Dies erleichtert die Identifizierung, welcher Laser
welche Operation ausführt.
:::

### Laserkopf-Modell

Die 3D-Ansicht rendert ein Modell deines Laserkopfes, das während der
Simulation dem Werkzeugweg folgt. Du kannst jedem Laserkopf ein 3D-Modell
auf der Seite [Lasereinstellungen](../machine/laser) in den
Maschineneinstellungen zuweisen. Skalierung, Rotation und Fokusabstand des
Modells können an dein physisches Setup angepasst werden.

Während der Simulation wird ein leuchtender Laserstrahl vom Kopf nach unten
gezeichnet, wenn der Laser aktiv ist.

## Simulation

Die 3D-Ansicht enthält einen eingebauten Simulator mit
Wiedergabesteuerungen, die am unteren Rand der Canvas überlagert sind.

### Wiedergabesteuerung

- **Wiedergabe/Pause** (<kbd>Leertaste</kbd>): Werkzeugweg-Ausführung animieren
- **Schritt vor/zurück**: Jeweils eine Operation vorwärts oder zurück
- **Geschwindigkeit**: Wiedergabegeschwindigkeiten durchschalten (1x, 2x, 4x, 8x, 16x)
- **Timeline-Schieberegler**: Ziehen, um durch den Auftrag zu navigieren

### Synchronisierte G-Code-Ansicht

Die Simulation bleibt mit dem G-Code-Viewer im unteren Panel synchronisiert.
Das Durchgehen der Simulation hebt die entsprechende Zeile im G-Code-Viewer
hervor, und das Klicken auf eine Zeile im G-Code-Viewer springt die Simulation
zu diesem Punkt.

### Ebenen-Sichtbarkeit

Sichtbarkeit einzelner Ebenen umschalten:

- Auf einen Ebenennamen klicken, um ihn anzuzeigen oder zu verbergen
- Auf bestimmte Ebenen für Inspektion fokussieren

## Verifizierungs-Checkliste

Vor dem Senden an die Maschine verifizieren:

- [ ] Werkzeugweg ist vollständig ohne fehlende Segmente
- [ ] Pfade bleiben innerhalb des Maschinenarbeitsbereichs
- [ ] Gravur-Operationen werden vor Schnitten ausgeführt
- [ ] Kein Werkzeugweg betritt eine No-Go-Zone
- [ ] Auftrag beginnt an der erwarteten Position
- [ ] Halterungen befinden sich an den korrekten Positionen

## Leistungstipps

Für große oder komplexe Aufträge:

1. Eilgänge ausblenden, um sich auf Arbeitsbewegungen zu fokussieren
2. Anzahl der sichtbaren Ebenen reduzieren
3. Andere Anwendungen schließen, um GPU-Ressourcen freizugeben

## Fehlerbehebung

### Vorschau ist leer oder schwarz

- Prüfen, dass Operationen aktiviert sind
- Verifizieren, dass Objekten Operationen zugewiesen sind

### Langsame oder ruckelige Vorschau

- Eilgänge ausblenden
- 3D-Modelle ausblenden
- Anzahl der sichtbaren Ebenen reduzieren

---

**Verwandte Seiten:**

- [Werkstückkoordinatensysteme](../general-info/coordinate-systems) - WCS
- [Hauptfenster](main-window) - Hauptoberflächenübersicht
- [Einstellungen](settings) - Anwendungseinstellungen
