# 3D-Vorschau

Das 3D-Vorschau-Fenster ermöglicht dir, deine G-Code-Werkzeugwege zu visualisieren, bevor du sie an deine Maschine sendest. Diese leistungsstarke Funktion hilft dir, Fehler zu erkennen und deine Auftragskonfiguration zu verifizieren.

![3D-Vorschau](/screenshots/main-3d.png)

## 3D-Vorschau öffnen

Zugriff auf die 3D-Vorschau:

- **Menü**: Ansicht → 3D-Vorschau
- **Tastatur**: <kbd>Strg+3</kbd>
- **Nach G-Code-Generierung**: Öffnet automatisch (konfigurierbar)

## Navigation

### Maussteuerung

- **Drehen**: Linksklick und ziehen
- **Verschieben**: Rechtsklick und ziehen, oder Mittelklick und ziehen
- **Zoom**: Mausrad, oder <kbd>Strg</kbd> + Linksklick und ziehen

### Tastatursteuerung

- <kbd>r</kbd>: Kamera auf Standardansicht zurücksetzen
- <kbd>Pos1</kbd>: Zoom und Position zurücksetzen
- <kbd>f</kbd>: Ansicht an Werkzeugweg anpassen
- Pfeiltasten: Kamera drehen

### Ansicht-Presets

Schnelle Kamerawinkel:

- **Draufsicht** (<kbd>1</kbd>): Vogelperspektive
- **Vorderansicht** (<kbd>2</kbd>): Frontale Ansicht
- **Rechtsansicht** (<kbd>3</kbd>): Rechte Seitenansicht
- **Isometrisch** (<kbd>4</kbd>): 3D-isometrische Ansicht

## Werkstückkoordinatensystem-Anzeige

Die 3D-Vorschau visualisiert das aktive Werkstückkoordinatensystem (WCS) anders als die 2D-Canvas:

### Raster und Achsen

- **Isolierte Anzeige**: Das Raster und die Achsen erscheinen so, als wäre der WCS-Ursprung der Weltursprung
- **Offset angewendet**: Das gesamte Raster ist verschoben, um mit dem ausgewählten WCS-Offset ausgerichtet zu sein
- **Beschriftungen relativ zum WCS**: Koordinatenbeschriftungen zeigen Positionen relativ zum WCS-Ursprung, nicht zum Maschinenursprung

Diese "in Isolation"-Anzeige erleichtert es zu verstehen, wo dein Auftrag relativ zum ausgewählten Werkstückkoordinatensystem laufen wird, ohne durch die absolute Position der Maschine verwirrt zu werden.

### WCS ändern

Die 3D-Vorschau aktualisiert sich automatisch, wenn du das aktive WCS änderst:
- Ein anderes WCS aus der Symbolleisten-Dropdown auswählen
- Das Raster und die Achsen verschieben sich, um den neuen WCS-Ursprung widerzuspiegeln
- Beschriftungen aktualisieren sich, um Koordinaten relativ zum neuen WCS zu zeigen

:::tip WCS in der 3D-Vorschau
Die 3D-Vorschau zeigt deine Werkzeugwege relativ zum ausgewählten WCS. Wenn du das WCS änderst, siehst du die Werkzeugwege scheinbar wandern, weil sich der Referenzpunkt (das Raster) geändert hat, nicht weil sich die Werkzeugwege selbst bewegt haben.
:::


## Anzeigeoptionen

### Werkzeugweg-Visualisierung

Passe an, was du siehst:

- **Eilgänge anzeigen**: Positionierbewegungen anzeigen (gepunktete Linien)
- **Arbeitsbewegungen anzeigen**: Schneid-/Gravurbewegungen anzeigen (durchgezogene Linien)
- **Nach Operation färben**: Verschiedene Farben für jede Operation
- **Nach Leistung färben**: Farbverlauf basierend auf Laserleistung
- **Nach Geschwindigkeit färben**: Farbverlauf basierend auf Vorschubrate

### Maschinen-Visualisierung

- **Ursprung anzeigen**: (0,0)-Referenzpunkt anzeigen
- **Arbeitsbereich anzeigen**: Maschinengrenzen anzeigen
- **Laserkopf anzeigen**: Aktuelle Positionsanzeige anzeigen

### Qualitätseinstellungen

- **Linienbreite**: Dicke der Werkzeugweg-Linien
- **Kantenglättung**: Glattes Linien-Rendering (kann Leistung beeinträchtigen)
- **Hintergrund**: Hell, dunkel oder benutzerdefinierte Farbe

## Wiedergabesteuerung

Auftragsausführung simulieren:

- **Wiedergabe/Pause** (<kbd>Leertaste</kbd>): Werkzeugweg-Ausführung animieren
- **Geschwindigkeit**: Wiedergabegeschwindigkeit anpassen (0,5x - 10x)
- **Schritt vor/zurück**: Um einzelne G-Code-Befehle vorwärts
- **Zu Position springen**: Auf Timeline klicken um zu spezifischem Punkt zu springen

### Timeline

Die Timeline zeigt:

- Aktuelle Position im Auftrag
- Operationsgrenzen (farbige Segmente)
- Geschätzte Zeit an jedem Punkt

## Analysewerkzeuge

### Distanzmessung

Distanzen in 3D messen:

1. Messwerkzeug aktivieren
2. Zwei Punkte auf dem Werkzeugweg anklicken
3. Distanz in aktuellen Einheiten anzeigen

### Statistik-Panel

Auftragsstatistiken anzeigen:

- **Gesamtdistanz**: Summe aller Bewegungen
- **Arbeitsdistanz**: Nur Schneid-/Gravurdistanz
- **Eilgangdistanz**: Nur Positionierbewegungen
- **Geschätzte Zeit**: Auftragsdauer-Schätzung
- **Begrenzungsrahmen**: Gesamtmaße

### Ebenen-Sichtbarkeit

Sichtbarkeit von Operationen umschalten:

- Auf Operationsnamen klicken zum Anzeigen/Verbergen
- Auf bestimmte Operationen für Inspektion fokussieren
- Probleme isolieren ohne G-Code neu zu generieren

## Verifizierungs-Checkliste

Vor dem Senden an die Maschine verifizieren:

- [ ] **Werkzeugweg ist vollständig**: Keine fehlenden Segmente
- [ ] **Innerhalb des Arbeitsbereichs**: Bleibt innerhalb der Maschinengrenzen
- [ ] **Korrekte Operationsreihenfolge**: Gravieren vor Schneiden
- [ ] **Keine Kollisionen**: Kopf trifft keine Spannvorrichtungen
- [ ] **Richtiger Ursprung**: Beginnt an erwarteter Position
- [ ] **Halterungspositionen**: Halterungen an korrekten Orten (falls verwendet)

## Leistungstipps

Für große oder komplexe Aufträge:

1. **Liniendetail reduzieren**: Niedrigere Anzeigequalität für schnelleres Rendering
2. **Eilgänge ausblenden**: Nur auf Arbeitsbewegungen fokussieren
3. **Kantenglättung deaktivieren**: Verbessert Bildrate
4. **Andere Anwendungen schließen**: GPU-Ressourcen freigeben

## Fehlerbehebung

### Vorschau ist leer oder schwarz

- G-Code neu generieren (<kbd>Strg+g</kbd>)
- Prüfen, dass Operationen aktiviert sind
- Verifizieren, dass Objekten Operationen zugewiesen sind

### Langsame oder ruckelige Vorschau

- Linienbreite reduzieren
- Kantenglättung deaktivieren
- Eilgänge ausblenden
- Grafiktreiber aktualisieren

### Farben werden nicht korrekt angezeigt

- Färbung-nach-Einstellung prüfen (Operation/Leistung/Geschwindigkeit)
- Sicherstellen, dass Operationen verschiedene Farben zugewiesen haben
- Ansichtseinstellungen auf Standard zurücksetzen

---

**Verwandte Seiten:**

- [Werkstückkoordinatensysteme](../general-info/work-coordinate-systems) - WCS
- [Hauptfenster](main-window) - Hauptoberflächenübersicht
- [Einstellungen](settings) - Anwendungseinstellungen
