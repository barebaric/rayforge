# Hauptfenster

Das Rayforge-Hauptfenster ist Ihr primärer Arbeitsbereich zum Erstellen und Verwalten von Laseraufträgen.

## Fensterlayout

![Hauptfenster](/screenshots/main-standard.png)

### 1. Menüleiste

Zugriff auf alle Rayforge-Funktionen durch organisierte Menüs:

- **Datei**: Öffnen, speichern, importieren, exportieren und zuletzt verwendete Dateien
- **Bearbeiten**: Rückgängig, wiederholen, kopieren, einfügen, Einstellungen
- **Ansicht**: Zoom, Raster, Lineale, Panels und Ansichtsmodi
- **Operationen**: Hinzufügen, bearbeiten und Operationen verwalten
- **Maschine**: Verbinden, jogen, referenzieren, Aufträge starten/stoppen
- **Hilfe**: Dokumentation, Über und Support

### 2. Symbolleiste

Schneller Zugriff auf häufig verwendete Werkzeuge:

- **Auswahlwerkzeug**: Objekte auswählen und bewegen
- **Verschiebe-Werkzeug**: Auf der Canvas navigieren
- **Zoom-Werkzeug**: Auf bestimmte Bereiche vergrößern/verkleinern
- **Messwerkzeug**: Distanzen und Winkel messen
- **Ausrichtungswerkzeuge**: Objekte ausrichten und verteilen
- **WCS-Dropdown**: Das aktive Werkstückkoordinatensystem auswählen (G53-G59)

Das WCS-Dropdown ermöglicht Ihnen, schnell zwischen Koordinatensystemen zu wechseln.
Siehe [Werkstückkoordinatensysteme](../general-info/work-coordinate-systems) für
weitere Informationen.

### 3. Canvas

Der Hauptarbeitsbereich, in dem Sie:

- Designs importieren und anordnen
- Werkzeugwege voranschauen
- Objekte relativ zum Maschinennullpunkt positionieren
- Rahmen-Grenzen testen

**Canvas-Steuerung:**

- **Verschieben**: Mittelklick ziehen oder <kbd>Leertaste</kbd> + ziehen
- **Zoom**: Mausrad oder <kbd>Strg+"+"</kbd> / <kbd>Strg+"-"</kbd>
- **Ansicht zurücksetzen**: <kbd>Strg+0</kbd> oder Ansicht → Zoom zurücksetzen

### 4. Ebenen-Panel

Operationen und Ebenenzuweisungen verwalten:

- Alle Operationen in Ihrem Projekt anzeigen
- Operationen zu Design-Elementen zuweisen
- Reihenfolge der Operationsausführung ändern
- Einzelne Operationen aktivieren/deaktivieren
- Operationsparameter konfigurieren

### 5. Eigenschaften-Panel

Einstellungen für ausgewählte Objekte oder Operationen konfigurieren:

- Operationstyp (Kontur, Raster usw.)
- Leistungs- und Geschwindigkeitseinstellungen
- Anzahl der Durchgänge
- Erweiterte Optionen (Overscan, Schnittfuge, Halterungen)

### 6. Bedienfeld

Das Bedienfeld am unteren Rand des Fensters bietet:

- **Jog-Steuerung**: Manuelle Maschinenbewegung und Positionierung
- **Maschinenstatus**: Echtzeit-Position und Verbindungszustand
- **Protokollansicht**: G-Code-Kommunikation und Operationsverlauf
- **WCS-Verwaltung**: Werkstückkoordinatensystem-Auswahl und Nullsetzung

Siehe [Bedienfeld](control-panel) für detaillierte Informationen.

## Fensterverwaltung

### Panels

Panels nach Bedarf anzeigen/verbergen:

- **Ebenen-Panel**: Ansicht → Ebenen-Panel (<kbd>Strg+l</kbd>)
- **Eigenschaften-Panel**: Ansicht → Eigenschaften-Panel (<kbd>Strg+i</kbd>)

### Vollbildmodus

Auf Ihre Arbeit fokussieren mit Vollbild:

- Aktivieren: <kbd>F11</kbd> oder Ansicht → Vollbild
- Beenden: <kbd>F11</kbd> oder <kbd>Esc</kbd>

## Anpassung

Passen Sie die Oberfläche an in **Bearbeiten → Einstellungen**:

- **Design**: Hell, dunkel oder System
- **Einheiten**: Millimeter oder Zoll
- **Raster**: Anzeigen/verbergen und Rasterabstand konfigurieren
- **Lineale**: Lineale auf Canvas anzeigen/verbergen
- **Symbolleiste**: Sichtbare Schaltflächen anpassen

---

**Verwandte Seiten:**

- [Werkstückkoordinatensysteme](../general-info/work-coordinate-systems) - WCS
- [Canvas-Werkzeuge](canvas-tools) - Werkzeuge zum Bearbeiten von Designs
- [Bedienfeld](control-panel) - Manuelle Maschinensteuerung, Status und Protokolle
- [3D-Vorschau](3d-preview) - Werkzeugwege in 3D visualisieren
