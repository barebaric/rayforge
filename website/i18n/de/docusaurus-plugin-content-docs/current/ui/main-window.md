# Hauptfenster

Das Rayforge-Hauptfenster ist dein primärer Arbeitsbereich zum Erstellen und
Verwalten von Laseraufträgen.

## Fensterlayout

![Hauptfenster](/screenshots/main-standard.png)

### 1. Menüleiste

Zugriff auf alle Rayforge-Funktionen durch organisierte Menüs:

- **Datei**: Öffnen, speichern, importieren, exportieren und zuletzt verwendete Dateien
- **Bearbeiten**: Rückgängig, wiederholen, kopieren, einfügen, Einstellungen
- **Ansicht**: Zoom, Raster, Lineale, Panels und Ansichtsmodi
- **Operationen**: Hinzufügen, bearbeiten und Operationen verwalten
- **Maschine**: Verbinden, jogen, referenzieren, Aufträge starten/stoppen
- **Hilfe**: Über, Spenden, Debug-Protokoll speichern

### 2. Symbolleiste

Schneller Zugriff auf häufig verwendete Steuerungen:

- **Maschinen-Dropdown**: Wähle deine Maschine, sieh den Verbindungsstatus und
  die geschätzte verbleibende Zeit während Aufträgen
- **WCS-Dropdown**: Das aktive Werkstückkoordinatensystem auswählen (G53-G59)
- **Simulation umschalten**: Simulationsmodus aktivieren/deaktivieren
- **Laser fokussieren**: Laserfokus-Modus umschalten
- **Auftragssteuerung**: Referenzieren, Einrahmen, Senden, Halten und Abbrechen

Das Maschinen-Dropdown zeigt den Verbindungsstatus und aktuellen Zustand
(z.B. Bereit, Ausführung) deiner Maschine direkt in der Symbolleiste an.
Während der Auftragsausführung wird zusätzlich eine geschätzte verbleibende
Zeit angezeigt.

Das WCS-Dropdown ermöglicht dir, schnell zwischen Koordinatensystemen zu
wechseln. Siehe [Werkstückkoordinatensysteme](../general-info/coordinate-systems)
für weitere Informationen.

Sichtbarkeits-Umschaltungen für Werkstücke, Halterungen, Kamera-Feed,
Eilgänge und andere Elemente sind als Overlay-Schaltflächen direkt auf der
Canvas verfügbar, sodass sie immer griffbereit sind, während du arbeitest.

### 3. Canvas

Der Hauptarbeitsbereich, in dem du:

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

- Alle Operationen in deinem Projekt anzeigen
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

### 6. Unteres Panel

Das untere Panel bietet Tabs für die Konsole, den G-Code-Viewer und deine
Dokument-Assets (Rohmaterial und Skizzen). Jog-Steuerung und WCS-Verwaltung
sind auf der rechten Seite immer sichtbar. Die geschätzte Auftragszeit wird
in der Ebenenlistenüberschrift über dem Ebenen-Panel angezeigt.

Siehe [Unteres Panel](bottom-panel) für detaillierte Informationen.

## Fensterverwaltung

### Panels

Panels nach Bedarf anzeigen/verbergen:

- **Unteres Panel**: Ansicht → Unteres Panel (<kbd>Strg+l</kbd>)

### Vollbildmodus

Auf deine Arbeit fokussieren mit Vollbild:

- Aktivieren: <kbd>F11</kbd> oder Ansicht → Vollbild
- Beenden: <kbd>F11</kbd> oder <kbd>Esc</kbd>

## Anpassung

Passe die Oberfläche an in **Bearbeiten → Einstellungen**:

- **Design**: Hell, dunkel oder System
- **Einheiten**: Millimeter oder Zoll
- **Raster**: Anzeigen/verbergen und Rasterabstand konfigurieren
- **Lineale**: Lineale auf Canvas anzeigen/verbergen

---

**Verwandte Seiten:**

- [Werkstückkoordinatensysteme](../general-info/coordinate-systems) - WCS
- [Canvas-Werkzeuge](canvas-tools) - Werkzeuge zum Bearbeiten von Designs
- [Unteres Panel](bottom-panel) - Manuelle Maschinensteuerung, Status und Protokolle
- [3D-Vorschau](3d-preview) - Werkzeugwege in 3D visualisieren
