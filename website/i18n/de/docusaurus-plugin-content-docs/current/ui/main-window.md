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

### 4. Seitenpanel

Das Seitenpanel ist ein schwebendes Overlay auf der rechten Seite der Canvas. Es
zeigt den Workflow der aktiven Ebene als vertikale Liste von Schritten an. Jeder
Schritt zeigt seinen Namen, eine Zusammenfassung (z.B. Leistung und Geschwindigkeit)
und Schaltflächen für Sichtbarkeit, Einstellungen und Löschung. Verwende die **+**-
Schaltfläche, um neue Schritte hinzuzufügen. Schritte können per Drag-and-Drop
neu angeordnet werden.

Ein Klick auf die Einstellungen-Schaltfläche eines Schritts öffnet einen Dialog, in dem
du den Operationstyp, Laserleistung, Schnittgeschwindigkeit, Air Assist, Strahlbreite
und Nachbearbeitungsoptionen konfigurierst. Schieberegler-Werte sind bearbeitbar —
klicke auf einen Wert neben einem Schieberegler und gib die genaue Zahl ein, die du
möchtest.

Das Panel kann beiseitegeschoben werden, wenn es nicht benötigt wird.

### 5. Unteres Panel

Das untere Panel bietet andockbare Tabs, die durch Ziehen neu angeordnet und in
mehrere Spalten aufgeteilt werden können. Die verfügbaren Tabs sind:

- **Ebenen**: Zeigt alle Ebenen als nebeneinanderliegende Spalten an. Jede Spalte hat
  eine Kopfzeile mit dem Ebenennamen und Steuerungen, eine kompakte horizontale
  Pipeline von Schritt-Symbolen, die den Workflow darstellen, und eine Liste von
  Werkstücken. Ebenen und Werkstücke können per Drag-and-Drop neu angeordnet werden.
- **Assets**: Listet Rohmaterial und Skizzen in deinem Dokument auf.
- **Konsole**: Interaktives Terminal zum Senden von G-code und Überwachen der
  Maschinenkommunikation.
- **G-code-Viewer**: Zeigt den generierten G-code mit Syntaxhervorhebung an.
- **Steuerungen**: Jog-Steuerungen zur manuellen Positionierung und WCS-Verwaltung.

Die geschätzte Auftragszeit wird in der Kopfzeile der Ebenenliste angezeigt.

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
