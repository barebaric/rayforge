# Exportieren aus Rayforge

Rayforge unterstützt mehrere Exportoptionen für verschiedene Zwecke:

- **G-Code** - Maschinensteuerungsausgabe zum Ausführen von Aufträgen
- **Objekt-Export** - Einzelne Werkstücke in Vektorformate exportieren
- **Dokument-Export** - Alle Werkstücke als einzelne Datei exportieren

---

## Objekte exportieren

Du kannst jedes Werkstück in Vektorformate exportieren zur Verwendung in Design-Software, CAD-Anwendungen oder zur Archivierung.

### Wie man exportiert

1. **Werkstück auswählen** auf der Canvas
2. **Objekt → Objekt exportieren... wählen** (oder Rechtsklick → Objekt exportieren...)
3. **Format und** Speicherort auswählen

### Verfügbare Formate

| Format  | Erweiterung | Beschreibung                                                                                                  |
| ------- | ----------- | ------------------------------------------------------------------------------------------------------------- |
| **RFS** | `.rfs`      | Rayforge's natives parametrisches Skizzenformat. Bewahrt alle Bedingungen und kann zum Bearbeiten wieder importiert werden. |
| **SVG** | `.svg`      | Scalable Vector Graphics. Weit kompatibel mit Design-Software wie Inkscape, Illustrator und Webbrowsern.      |
| **DXF** | `.dxf`      | Drawing Exchange Format. Kompatibel mit den meisten CAD-Anwendungen wie AutoCAD, FreeCAD und LibreCAD.        |

### Export-Hinweise

- **SVG und DXF** exportieren die aufgelöste Geometrie (nicht parametrische Bedingungen)
- Exporte verwenden **Millimeter-Einheiten**
- Geometrie ist auf tatsächliche Abmessungen skaliert (Weltkoordinaten)
- Mehrere Teilpfade (getrennte Formen) werden als separate Elemente bewahrt

### Anwendungsfälle

**Designs teilen:**

- Nach SVG exportieren zum Teilen mit Inkscape-Benutzern
- Nach DXF exportieren für CAD-Software-Benutzer

**Weiterbearbeitung:**

- Nach SVG/DXF exportieren, in externer Software bearbeiten, wieder importieren

**Archivierung:**

- RFS für skizzenbasierte Designs verwenden um Bearbeitbarkeit zu bewahren
- SVG/DXF für langfristige Speicherung oder Nicht-Rayforge-Benutzer

---

## Dokumente exportieren

Du kannst alle Werkstücke in einem Dokument in eine einzelne Vektordatei exportieren. Dies ist nützlich zum Teilen kompletter Projekte oder Erstellen von Backups in Standardformaten.

### Wie man exportiert

1. **Datei → Dokument exportieren... wählen**
2. **Format auswählen** (SVG oder DXF)
3. **Speicherort wählen**

### Verfügbare Formate

| Format  | Erweiterung | Beschreibung                                                                                                  |
| ------- | ----------- | ------------------------------------------------------------------------------------------------------------- |
| **SVG** | `.svg`      | Scalable Vector Graphics. Weit kompatibel mit Design-Software wie Inkscape, Illustrator und Webbrowsern.      |
| **DXF** | `.dxf`      | Drawing Exchange Format. Kompatibel mit den meisten CAD-Anwendungen wie AutoCAD, FreeCAD und LibreCAD.        |

### Export-Hinweise

- Alle Werkstücke aus allen Ebenen werden in einer einzigen Datei kombiniert
- Werkstückpositionen werden bewahrt
- Leere Werkstücke werden übersprungen
- Der Begrenzungsrahmen umfasst die gesamte Geometrie

### Anwendungsfälle

- **Projekt-Teilung**: Gesamtes Projekt für Zusammenarbeit exportieren
- **Backup**: Visuelles Archiv deiner Arbeit erstellen
- **Weiterbearbeitung**: Das gesamte Design in Inkscape oder CAD-Software übernehmen

---

## G-Code exportieren

Generierter G-Code enthält alles genau so, wie es an die Maschine gesendet würde.
Das genaue Format, die Befehle, numerische Präzision usw. hängen von den Einstellungen der aktuell ausgewählten Maschine und ihres G-Code-Dialekts ab.

---

### Export-Methoden

### Methode 1: Dateimenü

**Datei G-Code exportieren** (Strg+E)

- Öffnet Datei-Speichern-Dialog
- Speicherort und Dateinamen wählen
- G-Code wird generiert und gespeichert

### Methode 2: Kommandozeile

```bash
# Aus Kommandozeile exportieren (falls unterstützt)
rayforge --export ausgabe.gcode eingabe.svg
```

---

### G-Code-Ausgabe

Generierter G-Code enthält alles genau so, wie es an die Maschine gesendet würde.
Das genaue Format, die Befehle, numerische Präzision usw. hängen von den Einstellungen der aktuell ausgewählten Maschine und ihres G-Code-Dialekts ab.

---

## Verwandte Seiten

- [Dateien importieren](importing) - Designs in Rayforge importieren
- [Unterstützte Formate](formats) - Dateiformat-Details
- [G-Code-Dialekte](../reference/gcode-dialects) - Dialekt-Unterschiede
- [Hooks & Makros](../machine/hooks-macros) - Ausgabe anpassen
- [Simulationsmodus](../features/simulation-mode) - Vor dem Export voranschauen
