# Unterstützte Dateiformate

Diese Seite bietet detaillierte Informationen über alle von Rayforge unterstützten Dateiformate, einschließlich Fähigkeiten, Einschränkungen und Empfehlungen.

## Format-Übersicht

### Schnellreferenz

| Format               | Typ     | Import        | Export          | Empfohlene Verwendung     |
| -------------------- | ------- | ------------- | --------------- | ------------------------- |
| **SVG**              | Vektor  | ✓ Direkt      | ✓ Objekt-Export | Primäres Design-Format    |
| **DXF**              | Vektor  | ✓ Direkt      | ✓ Objekt-Export | CAD-Datenaustausch        |
| **PDF**              | Gemischt| ✓ Nachverfolgung | –            | Dokumenten-Export (begrenzt) |
| **PNG**              | Raster  | ✓ Nachverfolgung | –            | Fotos, Bilder             |
| **JPEG**             | Raster  | ✓ Nachverfolgung | –            | Fotos                     |
| **BMP**              | Raster  | ✓ Nachverfolgung | –            | Einfache Grafiken         |
| **RFS**              | Skizze  | ✓ Direkt      | ✓ Objekt-Export | Parametrische Skizzen     |
| **G-Code**           | Steuerung| –           | ✓ Primär       | Maschinenausgabe          |
| **Rayforge-Projekt** | Projekt | ✓             | ✓               | Projekte speichern/laden  |

---

## Vektorformate

### SVG (Scalable Vector Graphics)

**Erweiterung:** `.svg`
**MIME-Typ:** `image/svg+xml`
**Import:** Direktes Vektor-Parsing oder Bitmap-Nachverfolgung
**Export:** Objekt-Export (nur Geometrie)

**Was ist SVG?**

SVG ist ein XML-basiertes Vektorbildformat. Es ist das **bevorzugte Format** für den Import von Designs in Rayforge.

**Unterstützte Funktionen:**

- ✓ Pfade (Linien, Kurven, Bögen)
- ✓ Grundformen (Rechtecke, Kreise, Ellipsen, Polygone)
- ✓ Gruppen und Transformationen
- ✓ Strich- und Füllfarben
- ✓ Mehrere Ebenen
- ✓ Koordinatentransformationen (verschieben, drehen, skalieren)

**Nicht unterstützte/Eingeschränkte Funktionen:**

- ✗ Text (muss zuerst in Pfade konvertiert werden)
- ✗ Farbverläufe (vereinfacht oder ignoriert)
- ✗ Filter und Effekte (ignoriert)
- ✗ Masken und Beschneidungspfade (funktionieren möglicherweise nicht korrekt)
- ✗ Eingebettete Rasterbilder (separat importiert wenn möglich)
- ✗ Komplexe Strichstile (Strichelungen können vereinfacht werden)
- ✗ Symbole und use-Elemente (Instanzen aktualisieren sich möglicherweise nicht)

**Export-Hinweise:**

Beim Exportieren eines Werkstücks nach SVG exportiert Rayforge die Geometrie als Vektorpfade mit:

- Nur-Strich-Rendering (keine Füllung)
- Millimeter-Einheiten
- Schwarze Strichfarbe

**Best Practices:**

1. **Plain SVG-Format verwenden** (nicht Inkscape SVG oder andere werkzeugspezifische Varianten)
2. **Text in Pfade konvertieren** vor dem Export
3. **Komplexe Pfade vereinfachen** um Knotenanzahl zu reduzieren
4. **Gruppen abflachen** wenn möglich
5. **Unbenutzte Elemente entfernen** (Hilfslinien, Raster, versteckte Ebenen)
6. **Dokumenteneinheiten einstellen** auf mm (Rayforge's native Einheit)

**Software-Empfehlungen:**

- **Inkscape** (kostenlos) - Ausgezeichnete SVG-Unterstützung, natives Format

---

### DXF (Drawing Exchange Format)

**Erweiterung:** `.dxf`
**MIME-Typ:** `application/dxf`, `image/vnd.dxf`
**Import:** Direktes Vektor-Parsing
**Export:** Objekt-Export (nur Geometrie)

**Was ist DXF?**

DXF ist ein AutoCAD-Zeichnungsformat, weit verbreitet für CAD-Datenaustausch.

**Unterstützte Versionen:**

- ✓ **R12/LT2** (empfohlen - beste Kompatibilität)
- ✓ R13, R14
- ✓ R2000 und später (funktioniert normalerweise, aber R12 ist sicherer)

**Unterstützte Entitäten:**

- ✓ Linien (LINE)
- ✓ Polylinien (LWPOLYLINE, POLYLINE)
- ✓ Bögen (ARC)
- ✓ Kreise (CIRCLE)
- ✓ Splines (SPLINE) - konvertiert zu Polylinien
- ✓ Ellipsen (ELLIPSE)
- ✓ Ebenen

**Nicht unterstützte/Eingeschränkte Funktionen:**

- ✗ 3D-Entitäten (2D-Projektion verwenden)
- ✗ Bemaßungen und Anmerkungen (ignoriert)
- ✗ Blöcke/Einfügungen (instanziieren sich möglicherweise nicht korrekt)
- ✗ Komplexe Linientypen (vereinfacht zu durchgehend)
- ✗ Text (ignoriert, zuerst in Umrisse konvertieren)
- ✗ Schraffuren (können vereinfacht oder ignoriert werden)

**Export-Hinweise:**

Beim Exportieren eines Werkstücks nach DXF exportiert Rayforge:

- Linien als LWPOLYLINE-Entitäten
- Bögen als ARC-Entitäten
- Bezier-Kurven als SPLINE-Entitäten
- Millimeter-Einheiten (INSUNITS = 4)

---

### RFS (Rayforge-Skizze)

**Erweiterung:** `.rfs`
**MIME-Typ:** `application/x-rayforge-sketch`
**Import:** Direkt (skizzenbasierte Werkstücke)
**Export:** Objekt-Export (skizzenbasierte Werkstücke)

**Was ist RFS?**

RFS ist Rayforge's natives parametrisches Skizzenformat. Es bewahrt alle geometrischen Elemente und parametrischen Bedingungen, sodass Sie vollständig bearbeitbare Skizzen speichern und teilen können.

**Funktionen:**

- ✓ Alle geometrischen Elemente (Linien, Bögen, Kreise, Rechtecke usw.)
- ✓ Alle parametrischen Bedingungen
- ✓ Dimensionswerte und Ausdrücke
- ✓ Füllbereiche

**Wann zu verwenden:**

- Wiederverwendbare parametrische Designs speichern
- Bearbeitbare Skizzen mit anderen Rayforge-Benutzern teilen
- Arbeit in progress archivieren

---

### PDF (Portable Document Format)

**Erweiterung:** `.pdf`
**MIME-Typ:** `application/pdf`
**Import:** Zu Bitmap gerendert, dann nachverfolgt
**Export:** Nicht unterstützt

**Was ist PDF-Import?**

Rayforge kann PDF-Dateien importieren, indem sie zuerst rastert und dann zu Vektoren nachverfolgt.

**Prozess:**

1. PDF zu Rasterbild gerendert (Standard 300 DPI)
2. Raster nachverfolgt um Vektorpfade zu erstellen
3. Pfade zum Dokument hinzugefügt

**Einschränkungen:**

- **Kein echter Vektor-Import** - Selbst Vektor-PDFs werden gerastert
- **Qualitätsverlust** durch Rasterung
- **Nur erste Seite** - Mehrseitige PDFs importieren nur Seite 1
- **Langsam für komplexe PDFs** - Rendern und Nachverfolgen dauert Zeit

**Wann zu verwenden:**

- Letzte Option wenn SVG/DXF nicht verfügbar
- Schneller Import einfacher Designs
- Dokumente mit gemischtem Inhalt

**Bessere Alternativen:**

- **SVG aus Quelle exportieren** statt PDF
- **Vektorformate verwenden** (SVG, DXF) wenn möglich
- **Für Text:** Export mit in Umrisse konvertiertem Text

---

## Rasterformate

Alle Rasterformate werden **durch Nachverfolgung importiert** - automatisch in Vektorpfade konvertiert.

### PNG (Portable Network Graphics)

**Erweiterung:** `.png`
**MIME-Typ:** `image/png`
**Import:** Zu Vektoren nachverfolgen
**Export:** Nicht unterstützt

**Eigenschaften:**

- **Verlustfreie Komprimierung** - Kein Qualitätsverlust
- **Transparenz-Unterstützung** - Alpha-Kanal bewahrt
- **Gut für:** Logos, Strichzeichnungen, Screenshots, alles mit Transparenz

**Nachverfolgungsqualität:** (Ausgezeichnet für hochkontrastige Bilder)

**Best Practices:**

- PNG für Logos und Grafiken mit scharfen Kanten verwenden
- Hohen Kontrast zwischen Vordergrund und Hintergrund sicherstellen
- Transparenter Hintergrund funktioniert besser als weißer

---

### JPEG (Joint Photographic Experts Group)

**Erweiterung:** `.jpg`, `.jpeg`
**MIME-Typ:** `image/jpeg`
**Import:** Zu Vektoren nachverfolgen
**Export:** Nicht unterstützt

**Eigenschaften:**

- **Verlustbehaftete Komprimierung** - Etwas Qualitätsverlust
- **Keine Transparenz** - Hat immer Hintergrund
- **Gut für:** Fotos, Bilder mit kontinuierlichen Tönen

**Nachverfolgungsqualität:** (Gut für Fotos, aber komplex)

**Best Practices:**

- Hochwertiges JPEG verwenden (niedrige Komprimierung)
- Kontrast vor dem Importieren erhöhen
- Vorbearbeitung im Bildeditor erwägen
- Besser zuerst zu PNG konvertieren wenn möglich

---

### BMP (Bitmap)

**Erweiterung:** `.bmp`
**MIME-Typ:** `image/bmp`
**Import:** Zu Vektoren nachverfolgen
**Export:** Nicht unterstützt

**Eigenschaften:**

- **Unkomprimiert** - Große Dateigrößen
- **Einfaches Format** - Weit kompatibel
- **Gut für:** Einfache Grafiken, Ausgabe alter Software

**Nachverfolgungsqualität:** (Gut, aber nicht besser als PNG)

**Best Practices:**

- Zu PNG für kleinere Dateigröße konvertieren (kein Qualitätsunterschied)
- Nur verwenden wenn Quellsoftware kein PNG/SVG exportieren kann

---

## Verwandte Seiten

- [Dateien importieren](importing) - Wie jedes Format importiert wird
- [Exportieren](exporting) - G-Code-Exportoptionen
