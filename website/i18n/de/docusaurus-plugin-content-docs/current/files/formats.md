# Unterstützte Dateiformate

Diese Seite bietet detaillierte Informationen über alle von Rayforge unterstützten
Dateiformate, einschließlich Fähigkeiten, Einschränkungen und Empfehlungen.

## Format-Übersicht

### Schnellreferenz

| Format               | Typ       | Import             | Export          | Empfohlene Verwendung      |
| -------------------- | --------- | ------------------ | --------------- | -------------------------- |
| **SVG**              | Vektor    | ✓ Direkt / Tracing | ✓ Objekt-Export | Primäres Design-Format     |
| **DXF**              | Vektor    | ✓ Direkt           | ✓ Objekt-Export | CAD-Datenaustausch         |
| **PDF**              | Gemischt  | ✓ Direkt / Tracing | –               | Dokumente mit Vektorinhalt |
| **PNG**              | Raster    | ✓ Tracing          | –               | Fotos, Bilder              |
| **JPEG**             | Raster    | ✓ Tracing          | –               | Fotos                      |
| **BMP**              | Raster    | ✓ Tracing          | –               | Einfache Grafiken          |
| **RFS**              | Skizze    | ✓ Direkt           | ✓ Objekt-Export | Parametrische Skizzen      |
| **G-Code**           | Steuerung | –                  | ✓ Primär        | Maschinenausgabe           |
| **Rayforge-Projekt** | Projekt   | ✓                  | ✓               | Projekte speichern/laden   |

---

## Vektorformate

### SVG (Scalable Vector Graphics)

**Erweiterung:** `.svg`
**MIME-Typ:** `image/svg+xml`
**Import:** Direktes Vektor-Parsing oder Bitmap-Tracing
**Export:** Objekt-Export (nur Geometrie)

**Was ist SVG?**

SVG ist ein XML-basiertes Vektorbildformat. Es ist das **bevorzugte Format** für
den Import von Designs in Rayforge.

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
- ✗ Masken und Beschneidungspfade (funktionieren möglicherweise nicht
  korrekt)
- ✗ Eingebettete Rasterbilder (separat importiert wenn möglich)
- ✗ Komplexe Strichstile (Strichelungen können vereinfacht werden)
- ✗ Symbole und use-Elemente (Instanzen aktualisieren sich möglicherweise
  nicht)

**Export-Hinweise:**

Beim Exportieren eines Werkstücks nach SVG exportiert Rayforge die Geometrie als
Vektorpfade mit:

- Nur-Strich-Rendering (keine Füllung)
- Millimeter-Einheiten
- Schwarze Strichfarbe

**Best Practices:**

1. **Plain SVG-Format verwenden** (nicht Inkscape SVG oder andere
   werkzeugspezifische Varianten)
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

DXF ist ein AutoCAD-Zeichnungsformat, weit verbreitet für
CAD-Datenaustausch.

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

RFS ist Rayforge's natives parametrisches Skizzenformat. Es bewahrt alle
geometrischen Elemente und parametrischen Bedingungen, sodass du vollständig
bearbeitbare Skizzen speichern und teilen kannst.

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
**Import:** Direkte Vektoren (mit Ebenen-Unterstützung) oder Rendern und Tracing
**Export:** Nicht unterstützt

**Was ist PDF-Import?**

PDF-Dateien können echte Vektorpfade enthalten, die Rayforge direkt importiert,
wenn verfügbar — und dir so dieselbe saubere Geometrie liefert wie ein SVG. Wenn
das PDF Ebenen hat, kann jede Ebene als separates Werkstück importiert werden.

Für PDFs ohne nutzbaren Vektorinhalt (gescannte Dokumente, Fotos) greift
Rayforge auf Rendern und Tracing zurück.

**Funktionen:**

- ✓ **Direkter Vektor-Import** für vektorbasierte PDFs
- ✓ **Ebenen-Erkennung und -Auswahl** — wähle aus, welche Ebenen importiert
  werden
- ✓ Fallback auf Rendern und Tracing für Rasterinhalte

**Einschränkungen:**

- Nur erste Seite — mehrseitige PDFs importieren Seite 1
- Text muss möglicherweise in der Quellanwendung in Umrisse konvertiert werden

**Wann zu verwenden:**

- PDFs von Designern, die Vektorgrafiken enthalten
- Jedes PDF mit gut organisierten Ebenen
- Wenn SVG oder DXF aus der Quelle nicht verfügbar ist

---

## Rasterformate

Alle Rasterformate werden **durch Tracing importiert** - automatisch in
Vektorpfade konvertiert.

### PNG (Portable Network Graphics)

**Erweiterung:** `.png`
**MIME-Typ:** `image/png`
**Import:** Tracing zu Vektoren
**Export:** Nicht unterstützt

**Eigenschaften:**

- **Verlustfreie Komprimierung** - Kein Qualitätsverlust
- **Transparenz-Unterstützung** - Alpha-Kanal bewahrt
- **Gut für:** Logos, Strichzeichnungen, Screenshots, alles mit Transparenz

**Tracing-Qualität:** (Ausgezeichnet für hochkontrastige Bilder)

**Best Practices:**

- PNG für Logos und Grafiken mit scharfen Kanten verwenden
- Hohen Kontrast zwischen Vordergrund und Hintergrund sicherstellen
- Transparenter Hintergrund funktioniert besser als weißer

---

### JPEG (Joint Photographic Experts Group)

**Erweiterung:** `.jpg`, `.jpeg`
**MIME-Typ:** `image/jpeg`
**Import:** Tracing zu Vektoren
**Export:** Nicht unterstützt

**Eigenschaften:**

- **Verlustbehaftete Komprimierung** - Etwas Qualitätsverlust
- **Keine Transparenz** - Hat immer Hintergrund
- **Gut für:** Fotos, Bilder mit kontinuierlichen Tönen

**Tracing-Qualität:** (Gut für Fotos, aber komplex)

**Best Practices:**

- Hochwertiges JPEG verwenden (niedrige Komprimierung)
- Kontrast vor dem Importieren erhöhen
- Vorbearbeitung im Bildeditor erwägen
- Besser zuerst zu PNG konvertieren wenn möglich

---

### BMP (Bitmap)

**Erweiterung:** `.bmp`
**MIME-Typ:** `image/bmp`
**Import:** Tracing zu Vektoren
**Export:** Nicht unterstützt

**Eigenschaften:**

- **Unkomprimiert** - Große Dateigrößen
- **Einfaches Format** - Weit kompatibel
- **Gut für:** Einfache Grafiken, Ausgabe alter Software

**Tracing-Qualität:** (Gut, aber nicht besser als PNG)

**Best Practices:**

- Zu PNG für kleinere Dateigröße konvertieren (kein Qualitätsunterschied)
- Nur verwenden wenn Quellsoftware kein PNG/SVG exportieren kann

---

## Verwandte Seiten

- [Dateien importieren](importing) - Wie jedes Format importiert wird
- [Exportieren](exporting) - G-Code-Exportoptionen
