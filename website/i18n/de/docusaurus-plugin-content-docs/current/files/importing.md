# Dateien importieren

Rayforge unterstützt das Importieren verschiedener Dateiformate, sowohl Vektor- als auch
Rasterformate. Diese Seite erklärt, wie du Dateien importierst und für beste Ergebnisse
optimierst.

## Unterstützte Dateiformate

### Vektorformate

| Format    | Erweiterung | Importmethode                 | Am besten für                       |
| --------- | ----------- | ----------------------------- | ----------------------------------- |
| **SVG**   | `.svg`      | Direkte Vektoren oder Tracing | Vektorgrafiken, Logos, Designs      |
| **DXF**   | `.dxf`      | Direkte Vektoren              | CAD-Zeichnungen, technische Designs |
| **PDF**   | `.pdf`      | Direkte Vektoren oder Tracing | Dokumente mit Vektorinhalt          |
| **Ruida** | `.rd`       | Direkte Vektoren              | Ruida-Controller-Auftragsdateien    |

### Rasterformate

| Format   | Erweiterung     | Importmethode       | Am besten für                            |
| -------- | --------------- | ------------------- | ---------------------------------------- |
| **PNG**  | `.png`          | Tracing zu Vektoren | Fotos, Bilder mit Transparenz            |
| **JPEG** | `.jpg`, `.jpeg` | Tracing zu Vektoren | Fotos, Bilder mit kontinuierlichen Tönen |
| **BMP**  | `.bmp`          | Tracing zu Vektoren | Einfache Grafiken, Screenshots           |

:::note Raster-Import
:::

Alle Rasterbilder werden **traced**, um Vektorpfade zu erstellen, die für
Laseroperationen verwendet werden können. Die Qualität hängt von der
Tracing-Konfiguration ab.

---

## Dateien importieren

### Der Import-Dialog

Rayforge verfügt über einen einheitlichen Import-Dialog, der Live-Vorschau und
Konfigurationsoptionen für alle unterstützten Dateitypen bietet. Der Dialog
ermöglicht dir:

- **Import voranschauen**, bevor er zum Dokument hinzugefügt wird
- **Tracing-Einstellungen konfigurieren** für Rasterbilder
- **Importmethode wählen** für SVG-Dateien (direkte Vektoren oder Tracing)
- **Parameter anpassen** wie Schwellenwert, Invertieren und Auto-Schwellenwert

![Import-Dialog](/screenshots/import-dialog.png)

### Methode 1: Dateimenü

1. **Datei Importieren** (oder Strg+I)
2. **Datei auswählen** aus dem Dateiauswahldialog
3. **Importeinstellungen konfigurieren** im Import-Dialog
4. **Vorschau** des Ergebnisses vor dem Import
5. **Auf Import klicken** um zur Canvas und zum Dokumentenbaum hinzuzufügen

### Methode 2: Ziehen und Ablegen

1. **Datei ziehen** aus deinem Dateimanager
2. **Auf Rayforge-Canvas ablegen**
3. **Importeinstellungen konfigurieren** im Import-Dialog
4. **Vorschau** des Ergebnisses vor dem Import
5. **Auf Import klicken** um zur Canvas und zum Dokumentenbaum hinzuzufügen

### Methode 3: Kommandozeile

```bash
# Rayforge mit einer Datei öffnen
rayforge meineDatei.svg

# Mehrere Dateien
rayforge datei1.svg datei2.dxf
```

### Automatische Größenanpassung beim Import

Beim Importieren von Dateien, die größer als der Arbeitsbereich deiner Maschine sind,
wird Rayforge automatisch:

1. **Herunterskalieren** des importierten Inhalts, um in die Maschinengrenzen zu passen
2. **Seitenverhältnis beibehalten** während der Skalierung
3. **Zentrieren** des skalierten Inhalts im Arbeitsbereich
4. **Benachrichtigung anzeigen** mit der Option, die Größenänderung rückgängig zu machen

Die Größenänderungs-Benachrichtigung erscheint als Toast-Meldung:

- ⚠️ "Importiertes Element war größer als der Arbeitsbereich und wurde
  herunterskaliert um zu passen."
- Enthält eine **"Zurücksetzen"-Schaltfläche** um die automatische Größenänderung
  rückgängig zu machen
- Der Toast bleibt sichtbar bis er geschlossen wird oder die Zurücksetzen-Aktion
  ausgeführt wird

Dies stellt sicher, dass deine Designs immer in die Fähigkeiten deiner Maschine passen,
während du die Flexibilität hast, die Originalgröße bei Bedarf wiederherzustellen.

---

## SVG-Import

SVG (Scalable Vector Graphics) ist das **empfohlene Format** für Vektor-Designs.

### Importoptionen im Dialog

Beim Importieren von SVG bietet der Import-Dialog einen Umschalter, um zwischen zwei
Methoden zu wählen:

#### 1. Originalvektoren verwenden (Empfohlen)

Diese Option ist im Import-Dialog standardmäßig aktiviert.

**Wie es funktioniert:**

- Parst SVG und konvertiert Pfade direkt in Rayforge-Geometrie
- Hohe Wiedergabetreue bei Kurven und Formen
- Behält exakte Vektordaten bei

**Vorteile:**

- Beste Qualität und Präzision
- Bearbeitbare Pfade
- Kleine Dateigröße

**Nachteile:**

- Einige fortgeschrittene SVG-Funktionen werden nicht unterstützt
- Komplexe SVGs können Probleme haben

**Verwenden für:**

- Saubere Vektor-Designs aus Inkscape, Illustrator
- Einfache bis mittlere Komplexität
- Designs ohne fortgeschrittene SVG-Funktionen

#### 2. Bitmap tracen

Deaktiviere "Originalvektoren verwenden", um diese Methode zu nutzen.

**Wie es funktioniert:**

- Rendert SVG zuerst in ein Rasterbild
- Traced das gerenderte Bild, um Vektoren zu erstellen
- Kompatibler, aber weniger präzise

**Vorteile:**

- Verarbeitet komplexe SVG-Funktionen
- Robuste Fallback-Methode
- Unterstützt Effekte und Filter

**Nachteile:**

- Qualitätsverlust durch Rasterung
- Größere Dateigrößen
- Nicht so präzise

**Verwenden für:**

- SVGs, bei denen direkter Import fehlschlägt
- SVGs mit Effekten, Filtern, Farbverläufen
- Wenn direkter Import Fehler produziert

### Live-Vorschau

Der Import-Dialog zeigt eine Live-Vorschau, wie dein SVG importiert wird:

- Vektorpfade werden als blaues Overlay angezeigt
- Für Tracing-Modus wird das Originalbild mit den getraceten Pfaden angezeigt
- Vorschau aktualisiert sich in Echtzeit beim Ändern der Einstellungen

### SVG-Best Practices

**Bereite dein SVG für beste Ergebnisse vor:**

1. **Text in Pfade konvertieren:**
   - Inkscape: `Pfad → Objekt in Pfad`
   - Illustrator: `Typ → Umrisse erstellen`

2. **Komplexe Pfade vereinfachen:**
   - Inkscape: `Pfad → Vereinfachen` (Strg+L)
   - Unnötige Knoten entfernen

3. **Verschachtelte Gruppen auflösen:**
   - Hierarchie wo möglich abflachen
   - `Objekt → Gruppierung aufheben` (Strg+Umschalt+G)

4. **Versteckte Elemente entfernen:**
   - Hilfslinien, Raster, Konstruktionslinien löschen
   - Unsichtbare/transparente Objekte entfernen

5. **Als Plain SVG speichern:**
   - Inkscape: "Plain SVG" oder "Optimiertes SVG"
   - Nicht "Inkscape SVG" (hat zusätzliche Metadaten)

6. **Dokumenteneinheiten prüfen:**
   - Auf mm oder Zoll wie angemessen einstellen
   - Rayforge verwendet intern mm

**Häufige SVG-Funktionen, die möglicherweise nicht importiert werden:**

- Farbverläufe (in einfarbige Füllungen oder Raster konvertieren)
- Filter und Effekte (in Pfade umwandeln)
- Masken und Beschneidungspfade (erweitern/abflachen)
- Eingebettete Rasterbilder (separat exportieren)
- Text (zuerst in Pfade konvertieren)

---

## DXF-Import

DXF (Drawing Exchange Format) ist üblich für CAD-Software.

### DXF-Versionen

Rayforge unterstützt Standard-DXF-Formate:

- **R12/LT2** (empfohlen) - Beste Kompatibilität
- **R13, R14** - Gute Unterstützung
- **R2000+** - Funktioniert normalerweise, aber R12 ist sicherer

**Tipp:** Exportiere als R12/LT2 DXF für maximale Kompatibilität.

### DXF-Import-Tipps

**Vor dem Export aus CAD:**

1. **Zeichnung vereinfachen:**
   - Unnötige Ebenen entfernen
   - Bemaßungen und Anmerkungen löschen
   - 3D-Objekte entfernen (2D-Projektion verwenden)

2. **Einheiten prüfen:**
   - Zeichnungseinheiten verifizieren (mm vs. Zoll)
   - Rayforge geht standardmäßig von mm aus

3. **Ebenen abflachen:**
   - Erwäge, nur relevante Ebenen zu exportieren
   - Konstruktions-Ebenen ausblenden oder löschen

4. **Angemessene Präzision verwenden:**
   - Laser-Präzision ist typischerweise 0,1 mm
   - Präzision nicht überspezifizieren

**Nach dem Import:**

- Maßstab prüfen (DXF-Einheiten benötigen möglicherweise Anpassung)
- Verifizieren, dass alle Pfade korrekt importiert wurden
- Unerwünschte Konstruktionselemente löschen

---

## PDF-Import

PDF-Dateien können Vektorgrafiken, Rasterbilder oder beides enthalten.

### Direkter Vektorimport

Beim Importieren eines PDFs, das Vektorpfade enthält, kann Rayforge diese direkt
importieren – genau wie SVG- oder DXF-Dateien. Das liefert dir saubere, skalierbare
Geometrie ohne Qualitätsverlust durch Rasterung.

Wenn das PDF Ebenen enthält, erkennt Rayforge diese und du kannst auswählen, welche
importiert werden sollen. Jede Ebene wird zu einem separaten Werkstück in deinem
Dokument. Dies funktioniert genauso wie der SVG-Ebenenimport: aktiviere oder
deaktiviere einzelne Ebenen im Import-Dialog vor dem Importieren.

Dies ist besonders nützlich für PDFs, die aus Design-Software wie Illustrator oder
Inkscape exportiert wurden, wo die Vektorpfade sauber und gut organisiert sind.

### Fallback: Rendern und Tracen

Für PDFs, die keine verwendbaren Vektordaten enthalten – gescannte Dokumente,
eingebettete Fotos oder PDFs, bei denen Text nicht in Pfade umgewandelt wurde –
kann Rayforge auf das Rendern des PDFs als Bild und dessen Tracing
zurückgreifen. Dies funktioniert genauso wie der Rasterbild-Import.

### PDF-Import-Tipps

**Beste Ergebnisse:**

1. **Vektor-PDFs verwenden**: PDFs, die aus Vektorsoftware erstellt wurden
   (Illustrator, Inkscape), liefern die saubersten Ergebnisse beim direkten Import.

2. **Auf Ebenen prüfen**: Wenn dein PDF Ebenen hat, werden diese im Import-Dialog
   aufgelistet. Wähle nur die Ebenen aus, die du benötigst.

3. **Für Dokumente mit Text**: Als SVG mit in Pfade konvertierten Schriftarten
   exportieren für beste Qualität, oder das Rendern-und-Tracing-Fallback verwenden.

4. **Import-Dialog-Vorschau verwenden**: Schwellenwert- und Invertierungseinstellungen
   im Tracing-Modus anpassen. Die Vorschau zeigt genau, wie das PDF
   getraced wird.

---

## Ruida-Import

Ruida-Dateien (.rd) sind proprietäre binäre Auftragsdateien, die von Ruida-Controllern
in vielen Laserschneidmaschinen verwendet werden. Diese Dateien enthalten sowohl
Vektorgeometrie als auch Lasereinstellungen, organisiert in Ebenen (Farben).

**Nach dem Import:**

- **Maßstab prüfen** – Verifizieren, dass Abmessungen der erwarteten Größe entsprechen
- **Ebenen überprüfen** – Sicherstellen, dass alle Ebenen korrekt importiert wurden
- **Pfade validieren** – Bestätigen, dass alle Schneidepfade vorhanden sind

### Einschränkungen

- **Nur-Lese-Import** – Ruida-Dateien können nur importiert, nicht exportiert werden
- **Binärformat** – Direktes Bearbeiten von Original-.rd-Dateien nicht unterstützt
- **Proprietäre Funktionen** – Einige fortgeschrittene Ruida-Funktionen werden
  möglicherweise nicht vollständig unterstützt

---

## Rasterbild-Import (PNG, JPG, BMP)

Rasterbilder werden **traced**, um Vektorpfade über den Import-Dialog zu erstellen.

### Tracing-Prozess im Dialog

**Wie es funktioniert:**

1. **Bild geladen** in den Import-Dialog
2. **Live-Vorschau** zeigt das Tracing-Ergebnis
3. **Tracing-Einstellungen** können in Echtzeit angepasst werden
4. **Vektorpfade erstellt** aus den getraceten Kanten
5. **Pfade hinzugefügt** zum Dokument als Werkstücke beim Importieren

### Tracing-Konfiguration im Dialog

Der Import-Dialog bietet diese einstellbaren Parameter:

| Parameter              | Beschreibung           | Effekt                                                     |
| ---------------------- | ---------------------- | ---------------------------------------------------------- |
| **Auto-Schwellenwert** | Automatische Erkennung | Wenn aktiviert, automatisch optimalen Schwellenwert finden |
| **Schwellenwert**      | Schwarz/Weiß-Grenze    | Niedriger = mehr Details, höher = einfacher                |
| **Invertieren**        | Farben umkehren        | Helle Objekte auf dunklem Hintergrund tracen               |

**Standardeinstellungen** funktionieren gut für die meisten Bilder. Der Dialog zeigt
eine Live-Vorschau, die sich beim Anpassen dieser Einstellungen aktualisiert, sodass
du das Tracing vor dem Importieren feinabstimmen kannst.

### Bilder für Tracing vorbereiten

**Für beste Ergebnisse:**

1. **Hoher Kontrast:**
   - Helligkeit/Kontrast im Bildeditor anpassen
   - Klare Unterscheidung zwischen Vordergrund und Hintergrund

2. **Sauberer Hintergrund:**
   - Rauschen und Artefakte entfernen
   - Solider weißer oder transparenter Hintergrund

3. **Angemessene Auflösung:**
   - 300-500 DPI für Fotos
   - Zu hoch = langsames Tracing, zu niedrig = schlechte Qualität

4. **Auf Inhalt zuschneiden:**
   - Unnötige Ränder entfernen
   - Auf den zu gravierenden/schneidenden Bereich fokussieren

5. **In Schwarz-Weiß konvertieren:**
   - Für Schneiden: reines S/W
   - Für Gravur: Graustufen sind in Ordnung

**Bildbearbeitungswerkzeuge:**

- GIMP (kostenlos)
- Photoshop
- Krita (kostenlos)
- Paint.NET (kostenlos, Windows)

### Tracing-Qualität

**Gute Tracing-Kandidaten:**

- Logos mit klaren Kanten
- Hochkontrastbilder
- Strichzeichnungen und Skizzen
- Text (obwohl besser als Vektor)

**Schlechte Tracing-Kandidaten:**

- Niedrigauflösende Bilder
- Fotos mit weichen Kanten
- Bilder mit Farbverläufen
- Sehr detaillierte oder komplexe Fotos

---

## Verwandte Seiten

- [Unterstützte Formate](formats) - Detaillierte Formatspezifikationen
- [G-Code exportieren](exporting) - Ausgabeoptionen
- [Schnellstart](../getting-started/quick-start) - Erstes Import-Tutorial
