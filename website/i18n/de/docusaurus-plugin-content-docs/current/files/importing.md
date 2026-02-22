# Dateien importieren

Rayforge unterstützt das Importieren verschiedener Dateiformate, sowohl Vektor- als auch Rasterformate. Diese Seite erklärt, wie Sie Dateien importieren und für beste Ergebnisse optimieren.

## Unterstützte Dateiformate

### Vektorformate

| Format    | Erweiterung | Importmethode        | Am besten für                        |
| --------- | ----------- | -------------------- | ------------------------------------ |
| **SVG**   | `.svg`      | Direkte Vektoren oder Nachverfolgung | Vektorgrafiken, Logos, Designs |
| **DXF**   | `.dxf`      | Direkte Vektoren     | CAD-Zeichnungen, technische Designs  |
| **PDF**   | `.pdf`      | Rendern und nachverfolgen | Dokumente mit Vektorinhalt      |
| **Ruida** | `.rd`       | Direkte Vektoren     | Ruida-Controller-Auftragsdateien     |

### Rasterformate

| Format   | Erweiterung  | Importmethode      | Am besten für                         |
| -------- | ------------ | ------------------ | ------------------------------------- |
| **PNG**  | `.png`       | Nachverfolgung zu Vektoren | Fotos, Bilder mit Transparenz |
| **JPEG** | `.jpg`, `.jpeg` | Nachverfolgung zu Vektoren | Fotos, Bilder mit kontinuierlichen Tönen |
| **BMP**  | `.bmp`       | Nachverfolgung zu Vektoren | Einfache Grafiken, Screenshots  |

:::note Raster-Import
:::

Alle Rasterbilder werden **nachverfolgt**, um Vektorpfade zu erstellen, die für Laseroperationen verwendet werden können. Die Qualität hängt von der Nachverfolgungskonfiguration ab.

---

## Dateien importieren

### Der Import-Dialog

Rayforge verfügt über einen einheitlichen Import-Dialog, der Live-Vorschau und Konfigurationsoptionen für alle unterstützten Dateitypen bietet. Der Dialog ermöglicht Ihnen:

- **Import voranschauen**, bevor er zum Dokument hinzugefügt wird
- **Nachverfolgungseinstellungen konfigurieren** für Rasterbilder
- **Importmethode wählen** für SVG-Dateien (direkte Vektoren oder Nachverfolgung)
- **Parameter anpassen** wie Schwellenwert, Invertieren und Auto-Schwellenwert

![Import-Dialog](/screenshots/import-dialog.png)

### Methode 1: Dateimenü

1. **Datei Importieren** (oder Strg+I)
2. **Datei auswählen** aus dem Dateiauswahldialog
3. **Importeinstellungen konfigurieren** im Import-Dialog
4. **Vorschau** des Ergebnisses vor dem Import
5. **Auf Import klicken** um zur Canvas und zum Dokumentenbaum hinzuzufügen

### Methode 2: Ziehen und Ablegen

1. **Datei ziehen** aus Ihrem Dateimanager
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

Beim Importieren von Dateien, die größer als der Arbeitsbereich Ihrer Maschine sind, wird Rayforge automatisch:

1. **Herunterskalieren** des importierten Inhalts, um in die Maschinengrenzen zu passen
2. **Seitenverhältnis beibehalten** während der Skalierung
3. **Zentrieren** des skalierten Inhalts im Arbeitsbereich
4. **Benachrichtigung anzeigen** mit der Option, die Größenänderung rückgängig zu machen

Die Größenänderungs-Benachrichtigung erscheint als Toast-Meldung:

- ⚠️ "Importiertes Element war größer als der Arbeitsbereich und wurde herunterskaliert um zu passen."
- Enthält eine **"Zurücksetzen"-Schaltfläche** um die automatische Größenanpassung rückgängig zu machen
- Der Toast bleibt sichtbar bis er geschlossen wird oder die Zurücksetzen-Aktion ausgeführt wird

Dies stellt sicher, dass Ihre Designs immer in die Fähigkeiten Ihrer Maschine passen, während Sie die Flexibilität haben, die Originalgröße bei Bedarf wiederherzustellen.

---

## SVG-Import

SVG (Scalable Vector Graphics) ist das **empfohlene Format** für Vektor-Designs.

### Importoptionen im Dialog

Beim Importieren von SVG bietet der Import-Dialog einen Umschalter um zwischen zwei Methoden zu wählen:

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

#### 2. Bitmap nachverfolgen

Deaktivieren Sie "Originalvektoren verwenden" um diese Methode zu nutzen.

**Wie es funktioniert:**

- Rendert SVG zuerst in ein Rasterbild
- Verfolgt das gerenderte Bild um Vektoren zu erstellen
- Kompatibler aber weniger präzise

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

Der Import-Dialog zeigt eine Live-Vorschau, wie Ihr SVG importiert wird:

- Vektorpfade werden als blaues Overlay angezeigt
- Für Nachverfolgungsmodus wird das Originalbild mit den nachverfolgten Pfaden angezeigt
- Vorschau aktualisiert sich in Echtzeit beim Ändern der Einstellungen

### SVG-Best Practices

**Bereiten Sie Ihr SVG für beste Ergebnisse vor:**

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

**Tipp:** Exportieren Sie als R12/LT2 DXF für maximale Kompatibilität.

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

   - Erwägen Sie, nur relevante Ebenen zu exportieren
   - Konstruktions-Ebenen ausblenden oder löschen

4. **Angemessene Präzision verwenden:**
   - Laser-Präzision ist typischerweise 0,1mm
   - Präzision nicht überspezifizieren

**Nach dem Import:**

- Maßstab prüfen (DXF-Einheiten benötigen möglicherweise Anpassung)
- Verifizieren, dass alle Pfade korrekt importiert wurden
- Unerwünschte Konstruktionselemente löschen

---

## PDF-Import

PDF-Dateien können Vektorgrafiken, Rasterbilder oder beides enthalten.

### Wie PDF-Import funktioniert

Beim Importieren von PDF-Dateien über den Import-Dialog **rendert Rayforge das PDF** in ein Bild und **verfolgt** es dann, um Vektoren zu erstellen.

**Prozess:**

1. PDF gerendert und im Import-Dialog-Vorschau angezeigt
2. Sie können Nachverfolgungseinstellungen in Echtzeit anpassen
3. Gerendertes Bild mit Vektorisierung mit Ihren Einstellungen nachverfolgt
4. Resultierende Pfade zum Dokument hinzugefügt, wenn Sie auf Import klicken

**Einschränkungen:**

- Text wird gerastert (nicht als Pfade bearbeitbar)
- Vektorqualität hängt vom Rendering-DPI ab
- Mehrseitige PDFs: nur erste Seite importiert

### PDF-Import-Tipps

**Beste Ergebnisse:**

1. **Vektor-PDFs verwenden:**

   - PDFs aus Vektorsoftware erstellt (Illustrator, Inkscape)
   - Nicht gescannte Dokumente oder eingebettete Bilder

2. **SVG exportieren wenn möglich:**

   - Die meisten Design-Software kann SVG direkt exportieren
   - SVG wird bessere Qualität haben als PDF-Import

3. **Für Dokumente mit Text:**

   - Als SVG mit in Pfade konvertierten Schriftarten exportieren
   - Oder PDF mit hohem DPI (600+) rendern und nachverfolgen

4. **Import-Dialog-Vorschau verwenden:**
   - Schwellenwert- und Invertierungseinstellungen für beste Ergebnisse anpassen
   - Vorschau zeigt genau, wie das PDF nachverfolgt wird

---

## Ruida-Import

Ruida-Dateien (.rd) sind proprietäre binäre Auftragsdateien, die von Ruida-Controllern in vielen Laserschneidmaschinen verwendet werden. Diese Dateien enthalten sowohl Vektorgeometrie als auch Lasereinstellungen, organisiert in Ebenen (Farben).

**Nach dem Import:**

- **Maßstab prüfen** – Verifizieren, dass Abmessungen der erwarteten Größe entsprechen
- **Ebenen überprüfen** – Sicherstellen, dass alle Ebenen korrekt importiert wurden
- **Pfade validieren** – Bestätigen, dass alle Schneidepfade vorhanden sind

### Einschränkungen

- **Nur-Lese-Import** – Ruida-Dateien können nur importiert, nicht exportiert werden
- **Binärformat** – Direktes Bearbeiten von Original-.rd-Dateien nicht unterstützt
- **Proprietäre Funktionen** – Einige fortgeschrittene Ruida-Funktionen werden möglicherweise nicht vollständig unterstützt

---

## Rasterbild-Import (PNG, JPG, BMP)

Rasterbilder werden **nachverfolgt**, um Vektorpfade über den Import-Dialog zu erstellen.

### Nachverfolgungsprozess im Dialog

**Wie es funktioniert:**

1. **Bild geladen** in den Import-Dialog
2. **Live-Vorschau** zeigt das nachverfolgte Ergebnis
3. **Nachverfolgungseinstellungen** können in Echtzeit angepasst werden
4. **Vektorpfade erstellt** aus den nachverfolgten Kanten
5. **Pfade hinzugefügt** zum Dokument als Werkstücke beim Importieren

### Nachverfolgungskonfiguration im Dialog

Der Import-Dialog bietet diese einstellbaren Parameter:

| Parameter          | Beschreibung       | Effekt                                              |
| ------------------ | ------------------ | --------------------------------------------------- |
| **Auto-Schwellenwert** | Automatische Erkennung | Wenn aktiviert, findet automatisch optimalen Schwellenwert |
| **Schwellenwert**  | Schwarz/Weiß-Grenze| Niedriger = mehr Details, höher = einfacher         |
| **Invertieren**    | Farben umkehren    | Helle Objekte auf dunklem Hintergrund nachverfolgen |

**Standardeinstellungen** funktionieren gut für die meisten Bilder. Der Dialog zeigt eine Live-Vorschau, die sich beim Anpassen dieser Einstellungen aktualisiert, sodass Sie die Nachverfolgung vor dem Importieren feinabstimmen können.

### Bilder für Nachverfolgung vorbereiten

**Für beste Ergebnisse:**

1. **Hoher Kontrast:**

   - Helligkeit/Kontrast im Bildeditor anpassen
   - Klare Unterscheidung zwischen Vordergrund und Hintergrund

2. **Sauberer Hintergrund:**

   - Rauschen und Artefakte entfernen
   - Solider weißer oder transparenter Hintergrund

3. **Angemessene Auflösung:**

   - 300-500 DPI für Fotos
   - Zu hoch = langsame Nachverfolgung, zu niedrig = schlechte Qualität

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

### Nachverfolgungsqualität

**Gute Nachverfolgungskandidaten:**

- Logos mit klaren Kanten
- Hochkontrastbilder
- Strichzeichnungen und Skizzen
- Text (obwohl besser als Vektor)

**Schlechte Nachverfolgungskandidaten:**

- Niedrigauflösende Bilder
- Fotos mit weichen Kanten
- Bilder mit Farbverläufen
- Sehr detaillierte oder komplexe Fotos

---

## Verwandte Seiten

- [Unterstützte Formate](formats) - Detaillierte Formatspezifikationen
- [G-Code exportieren](exporting) - Ausgabeoptionen
- [Schnellstart](../getting-started/quick-start) - Erstes Import-Tutorial
