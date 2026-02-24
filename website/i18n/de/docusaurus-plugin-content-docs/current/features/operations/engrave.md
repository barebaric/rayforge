# Gravur

Gravur-Operationen füllen Bereiche mit Raster-Scan-Linien und unterstützen mehrere Modi für verschiedene Gravureffekte. Von glatten Graustufen-Fotos bis hin zu 3D-Relief-Effekten - wähle den Modus, der am besten zu deinem Design und Material passt.

## Übersicht

Gravur-Operationen:

- Füllen geschlossene Formen mit Scan-Linien
- Unterstützen mehrere Gravurmodi für verschiedene Effekte
- Arbeiten mit sowohl Vektorformen als auch Bitmap-Bildern
- Verwenden bidirektionales Scannen für Geschwindigkeit
- Erstellen dauerhafte Markierungen auf vielen Materialien

## Gravurmodi

### Variabler-Leistungs-Modus

Der Variabler-Leistungs-Modus variiert die Laserleistung kontinuierlich basierend auf der Bildhelligkeit und erzeugt sanfte Graustufen-Gravur mit allmählichen Übergängen.

**Am besten für:**

- Sanfte Graustufen-Fotos und Bilder
- Natürliche Verläufe und Übergänge
- Porträts und Kunstwerke
- Holz- und Ledergravur

**Hauptmerkmale:**

- Kontinuierliche Leistungsmodulation
- Min/Max-Leistungssteuerung
- Sanfte Verläufe
- Bessere Tonqualität als Dithering

### Konstante-Leistungs-Modus

Der Konstante-Leistungs-Modus graviert bei voller Leistung, wobei ein Schwellenwert bestimmt, welche Pixel graviert werden. Dies erzeugt saubere Schwarz/Weiß-Ergebnisse.

**Am besten für:**

- Text und Logos
- Hochkontrast-Grafiken
- Saubere Schwarz/Weiß-Gravuren
- Einfache Formen und Muster

**Hauptmerkmale:**

- Schwellenwert-basierte Gravur
- Konsistente Leistungsausgabe
- Schneller als Variabler-Leistungs-Modus
- Saubere Kanten

### Dither-Modus

Der Dither-Modus konvertiert Graustufenbilder in binäre Muster unter Verwendung von Dithering-Algorithmen und ermöglicht hochwertige Fotogravur mit besserer Tonwiedergabe als einfache schwellenwertbasierte Methoden.

**Am besten für:**

- Fotogravur auf Holz oder Leder
- Erstellen von Halbton-Stil-Kunstwerken
- Bilder mit sanften Verläufen
- Wenn Standard-Raster nicht genug Details erfasst

**Hauptmerkmale:**

- Mehrere Dithering-Algorithmus-Wahlmöglichkeiten
- Bessere Detailerhaltung
- Wahrgenommene kontinuierliche Töne
- Ideal für Fotografien

### Mehrfach-Tiefen-Modus

Der Mehrfach-Tiefen-Modus erzeugt 3D-Relief-Effekte durch Variation der Laserleistung basierend auf der Bildhelligkeit, mit mehreren Durchgängen für tieferes Schnitzen.

**Am besten für:**

- Erstellen von 3D-Porträts und Kunstwerken
- Gelände- und topografische Karten
- Lithophanen (lichtdurchlässige 3D-Bilder)
- Relief-Logos und Designs
- Relief-Skulpturen

**Hauptmerkmale:**

- Tiefenkarte aus Bildhelligkeit
- Konfigurierbare Min/Max-Tiefe
- Sanfte Verläufe
- Mehrere Durchgänge für tiefere Gravur
- Z-Schritt zwischen Durchgängen

## Wann Gravur verwenden

Verwende Gravur-Operationen für:

- Gravieren von Text und Logos
- Erstellen von Bildern und Fotos auf Holz/Leder
- Füllen von festen Bereichen mit Textur
- Markieren von Teilen und Produkten
- Erstellen von 3D-Relief-Effekten
- Halbton-Stil-Kunstwerke

**Verwende Gravur nicht für:**

- Durchschneiden von Material (verwende stattdessen [Kontur](contour))
- Präzise Umrisse (Raster erzeugt gefüllte Bereiche)
- Feine Linienarbeit (Vektoren sind sauberer)

## Eine Gravur-Operation erstellen

### Schritt 1: Inhalt vorbereiten

Gravur funktioniert mit:

- **Vektorformen** - Mit Scan-Linien gefüllt
- **Text** - In gefüllte Pfade konvertiert
- **Bilder** - In Graustufen konvertiert und graviert

### Schritt 2: Gravur-Operation hinzufügen

- **Menü:** Operationen → Gravur hinzufügen
- **Tastenkombination:** <kbd>ctrl+shift+e</kbd>
- **Rechtsklick:** Kontextmenü → Operation hinzufügen → Gravur

### Schritt 3: Modus wählen

Wähle den Gravurmodus, der am besten zu deinen Bedürfnissen passt:

- **Variable Leistung** - Sanfte Graustufen-Gravur
- **Konstante Leistung** - Saubere Schwarz/Weiß-Gravur
- **Dither** - Hochwertige Fotogravur
- **Mehrfach-Tiefen** - 3D-Relief-Effekte

### Schritt 4: Einstellungen konfigurieren

![Gravur-Schritt-Einstellungen](/screenshots/step-settings-engrave-general-variable.png)

## Gemeinsame Einstellungen

### Leistung & Geschwindigkeit

**Leistung (%):**

- Laserintensität zum Gravieren
- Niedrigere Leistung für leichtere Markierung
- Höhere Leistung für tiefere Gravur

**Geschwindigkeit (mm/min):**

- Wie schnell der Laser scannt
- Schneller = heller, langsamer = dunkler

### Linienabstand

**Linienabstand (mm):**

- Abstand zwischen Scan-Linien
- Kleiner = höhere Qualität, längere Job-Zeit
- Größer = schneller, sichtbare Linien

| Abstand   | Qualität | Geschwindigkeit | Verwendung für              |
| --------- | -------- | --------------- | --------------------------- |
| 0.05mm    | Höchste  | Langsamste      | Fotos, feine Details        |
| 0.1mm     | Hoch     | Mittel          | Text, Logos, Grafiken       |
| 0.2mm     | Mittel   | Schnell         | Feste Füllungen, Texturen   |
| 0.3mm+    | Niedrig  | Schnellste      | Entwurf, Testen             |

**Empfohlen:** 0.1mm für allgemeine Verwendung

:::tip Auflösungs-Anpassung
:::

Für Bilder sollte der Linienabstand der Bildauflösung entsprechen oder diese überschreiten. Wenn dein Bild 10 Pixel/mm (254 DPI) hat, verwende 0.1mm Linienabstand oder kleiner.

### Scan-Richtung

**Scan-Winkel (Grad):**

- Richtung der Scan-Linien
- 0 = horizontal (von links nach rechts)
- 90 = vertikal (von oben nach unten)
- 45 = diagonal

**Warum Winkel ändern?**

- Holzmaserung: Senkrecht zur Maserung für bessere Ergebnisse gravieren
- Musterorientierung: An Design-Ästhetik anpassen
- Bänderung reduzieren: Anderer Winkel kann Unvollkommenheiten verbergen

**Bidirektionales Scannen:**

- **Aktiviert:** Laser graviert in beide Richtungen (schneller)
- **Deaktiviert:** Laser graviert nur von links nach rechts (langsamer, konsistenter)

Für beste Qualität, bidirektional deaktivieren. Für Geschwindigkeit, aktivieren.

### Overscan

**Overscan-Distanz (mm):**

- Wie weit über das Design hinaus der Laser reist, bevor er umkehrt
- Ermöglicht dem Laser, volle Geschwindigkeit zu erreichen, bevor er in das Design eintritt
- Verhindert Brandflecken an Zeilenanfängen/-enden

**Typische Werte:**

- 2-5mm für die meisten Jobs
- Größer für hohe Geschwindigkeiten

Siehe [Overscan](../overscan) für Details.

## Modus-spezifische Einstellungen

### Variabler-Leistungs-Modus-Einstellungen

![Einstellungen für Variabler-Leistungs-Modus](/screenshots/step-settings-engrave-general-variable.png)

**Min. Leistung (%):**

- Laserleistung für hellste Bereiche (weiße Pixel)
- Normalerweise 0-20%
- Höher einstellen, um sehr flache Bereiche zu vermeiden

**Max. Leistung (%):**

- Laserleistung für dunkelste Bereiche (schwarze Pixel)
- Normalerweise 40-80% je nach Material
- Niedriger = subtiles Relief, höher = dramatische Tiefe

**Leistungsbereich-Beispiele:**

| Min | Max | Effekt                    |
| --- | --- | ------------------------- |
| 0%  | 40% | Subtiles, leichtes Relief |
| 10% | 60% | Mittlere Tiefe, sicher    |
| 20% | 80% | Tiefes, dramatisches Relief |

**Invertieren:**

- **Aus** (Standard): Weiß = flach, Schwarz = tief
- **Ein**: Weiß = tief, Schwarz = flach

Invertieren für Lithophanen verwenden (helle Bereiche sollten dünn sein) oder Prägen (erhabene Bereiche).

**Helligkeitsbereich:**

Steuert, wie Bildhelligkeitswerte auf Laserleistung abgebildet werden. Das Histogramm zeigt die Verteilung der Helligkeitswerte in deinem Bild.

- **Auto-Levels** (Standard): Passt die Schwarz- und Weißpunkte automatisch basierend auf dem Bildinhalt an. Werte unterhalb des Schwarzpunkts werden als Schwarz behandelt, Werte oberhalb des Weißpunkts als Weiß. Dies streckt den Kontrast des Bildes, um den vollen Leistungsbereich zu nutzen.
- **Manueller Modus**: Auto-Levels deaktivieren, um Schwarz- und Weißpunkte manuell durch Ziehen der Markierungen auf dem Histogramm einzustellen.

Dies ist besonders nützlich für:
- Niedrigkontrast-Bilder, die Kontrastverbesserung benötigen
- Bilder mit begrenztem Tonwertbereich
- Sicherstellen konsistenter Ergebnisse über verschiedene Quellbilder hinweg

### Konstante-Leistungs-Modus-Einstellungen

![Einstellungen für Konstante-Leistungs-Modus](/screenshots/step-settings-engrave-general-constant_power.png)

**Schwellenwert (0-255):**

- Helligkeits-Grenzwert für Schwarz/Weiß-Trennung
- Niedriger = mehr Schwarz graviert
- Höher = mehr Weiß graviert

**Typische Werte:**

- 128 (50% Grau-Schwellenwert)
- Basierend auf Bildkontrast anpassen

### Dither-Modus-Einstellungen

![Dither-Modus-Einstellungen](/screenshots/step-settings-engrave-general-dither.png)

**Dithering-Algorithmus:**

Wähle den Algorithmus, der am besten zu deinem Bild und Material passt:

| Algorithmus        | Qualität | Geschwindigkeit | Am besten für                         |
| ------------------ | -------- | --------------- | ------------------------------------ |
| Floyd-Steinberg    | Höchste  | Langsamste      | Fotos, Porträts, sanfte Verläufe     |
| Bayer 2x2          | Niedrig  | Schnellste      | Grober Halbton-Effekt                |
| Bayer 4x4          | Mittel   | Schnell         | Ausgewogener Halbton                 |
| Bayer 8x8          | Hoch     | Mittel          | Feine Details, subtile Muster        |

**Floyd-Steinberg** ist Standard und empfohlen für die meisten Fotogravuren. Es verwendet Fehler-Diffusion, um Quantisierungsfehler auf benachbarte Pixel zu verteilen, was natürlich aussehende Ergebnisse erzeugt.

**Bayer-Dithering** erzeugt regelmäßige Muster, die künstlerische Effekte erzeugen können, die traditionellem Halbton-Druck ähneln.

### Mehrfach-Tiefen-Modus-Einstellungen

![Einstellungen für Mehrfach-Tiefen-Modus](/screenshots/step-settings-engrave-general-multi_pass.png)

**Anzahl der Tiefenstufen:**

- Anzahl diskreter Tiefenstufen
- Mehr Stufen = sanftere Verläufe
- Typisch: 5-10 Stufen

**Z-Abstieg pro Stufe (mm):**

- Wie weit zwischen Tiefendurchgängen abgestuft wird
- Erzeugt tiefere Gesamttiefe mit mehreren Durchgängen
- Typisch: 0.1-0.5mm

**Drehwinkel pro Durchgang:**

- Grad, um jeden aufeinanderfolgenden Durchgang zu drehen
- Erzeugt kreuzschraffur-ähnlichen 3D-Effekt
- Typisch: 0-45 Grad

**Invertieren:**

- **Aktiviert:** Weiß = tief, Schwarz = flach
- **Deaktiviert:** Schwarz = tief, Weiß = flach

Invertieren für Lithophanen verwenden (helle Bereiche sollten dünn sein) oder Prägen (erhabene Bereiche).

## Tipps & Best Practices

![Gravur-Nachbearbeitungseinstellungen](/screenshots/step-settings-engrave-post.png)

### Materialauswahl

**Beste Materialien zum Gravieren:**

- Holz (natürliche Variationen erzeugen schöne Ergebnisse)
- Leder (verbrennt zu dunkelbraun/schwarz)
- Eloxiertes Aluminium (entfernt Beschichtung, zeigt Metall)
- Beschichtete Metalle (entfernt Beschichtungsschicht)
- Einige Kunststoffe (erst testen!)

**Herausfordernde Materialien:**

- Klares Acryl (zeigt Gravur nicht gut)
- Metalle ohne Beschichtung (erfordert spezielle Markierungsverbindungen)
- Glas (erfordert spezielle Einstellungen/Beschichtungen)

### Qualitätseinstellungen

**Für beste Qualität:**

- Kleineren Linienabstand verwenden (0.05-0.1mm)
- Bidirektionales Scannen deaktivieren
- Overscan erhöhen (3-5mm)
- Niedrigere Leistung, mehrere Durchgänge verwenden
- Sicherstellen, dass Material flach und befestigt ist

**Für schnellere Gravur:**

- Größeren Linienabstand verwenden (0.15-0.2mm)
- Bidirektionales Scannen aktivieren
- Minimales Overscan (1-2mm)
- Einzelner Durchgang bei höherer Leistung

### Häufige Probleme

**Brandflecken an Zeilenenden:**

- Overscan-Distanz erhöhen
- Beschleunigungseinstellungen überprüfen
- Leistung leicht reduzieren

**Sichtbare Scan-Linien:**

- Linienabstand verringern
- Leistung reduzieren (Überbrennen erzeugt Lücken)
- Überprüfen, dass Material flach ist

**Ungleiche Gravur:**

- Sicherstellen, dass Material flach ist
- Fokuskonsistenz überprüfen
- Laserleistungsstabilität verifizieren
- Laserlinse reinigen

**Bänderung (dunkle/helle Streifen):**

- Bidirektionales Scannen deaktivieren
- Riemenzug überprüfen
- Geschwindigkeit reduzieren
- Anderen Scan-Winkel versuchen

## Fehlerbehebung

### Gravur zu hell

- **Erhöhen:** Leistungseinstellung
- **Verringern:** Geschwindigkeitseinstellung
- **Überprüfen:** Fokus ist korrekt
- **Versuchen:** Mehrere Durchgänge

### Gravur zu dunkel/verbrennend

- **Verringern:** Leistungseinstellung
- **Erhöhen:** Geschwindigkeitseinstellung
- **Erhöhen:** Linienabstand
- **Überprüfen:** Material ist angemessen

### Inkonsistente Dunkelheit

- **Überprüfen:** Material ist flach
- **Überprüfen:** Fokusdistanz ist konsistent
- **Verifizieren:** Laserstrahl ist sauber
- **Testen:** Anderer Bereich des Materials (Maserung variiert)

### Bild sieht pixelig aus

- **Verringern:** Linienabstand
- **Überprüfen:** Quellbild-Auflösung
- **Versuchen:** Kleineren Linienabstand (0.05mm)
- **Verifizieren:** Bild wird nicht hochskaliert

### Scan-Linien sichtbar

- **Verringern:** Linienabstand
- **Reduzieren:** Leistung (Überbrennen erzeugt Lücken)
- **Versuchen:** Anderen Scan-Winkel
- **Sicherstellen:** Materialoberfläche ist glatt

## Verwandte Themen

- **[Kontur-Schneiden](contour)** - Umrisse und Formen schneiden
- **[Overscan](../overscan)** - Gravurqualität verbessern
- **[Materialtest-Raster](material-test-grid)** - Optimale Einstellungen finden
- **[Mehrschicht-Workflow](../multi-layer)** - Gravur mit anderen Operationen kombinieren
