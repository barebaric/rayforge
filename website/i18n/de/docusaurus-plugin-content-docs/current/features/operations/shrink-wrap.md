# Schrumpfumhüllung

Schrumpfumhüllung erzeugt einen effizienten Schneidepfad um mehrere Objekte, indem sie eine Grenze erzeugt, die sich um sie "schrumpft". Sie ist nützlich, um mehrere Teile aus einem Blatt mit minimalem Abfall zu schneiden.

## Übersicht

Schrumpfumhüllungs-Operationen:

- Erstellen Grenzpfade um Gruppen von Objekten
- Minimieren Materialabfall
- Reduzieren Schneidezeit durch Kombinieren von Pfaden
- Unterstützen Offset-Distanzen für Spielraum
- Arbeiten mit jeder Kombination von Vektorformen

## Wann Schrumpfumhüllung verwenden

Verwenden Sie Schrumpfumhüllung für:

- Schneiden mehrerer kleiner Teile aus einem Blatt
- Minimieren von Materialabfall
- Erstellen effizienter Nesting-Grenzen
- Trennen von Gruppen von Teilen
- Reduzieren der Gesamtschneidezeit

**Verwenden Sie Schrumpfumhüllung nicht für:**

- Einzelne Objekte (verwenden Sie stattdessen [Kontur](contour))
- Teile, die individuelle Grenzen benötigen
- Präzise rechteckige Schnitte

## Wie Schrumpfumhüllung funktioniert

Schrumpfumhüllung erzeugt eine Grenze mit einem algorithmus für rechnerische Geometrie:

1. **Starten** mit einer konvexen Hülle um alle Objekte
2. **Schrumpfen** der Grenze nach innen in Richtung der Objekte
3. **Umhüllen** eng um die Objektgruppe
4. **Offset** nach außen um die angegebene Distanz

Das Ergebnis ist ein effizienter Schneidepfad, der der Gesamtform Ihrer Teile folgt, während Spielraum beibehalten wird.

## Eine Schrumpfumhüllungs-Operation erstellen

### Schritt 1: Objekte anordnen

1. Platzieren Sie alle zu umhüllenden Teile auf der Arbeitsfläche
2. Positionieren Sie sie mit gewünschtem Abstand
3. Mehrere separate Gruppen können zusammen schrumpfumhüllt werden

### Schritt 2: Objekte auswählen

1. Wählen Sie alle in die Schrumpfumhüllung einzubeziehenden Objekte aus
2. Können verschiedene Formen, Größen und Typen sein
3. Alle ausgewählten Objekte werden zusammen umhüllt

### Schritt 3: Schrumpfumhüllungs-Operation hinzufügen

- **Menü:** Operationen Schrumpfumhüllung hinzufügen
- **Rechtsklick:** Kontextmenü Operation hinzufügen Schrumpfumhüllung

### Schritt 4: Einstellungen konfigurieren

![Schrumpfumhüllungs-Schritt-Einstellungen](/screenshots/step-settings-shrink-wrap-general.png)

## Haupt-Einstellungen

### Leistung & Geschwindigkeit

Wie andere Schneideoperationen:

**Leistung (%):**

- Laserintensität zum Schneiden
- Gleich wie für [Kontur](contour)-Schneiden

**Geschwindigkeit (mm/min):**

- Wie schnell sich der Laser bewegt
- An die Schnittgeschwindigkeit Ihres Materials anpassen

**Durchgänge:**

- Anzahl der Male, die Grenze zu schneiden
- Normalerweise 1-2 Durchgänge
- Gleich wie Kontur-Schneiden für Ihr Material

### Offset-Distanz

**Offset (mm):**

- Wie viel Spielraum um die Teile
- Distanz von Objekten zur Schrumpfumhüllungs-Grenze
- Größerer Offset = mehr Material um Teile gelassen

**Typische Werte:**

- **2-3mm:** Enge Umhüllung, minimaler Abfall
- **5mm:** Komfortabler Spielraum
- **10mm+:** Extra Material für Handhabung

**Warum Offset wichtig ist:**

- Zu klein: Risiko, in Teile zu schneiden
- Zu groß: Verschwendet Material
- Berücksichtigen: Schnittbreite, Schneidegenauigkeit

### Glättung

Steuert, wie eng die Grenze Objektformen folgt:

**Hohe Glättung:**

- Folgt Objekten enger
- Komplexerer Pfad
- Längere Schneidezeit
- Weniger Materialabfall

**Niedrige Glättung:**

- Einfacherer, gerundeterer Pfad
- Kürzere Schneidezeit
- Etwas mehr Materialabfall

**Empfohlen:** Mittlere Glättung für die meisten Fälle

## Anwendungsfälle

### Chargen-Teile-Produktion

**Szenario:** 20 kleine Teile aus einem großen Blatt schneiden

**Ohne Schrumpfumhüllung:**

- Volle Blatt-Grenze schneiden
- Alles Material um Teile verschwenden
- Lange Schneidezeit

**Mit Schrumpfumhüllung:**

- Enge Grenze um Teilgruppe schneiden
- Material für andere Projekte sparen
- Schnelleres Schneiden (kürzerer Umfang)

### Nesting-Optimierung

**Workflow:**

1. Teile effizient auf Blatt nesten
2. Teile in Abschnitte gruppieren
3. Jeden Abschnitt schrumpfumhüllen
4. Abschnitte separat schneiden

**Vorteile:**

- Fertige Abschnitte entfernen während Fortsetzung
- Einfachere Handhabung geschnittener Teile
- Reduziertes Risiko von Teilbewegung

### Material-Schonung

**Beispiel:** Kleine Teile auf teurem Material

**Prozess:**

1. Teile eng anordnen
2. Schrumpfumhüllung mit 3mm Offset
3. Frei aus Blatt schneiden
4. Verbleibendes Material sparen

**Ergebnis:** Maximale Materialeffizienz

## Mit anderen Operationen kombinieren

### Schrumpfumhüllung + Kontur

Häufiger Workflow:

1. **Kontur**-Operationen auf einzelnen Teilen (Details schneiden)
2. **Schrumpfumhüllung** um die Gruppe (frei aus Blatt schneiden)

**Ausführungsreihenfolge:**

- Zuerst: Details in Teilen schneiden (während befestigt)
- Zuletzt: Schrumpfumhüllung schneidet Gruppe frei

Siehe [Mehrschicht-Workflow](../multi-layer) für Details.

### Schrumpfumhüllung + Raster

**Beispiel:** Gravierte und geschnittene Teile

1. **Raster** Logos auf Teilen gravieren
2. **Kontur** Teilumrisse schneiden
3. **Schrumpfumhüllung** um gesamte Gruppe

**Vorteile:**

- Alle Gravur geschieht während Material befestigt ist
- Finale Schrumpfumhüllung schneidet gesamte Charge frei

## Tipps & Best Practices

![Schrumpfumhüllungs-Nachbearbeitungseinstellungen](/screenshots/step-settings-shrink-wrap-post.png)

### Teil-Abstand

**Optimaler Abstand:**

- 5-10mm zwischen Teilen
- Genug, damit Schrumpfumhüllung separate Objekte unterscheidet
- Nicht so viel, dass Sie Material verschwenden

**Zu nah:**

- Teile können zusammen umhüllt werden
- Schrumpfumhüllung kann Lücken überbrücken
- Schwierig nach dem Schneiden zu trennen

**Zu weit:**

- Verschwendet Material
- Längere Schneidezeit
- Ineffiziente Nutzung des Blattes

### Material-Überlegungen

**Am besten für:**

- Produktionsläufe (viele identische Teile)
- Kleine Teile aus großen Blättern
- Teure Materialien (Abfall minimieren)
- Chargen-Schneide-Jobs

**Nicht ideal für:**

- Einzelne große Teile
- Teile, die gesamtes Blatt füllen
- Wenn Sie volles Blatt schneiden müssen

### Sicherheit

**Immer:**

- Überprüfen, dass Grenze keine Teile überschneidet
- Verifizieren, dass Offset ausreichend ist
- Vorschau im [Simulationsmodus](../simulation-mode)
- Erst auf Abfall testen

**Achten auf:**

- Schrumpfumhüllung schneidet in Teile (Offset erhöhen)
- Teile bewegen sich, bevor Schrumpfumhüllung abgeschlossen ist
- Material-Warping zieht Teile aus der Position

## Erweiterte Techniken

### Mehrere Schrumpfumhüllungen

Separate Grenzen für verschiedene Gruppen erstellen:

**Prozess:**

1. Teile in logische Gruppen anordnen
2. Schrumpfumhüllung Gruppe 1 (obere Teile)
3. Schrumpfumhüllung Gruppe 2 (untere Teile)
4. Gruppen separat schneiden

**Vorteile:**

- Fertige Gruppen während Job entfernen
- Bessere Organisation
- Einfachere Teil-Entnahme

### Verschachtelte Schrumpfumhüllungen

Schrumpfumhüllung innerhalb einer größeren Grenze:

**Beispiel:**

1. Innere Schrumpfumhüllung: Kleine detaillierte Teile
2. Äußere Schrumpfumhüllung: Inklusive größerer Teile
3. Kontur: Volles Blatt-Grenze

**Verwendung für:** Komplexe Multi-Teil-Layouts

### Spielraum-Testen

Vor Produktionslauf:

1. Schrumpfumhüllung erstellen
2. Vorschau mit [Simulationsmodus](../simulation-mode)
3. Verifizieren, dass Spielraum ausreichend ist
4. Überprüfen, dass keine Teile geschnitten werden
5. Test auf Abfallmaterial ausführen

## Fehlerbehebung

### Schrumpfumhüllung schneidet in Teile

- **Erhöhen:** Offset-Distanz
- **Überprüfen:** Teile sind nicht zu nah beieinander
- **Verifizieren:** Schrumpfumhüllungspfad in Vorschau
- **Berücksichtigen:** Schnittbreite (Laserstrahl-Breite)

### Grenze folgt Formen nicht

- **Erhöhen:** Glättungseinstellung
- **Überprüfen:** Teile sind richtig ausgewählt
- **Versuchen:** Kleinerer Offset (könnte zu weit außen umhüllen)

### Teile werden zusammen umhüllt

- **Erhöhen:** Abstand zwischen Teilen
- **Hinzufügen:** Manuelle Konturen um einzelne Teile
- **Aufteilen:** In mehrere Schrumpfumhüllungs-Operationen

### Schneiden dauert zu lange

- **Verringern:** Glättung (einfacherer Pfad)
- **Erhöhen:** Offset (geradere Grenzen)
- **In Betracht ziehen:** Mehrere kleinere Schrumpfumhüllungen

### Teile bewegen sich während des Schneidens

- **Hinzufügen:** Kleine Laschen, um Teile zu halten (siehe [Halte-Laschen](../holding-tabs))
- **Verwenden:** Schneide-Reihenfolge: von innen nach außen
- **Sicherstellen:** Material ist flach und befestigt
- **Überprüfen:** Blatt ist nicht gewölbt

## Technische Details

### Algorithmus

Schrumpfumhüllung verwendet rechnerische Geometrie:

1. **Konvexe Hülle** - Äußere Grenze finden
2. **Alpha-Form** - In Richtung Objekte schrumpfen
3. **Offset** - Um Offset-Distanz erweitern
4. **Vereinfachen** - Basierend auf Glättungseinstellung

### Pfad-Optimierung

Der Grenzpfad wird optimiert für:

- Minimale Gesamtlänge
- Glatte Kurven (basierend auf Glättung)
- Effiziente Start/End-Punkte

### Koordinatensystem

- **Einheiten:** Millimeter (mm)
- **Präzision:** 0.01mm typisch
- **Koordinaten:** Gleich wie Arbeitsbereich

## Verwandte Themen

- **[Kontur-Schneiden](contour)** - Individuelle Objektumrisse schneiden
- **[Mehrschicht-Workflow](../multi-layer)** - Operationen effektiv kombinieren
- **[Halte-Laschen](../holding-tabs)** - Teile während des Schneidens sichern
- **[Simulationsmodus](../simulation-mode)** - Schneidepfade vorschauen
- **[Materialtest-Raster](material-test-grid)** - Optimale Schneideeinstellungen finden
