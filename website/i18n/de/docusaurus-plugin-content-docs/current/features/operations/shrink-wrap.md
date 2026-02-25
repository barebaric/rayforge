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

Verwende Schrumpfumhüllung für:

- Schneiden mehrerer kleiner Teile aus einem Blatt
- Minimieren von Materialabfall
- Erstellen effizienter Nesting-Grenzen
- Trennen von Gruppen von Teilen
- Reduzieren der Gesamtschneidezeit

**Verwende Schrumpfumhüllung nicht für:**

- Einzelne Objekte (verwende stattdessen [Kontur](contour))
- Teile, die individuelle Grenzen benötigen
- Präzise rechteckige Schnitte

## Wie Schrumpfumhüllung funktioniert

Schrumpfumhüllung erzeugt eine Grenze mit einem algorithmus für rechnerische Geometrie:

1. **Starten** mit einer konvexen Hülle um alle Objekte
2. **Schrumpfen** der Grenze nach innen in Richtung der Objekte
3. **Umhüllen** eng um die Objektgruppe
4. **Offset** nach außen um die angegebene Distanz

Das Ergebnis ist ein effizienter Schneidepfad, der der Gesamtform deiner Teile folgt, während Spielraum beibehalten wird.

## Eine Schrumpfumhüllungs-Operation erstellen

### Schritt 1: Objekte anordnen

1. Platziere alle zu umhüllenden Teile auf der Arbeitsfläche
2. Positioniere sie mit gewünschtem Abstand
3. Mehrere separate Gruppen können zusammen schrumpfumhüllt werden

### Schritt 2: Objekte auswählen

1. Wähle alle in die Schrumpfumhüllung einzubeziehenden Objekte aus
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
- An die Schnittgeschwindigkeit deines Materials anpassen

**Durchgänge:**

- Anzahl der Male, die Grenze zu schneiden
- Normalerweise 1-2 Durchgänge
- Gleich wie Kontur-Schneiden für dein Material

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

- Zuerst: Details in Teile schneiden (während befestigt)
- Zuletzt: Schrumpfumhüllung schneidet Gruppe frei

Siehe [Mehrschicht-Workflow](../multi-layer) für Details.

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
