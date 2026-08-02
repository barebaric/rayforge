# Schrumpfumhüllung

Schrumpfumhüllung erzeugt einen effizienten Schneidepfad um mehrere Objekte, indem sie eine Grenze erzeugt, die sich um sie "schrumpft". Sie ist nützlich, um mehrere Teile mit minimalem Abfall aus einem Blatt zu schneiden.

## Übersicht

Schrumpfumhüllungs-Operationen:

- Erzeugen Grenzpfade um Gruppen von Objekten
- Minimieren Materialabfall
- Reduzieren die Schneidezeit durch Kombinieren von Pfaden
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

Schrumpfumhüllung erzeugt eine Grenze mithilfe eines Algorithmus der rechnerischen Geometrie:

1. **Starte** mit einer konvexen Hülle um alle Objekte
2. **Schrumpfe** die Grenze nach innen in Richtung der Objekte
3. **Umhülle** die Objektgruppe eng
4. **Versetzte** um die angegebene Distanz nach außen

Das Ergebnis ist ein effizienter Schneidepfad, der der Gesamtform deiner Teile folgt, während ein Spielraum erhalten bleibt.

## Eine Schrumpfumhüllungs-Operation erstellen

### Schritt 1: Objekte anordnen

1. Platziere alle zu umhüllenden Teile auf der Arbeitsfläche
2. Positioniere sie mit dem gewünschten Abstand
3. Mehrere separate Gruppen können zusammen schrumpfumhüllt werden

### Schritt 2: Objekte auswählen

1. Wähle alle in die Schrumpfumhüllung einzubeziehenden Objekte aus
2. Es können verschiedene Formen, Größen und Typen sein
3. Alle ausgewählten Objekte werden zusammen umhüllt

### Schritt 3: Schrumpfumhüllungs-Operation hinzufügen

- **Menü:** Operationen → Schrumpfumhüllung hinzufügen
- **Rechtsklick:** Kontextmenü → Operation hinzufügen → Schrumpfumhüllung

### Schritt 4: Einstellungen konfigurieren

## Haupt-Einstellungen

Der Schritt-Einstellungen-Dialog hat drei Registerkarten: **Schritt-Einstellungen**, **Laser** und **Nachbearbeitung**. Die Einstellungen werden unten in Registerkarten-Reihenfolge beschrieben.

### Schrumpfumhüllung

![Schrumpfumhüllungs-Schritt-Einstellungen](/screenshots/step-settings-shrink-wrap-general.png)

Die Gruppe **Schrumpfumhüllung** auf der Registerkarte _Schritt-Einstellungen_ steuert, wie die Hülle um den Inhalt passt.

#### Glättung

Steuert, wie eng die Grenze den Objektformen folgt:

**Hohe Glättung:**

- Folgt den Objekten enger
- Komplexerer Pfad
- Längere Schneidezeit
- Weniger Materialabfall

**Niedrige Glättung:**

- Einfacherer, gerundeterer Pfad
- Kürzere Schneidezeit
- Etwas mehr Materialabfall

**Empfohlen:** Mittlere Glättung für die meisten Fälle

#### Schnitt-Seite

Steuert, wo der Laser relativ zum Schrumpfumhüllungs-Pfad schneidet:

| Schnitt-Seite   | Beschreibung                   | Verwendung für                   |
| --------------- | ------------------------------ | -------------------------------- |
| **Mittellinie** | Schneidet direkt auf dem Pfad  | Standardschneiden                |
| **Außen**       | Schneidet außerhalb der Grenze | Den Schnitt etwas größer machen  |
| **Innen**       | Schneidet innerhalb der Grenze | Den Schnitt etwas kleiner machen |

#### Offset-Distanz

**Offset (mm):**

- Wie viel Spielraum um die Teile
- Abstand von den Objekten zur Schrumpfumhüllungs-Grenze
- Größerer Offset = mehr Material um die Teile gelassen

**Typische Werte:**

- **2-3mm:** Enge Umhüllung, minimaler Abfall
- **5mm:** Komfortabler Spielraum
- **10mm+:** Extra Material für die Handhabung

**Warum der Offset wichtig ist:**

- Zu klein: Risiko, in Teile zu schneiden
- Zu groß: Verschwendet Material
- Berücksichtigen: Schnittbreite, Schneidegenauigkeit

### Laser-Einstellungen

![Laser-Einstellungen](/screenshots/step-settings-shrink-wrap-laser.png)

Leistung, Geschwindigkeit und Laserkopf-Auswahl befinden sich auf der Seite **Laser** des Schritt-Einstellungen-Dialogs.

Wie andere Schneideoperationen:

**Leistung (%):**

- Laserintensität zum Schneiden
- Dieselbe wie beim [Kontur](contour)-Schneiden

**Geschwindigkeit (mm/min):**

- Wie schnell sich der Laser bewegt
- An die Schnittgeschwindigkeit deines Materials anpassen

Um die Grenze mehr als einmal zu schneiden, füge einen [Mehrfach-Durchgang](../multi-pass.md)-Nachbearbeitungsprozessor hinzu.

## Anwendungsfälle

### Chargen-Teile-Produktion

**Szenario:** 20 kleine Teile aus einem großen Blatt schneiden

**Ohne Schrumpfumhüllung:**

- Volle Blatt-Grenze schneiden
- Alles Material um die Teile verschwenden
- Lange Schneidezeit

**Mit Schrumpfumhüllung:**

- Enge Grenze um die Teilgruppe schneiden
- Material für andere Projekte sparen
- Schnelleres Schneiden (kürzerer Umfang)

### Nesting-Optimierung

**Workflow:**

1. Teile effizient auf dem Blatt nesten
2. Teile in Abschnitte gruppieren
3. Jeden Abschnitt schrumpfumhüllen
4. Abschnitte separat schneiden

**Vorteile:**

- Fertige Abschnitte können entfernt werden, während du fortfährst
- Einfachere Handhabung geschnittener Teile
- Reduziertes Risiko von Teilbewegung

### Material-Schonung

**Beispiel:** Kleine Teile auf teurem Material

**Prozess:**

1. Teile eng anordnen
2. Schrumpfumhüllung mit 3mm Offset
3. Frei aus dem Blatt schneiden
4. Verbleibendes Material sparen

**Ergebnis:** Maximale Materialeffizienz

## Mit anderen Operationen kombinieren

### Schrumpfumhüllung + Kontur

Häufiger Workflow:

1. **Kontur**-Operationen auf einzelnen Teilen (Details schneiden)
2. **Schrumpfumhüllung** um die Gruppe (frei aus dem Blatt schneiden)

**Ausführungsreihenfolge:**

- Zuerst: Details in die Teile schneiden (während befestigt)
- Zuletzt: Schrumpfumhüllung schneidet die Gruppe frei

Siehe [Mehrschicht-Workflow](../multi-layer.md) für Details.

### Schrumpfumhüllung + Raster

**Beispiel:** Gravur- und Schnittteile

1. **Raster**-Logos auf Teile gravieren
2. **Kontur**-Schnitt der Teilumrisse
3. **Schrumpfumhüllung** um die gesamte Gruppe

**Vorteile:**

- Alle Gravuren erfolgen, während das Material befestigt ist
- Die letzte Schrumpfumhüllung schneidet die gesamte Charge frei

## Nachbearbeitung

![Schrumpfumhüllungs-Nachbearbeitungseinstellungen](/screenshots/step-settings-shrink-wrap-post.png)

Schrumpfumhüllungs-Operationen unterstützen mehrere Nachbearbeitungsoptionen:

- **[Pfad-Glättung](../smooth.md)** - Gezackte Kanten im Grenzpfad reduzieren
- **[Halte-Laschen](../holding-tabs.md)** - Geschnittene Teile am Rohmaterial befestigt halten
- **[Auf Rohmaterial zuschneiden](../crop-to-stock.md)** - Schnitte auf die Materialgrenze beschränken
- **[Pfad-Optimierung](../path-optimization.md)** - Verfahrdistanz reduzieren
- **[Mehrfach-Durchgang](../multi-pass.md)** - Schnitte für dicke Materialien wiederholen
- **[Ein-/Auslauf](../lead-in-out.md)** - Nullleistungs-An- und Abfahrtsbewegungen für sauberere Schnittenden hinzufügen

### Teilabstand

**Optimaler Abstand:**

- 5-10mm zwischen den Teilen
- Genug, damit die Schrumpfumhüllung separate Objekte unterscheiden kann
- Nicht so viel, dass du Material verschwendest

**Zu nah:**

- Teile können zusammen umhüllt werden
- Schrumpfumhüllung kann Lücken überbrücken
- Nach dem Schneiden schwer zu trennen

**Zu weit:**

- Verschwendet Material
- Längere Schneidezeit
- Ineffiziente Blattnutzung

### Material-Überlegungen

**Am besten geeignet für:**

- Produktionsläufe (viele identische Teile)
- Kleine Teile aus großen Blättern
- Teure Materialien (Abfall minimieren)
- Chargen-Schneidejobs

**Nicht ideal für:**

- Einzelne große Teile
- Teile, die das gesamte Blatt ausfüllen
- Wenn du das ganze Blatt schneiden musst

### Sicherheit

**Immer:**

- Überprüfen, dass die Grenze keine Teile überlappt
- Verifizieren, dass der Offset ausreichend ist
- Vorschau in der [3D-Vorschau](../../ui/3d-preview.md)
- Zuerst auf Abfallmaterial testen

**Achten auf:**

- Schrumpfumhüllung, die in Teile schneidet (Offset erhöhen)
- Teile, die sich bewegen, bevor die Schrumpfumhüllung abgeschlossen ist
- Materialverzug, der Teile aus der Position zieht

## Fortgeschrittene Techniken

### Mehrere Schrumpfumhüllungen

Separate Grenzen für verschiedene Gruppen erstellen:

**Prozess:**

1. Teile in logische Gruppen anordnen
2. Gruppe 1 schrumpfumhüllen (obere Teile)
3. Gruppe 2 schrumpfumhüllen (untere Teile)
4. Gruppen separat schneiden

**Vorteile:**

- Fertige Gruppen während des Jobs entfernen
- Bessere Organisation
- Einfachere Teileentnahme

### Verschachtelte Schrumpfumhüllungen

Schrumpfumhüllung innerhalb einer größeren Grenze:

**Beispiel:**

1. Innere Schrumpfumhüllung: Kleine detaillierte Teile
2. Äußere Schrumpfumhüllung: Enthält größere Teile
3. Kontur: Volle Blatt-Grenze

**Verwenden für:** Komplexe Mehrteil-Layouts

### Spielraum-Test

Vor dem Produktionslauf:

1. Schrumpfumhüllung erstellen
2. Mit der [3D-Vorschau](../../ui/3d-preview.md) vorschauen
3. Verifizieren, dass der Spielraum ausreichend ist
4. Prüfen, dass keine Teile geschnitten werden
5. Test auf Abfallmaterial durchführen

## Fehlerbehebung

### Schrumpfumhüllung schneidet in Teile

- **Erhöhen:** Offset-Distanz
- **Überprüfen:** Teile sind nicht zu nah beieinander
- **Verifizieren:** Schrumpfumhüllungs-Pfad in der Vorschau
- **Einrechnen:** Schnittbreite (Laserstrahl-Breite)

### Grenze folgt den Formen nicht

- **Erhöhen:** Glättungs-Einstellung
- **Überprüfen:** Teile sind richtig ausgewählt
- **Versuchen:** Kleinerer Offset (umhüllt möglicherweise zu weit außen)

### Teile werden zusammen umhüllt

- **Erhöhen:** Abstand zwischen den Teilen
- **Hinzufügen:** Manuelle Konturen um einzelne Teile
- **Aufteilen:** In mehrere Schrumpfumhüllungs-Operationen

### Schneiden dauert zu lange

- **Verringern:** Glättung (einfacherer Pfad)
- **Erhöhen:** Offset (geradere Grenzen)
- **In Betracht ziehen:** Mehrere kleinere Schrumpfumhüllungen

### Teile bewegen sich während des Schneidens

- **Hinzufügen:** Kleine Laschen, um Teile zu halten (siehe [Halte-Laschen](../holding-tabs.md))
- **Verwenden:** Schneidreihenfolge: von innen nach außen
- **Sicherstellen:** Material ist flach und befestigt
- **Überprüfen:** Blatt ist nicht verzogen

## Technische Details

### Algorithmus

Schrumpfumhüllung verwendet rechnerische Geometrie:

1. **Konvexe Hülle** - Äußere Grenze finden
2. **Alpha-Form** - In Richtung der Objekte schrumpfen
3. **Offset** - Um die Offset-Distanz erweitern
4. **Vereinfachen** - Basierend auf der Glättungs-Einstellung

### Pfad-Optimierung

Der Grenzpfad wird optimiert für:

- Minimale Gesamtlänge
- Glatte Kurven (basierend auf Glättung)
- Effiziente Start-/Endpunkte

### Koordinatensystem

- **Einheiten:** Millimeter (mm)
- **Präzision:** 0.01mm typisch
- **Koordinaten:** Gleich wie im Arbeitsbereich

## Verwandte Themen

- **[Kontur-Schneiden](contour)** - Einzelne Objektumrisse schneiden
- **[Mehrschicht-Workflow](../multi-layer.md)** - Operationen effektiv kombinieren
- **[Halte-Laschen](../holding-tabs.md)** - Teile während des Schneidens sichern
- **[3D-Vorschau](../../ui/3d-preview.md)** - Schneidepfade vorschauen
- **[Materialtest-Raster](material-test-grid)** - Optimale Schneideeinstellungen finden
