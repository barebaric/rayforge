# Kontur-Schneiden

Kontur-Schneiden zeichnet die Umrisse von Vektorformen nach, um sie aus Material zu schneiden. Es ist die häufigste Laser-Operation zum Erstellen von Teilen, Schildern und dekorativen Stücken.

## Übersicht

Kontur-Operationen:

- Folgen Vektorpfaden (Linien, Kurven, Formen)
- Schneiden entlang der Perimeter von Objekten
- Unterstützen einzelne oder mehrere Durchgänge für dicke Materialien
- Können Innen-, Außen- oder Auf-Linie-Schneidepfade verwenden
- Arbeiten mit jeder geschlossenen oder offenen Vektorform

## Wann Kontur verwenden

Verwenden Sie Kontur-Schneiden für:

- Teile aus Rohmaterial schneiden
- Umrisse und Rahmen erstellen
- Formen aus Holz, Acryl, Karton schneiden
- Perforieren oder Ritzen (mit reduzierter Leistung)
- Schablonen und Vorlagen erstellen

**Verwenden Sie Kontur nicht für:**

- Bereiche füllen (verwenden Sie stattdessen [Gravur](engrave))
- Bitmap-Bilder (zuerst in Vektoren konvertieren)

## Eine Kontur-Operation erstellen

### Schritt 1: Objekte auswählen

1. Vektorformen auf der Arbeitsfläche importieren oder zeichnen
2. Die zu schneidenden Objekte auswählen
3. Sicherstellen, dass Formen geschlossene Pfade für vollständige Schnitte sind

### Schritt 2: Kontur-Operation hinzufügen

- **Menü:** Operationen Kontur hinzufügen
- **Tastenkombination:** <kbd>ctrl+shift+c</kbd>
- **Rechtsklick:** Kontextmenü Operation hinzufügen Kontur

### Schritt 3: Einstellungen konfigurieren

![Kontur-Schritt-Einstellungen](/screenshots/step-settings-contour-general.png)

## Haupt-Einstellungen

### Leistung & Geschwindigkeit

**Leistung (%):**

- Laserintensität von 0-100%
- Höhere Leistung für dickere Materialien
- Niedrigere Leistung zum Ritzen oder Markieren

**Geschwindigkeit (mm/min):**

- Wie schnell sich der Laser bewegt
- Langsamer = mehr Energie = tieferer Schnitt
- Schneller = weniger Energie = leichterer Schnitt

### Mehrfach-Durchgang-Schneiden

Für Materialien dicker als ein einzelner Durchgang schneiden kann:

**Durchgänge:**

- Anzahl der Male, den Schnitt zu wiederholen
- Jeder Durchgang schneidet tiefer

**Durchgang-Tiefe (Z-Schritt):**

- Wie viel pro Durchgang die Z-Achse abzusenken ist (falls unterstützt)
- Erfordert Z-Achsen-Steuerung auf Ihrer Maschine
- Erzeugt echtes 2.5D-Schneiden
- Auf 0 setzen für gleich tiefe Mehrfach-Durchgänge

:::warning Z-Achse erforderlich
:::

Durchgang-Tiefe funktioniert nur, wenn Ihre Maschine über Z-Achsen-Steuerung verfügt. Für Maschinen ohne Z-Achse verwenden Sie mehrere Durchgänge auf gleicher Tiefe.

### Pfad-Offset

Steuert, wo der Laser relativ zum Vektorpfad schneidet:

| Offset      | Beschreibung                | Verwendung für                       |
| ----------- | --------------------------- | ------------------------------------ |
| **Auf Linie** | Schneidet direkt auf dem Pfad | Mittellinien-Schnitte, Ritzen        |
| **Innen**   | Schneidet innerhalb der Form | Teile, die exakter Größe entsprechen müssen |
| **Außen**   | Schneidet außerhalb der Form | Löcher, in die Teile passen          |

**Offset-Distanz:**

- Wie weit innen/außen zu offseten (mm)
- Typischerweise auf die Hälfte Ihrer Schnittbreite eingestellt
- Schnittbreite = Breite des vom Laser entfernten Materials
- Beispiel: 0.15mm Offset für 0.3mm Schnittbreite

### Schnittrichtung

**Uhrzeigersinn vs. Gegen-Uhrzeigersinn:**

- Beeinflusst, welche Seite des Schnitts mehr Hitze bekommt
- Normalerweise Uhrzeigersinn für Recht-Hand-Regel
- Ändern, wenn eine Seite mehr verbrennt als die andere

**Reihenfolge optimieren:**

- Sortiert automatisch Pfade für minimales Verfahren
- Reduziert Job-Zeit
- Verhindert verpasste Schnitte

## Erweiterte Funktionen

![Kontur-Nachbearbeitungseinstellungen](/screenshots/step-settings-contour-post.png)

### Halte-Laschen

Laschen halten geschnittene Teile während des Schneidens am Rohmaterial befestigt:

- Laschen hinzufügen, um zu verhindern, dass Teile fallen
- Laschen sind kleine ungeschnittene Abschnitte
- Laschen nach Job-Abschluss abbrechen
- Siehe [Halte-Laschen](../holding-tabs) für Details

### Schnittbreiten-Kompensation

Schnittbreite ist die Breite des vom Laserstrahl entfernten Materials:

**Warum es wichtig ist:**

- Ein Kreis, der "auf Linie" geschnitten wird, wird etwas kleiner sein als entworfen
- Der Laser entfernt ~0.2-0.4mm Material (je nach Strahlbreite)

**Wie zu kompensieren:**

1. Schnittbreite auf Testschnitten messen
2. Pfad-Offset = Schnittbreite/2 verwenden
3. Für Teile: um Schnittbreite/2 **nach innen** offseten
4. Für Löcher: um Schnittbreite/2 **nach außen** offseten

Siehe [Schnittbreite](../kerf) für detaillierte Anleitung.

### Ein-/Ausbewegung

Ein- und Ausbewegungen steuern, wo Schnitte beginnen und enden:

**Einbewegung:**

- Allmählicher Eintritt in den Schnitt
- Verhindert Brandflecken am Startpunkt
- Bewegt Laser auf volle Geschwindigkeit, bevor Materialkante getroffen wird

**Ausbewegung:**

- Allmählicher Austritt aus dem Schnitt
- Verhindert Schäden am Endpunkt
- Häufig bei Metallen und Acryl

**Konfiguration:**

- Länge: Wie weit die Einbewegung reicht (mm)
- Winkel: Richtung des Einbewegungs-Pfads
- Typ: Gerade Linie, Bogen oder Spirale

## Tipps & Best Practices

### Material-Testen

**Immer zuerst testen:**

1. Kleine Testformen auf Abfall schneiden
2. Mit konservativen Einstellungen beginnen (niedrigere Leistung, langsamere Geschwindigkeit)
3. Leistung schrittweise erhöhen oder Geschwindigkeit verringern
4. Erfolgreiche Einstellungen aufzeichnen

### Schneide-Reihenfolge

**Beste Praktiken:**

- Gravieren vor Schneiden (hält Material befestigt)
- Innen-Features vor Außen-Perimeter schneiden
- Halte-Laschen für Teile verwenden, die sich bewegen könnten
- Kleinste Teile zuerst schneiden (weniger Vibration)

## Fehlerbehebung

### Schnitte gehen nicht durch Material

- **Erhöhen:** Leistungseinstellung
- **Verringern:** Geschwindigkeitseinstellung
- **Hinzufügen:** Mehr Durchgänge
- **Überprüfen:** Fokus ist korrekt
- **Überprüfen:** Strahl ist sauber (verschmutzte Linse)

### Übermäßiges Verrußen oder Verbrennen

- **Verringern:** Leistungseinstellung
- **Erhöhen:** Geschwindigkeitseinstellung
- **Verwenden:** Luftunterstützung
- **Versuchen:** Mehr schnellere Durchgänge statt einem langsamen
- **Überprüfen:** Material ist für Laserschneiden geeignet

### Teile fallen während des Schneidens heraus

- **Hinzufügen:** [Halte-Laschen](../holding-tabs)
- **Verwenden:** Schneide-Reihenfolge-Optimierung
- **Schneiden:** Innen-Features vor Außen
- **Sicherstellen:** Material ist flach und befestigt

### Inkonsistente Schnitttiefe

- **Überprüfen:** Materialstärke ist einheitlich
- **Überprüfen:** Material ist flach (nicht gewölbt)
- **Überprüfen:** Fokusdistanz ist konsistent
- **Verifizieren:** Laserleistung ist stabil

### Verpasste Ecken oder Kurven

- **Verringern:** Geschwindigkeit (besonders an Ecken)
- **Überprüfen:** Maschinen-Beschleunigungseinstellungen
- **Verifizieren:** Riemen sind straff
- **Reduzieren:** Pfad-Komplexität (Kurven vereinfachen)

## Technische Details

### Koordinatensystem

Kontur-Operationen arbeiten in:

- **Einheiten:** Millimeter (mm)
- **Ursprung:** Hängt von Maschine und Job-Setup ab
- **Koordinaten:** X/Y-Ebene (Z für Mehrfach-Durchgang-Tiefe)

### Pfad-Generierung

Rayforge konvertiert Vektorformen in G-Code:

1. Pfad offseten (falls Innen-/Außen-Schneiden)
2. Pfad-Reihenfolge optimieren (Verfahren minimieren)
3. Ein-/Ausbewegung einfügen (falls konfiguriert)
4. Halte-Laschen hinzufügen (falls konfiguriert)
5. G-Code-Befehle generieren

### G-Code-Befehle

Typischer Kontur-G-Code:

```gcode
G0 X10 Y10          ; Eilgang zum Start
M3 S204             ; Laser an bei 80% Leistung
G1 X50 Y10 F500     ; Zu Punkt schneiden bei 500 mm/min
G1 X50 Y50 F500     ; Zu nächstem Punkt schneiden
G1 X10 Y50 F500     ; Schneiden fortsetzen
G1 X10 Y10 F500     ; Quadrat vervollständigen
M5                  ; Laser aus
```

## Verwandte Themen

- **[Gravur](engrave)** - Bereiche mit Gravurmustern füllen
- **[Halte-Laschen](../holding-tabs)** - Teile während des Schneidens sichern
- **[Schnittbreite](../kerf)** - Schnittgenauigkeit verbessern
- **[Materialtest-Raster](material-test-grid)** - Optimale Leistungs-/Geschwindigkeitseinstellungen finden
