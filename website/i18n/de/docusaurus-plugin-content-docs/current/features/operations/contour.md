# Kontur-Schneiden

Kontur-Schneiden zeichnet die Umrisse von Vektorformen nach, um sie aus Material zu schneiden. Es
ist die häufigste Laser-Operation zum Erstellen von Teilen, Schildern und dekorativen Stücken.

## Übersicht

Kontur-Operationen:

- Folgen Vektorpfaden (Linien, Kurven, Formen)
- Schneiden entlang der Perimeter von Objekten
- Unterstützen einzelne oder mehrere Durchgänge für dicke Materialien
- Können Innen-, Außen- oder Auf-Linie-Schneidepfade verwenden
- Arbeiten mit jeder geschlossenen oder offenen Vektorform

## Wann Kontur verwenden

Verwende Kontur-Schneiden für:

- Teile aus Rohmaterial schneiden
- Umrisse und Ränder erstellen
- Formen aus Holz, Acryl, Karton schneiden
- Perforieren oder Ritzen (mit reduzierter Leistung)
- Schablonen und Vorlagen erstellen

**Verwende Kontur nicht für:**

- Bereiche füllen (verwende stattdessen [Gravur](engrave))
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

![Kontur-Schritt-Einstellungen](/screenshots/step-settings-contour-general.webp)

## Haupt-Einstellungen

Der Schritt-Einstellungen-Dialog hat drei Registerkarten: **Schritt-Einstellungen**, **Laser** und
**Nachbearbeitung**. Die Einstellungen werden unten in Registerkarten-Reihenfolge beschrieben.

### Kontur-Einstellungen

![Kontur-Schritt-Einstellungen](/screenshots/step-settings-contour-general.webp)

Die Gruppe **Kontur-Einstellungen** auf der Registerkarte _Schritt-Einstellungen_ steuert, wie der
Umriss nachgezeichnet wird.

#### Schnittseite & Pfad-Offset

Steuert, wo der Laser relativ zum Vektorpfad schneidet:

| Offset        | Beschreibung                  | Verwendung für                              |
| ------------- | ----------------------------- | ------------------------------------------- |
| **Auf Linie** | Schneidet direkt auf dem Pfad | Mittellinien-Schnitte, Ritzen               |
| **Innen**     | Schneidet innerhalb der Form  | Teile, die exakter Größe entsprechen müssen |
| **Außen**     | Schneidet außerhalb der Form  | Löcher, in die Teile passen                 |

**Offset-Distanz:**

- Wie weit innen/außen zu offseten (mm)
- Typischerweise auf die Hälfte deiner Schnittbreite eingestellt
- Schnittbreite = Breite des vom Laser entfernten Materials
- Beispiel: 0.15mm Offset für 0.3mm Schnittbreite

#### Schnittreihenfolge

Steuert die Reihenfolge, in der verschachtelte Pfade verarbeitet werden:

**Innen-Außen:**

- Schneidet zuerst Innen-Features und arbeitet sich dann nach außen
- Hält äußere Materialteile am längsten intakt

**Außen-Innen:**

- Schneidet zuerst den äußeren Umfang und bewegt sich dann nach innen
- Hält das Werkstück am längsten am Rohmaterial befestigt

**Empfohlen:** Innen-Außen (Standard)

#### Innere Pfade entfernen

Für Designs mit Löchern oder inneren Ausschnitten kannst du wählen, nur die äußerste Begrenzung
nachzuzeichnen:

- **Innere Pfade entfernen**: Wenn aktiviert, wird nur die äußerste Kontur nachgezeichnet
- Innere Löcher und Ausschnitte werden ignoriert

Dies ist nützlich, wenn du eine Form ausschneiden, aber das Innere erhalten möchtest, zum Beispiel
um einen Rahmen oder Umriss zu erstellen, ohne innere Details zu schneiden.

#### Überlappungsschnitt

Erweitert geschlossene Schnittpfade über ihren Startpunkt hinaus, sodass der Laserstrahl den Anfang
des Schnitts überlappt:

**Überlappungsschnitt:**

- Distanz in Maschineneinheiten, um den Schnitt über die Start-/End-Verbindungsstelle hinaus zu
  verlängern
- Auf **0** setzen, um die Funktion zu deaktivieren (Standard)
- Typische Werte: 1–5 für die meisten Materialien
- Maximum: 100

**Warum Überlappungsschnitt verwenden:**

Am Anfang und Ende einer geschlossenen Kontur dringt der Laser aufgrund von Beschleunigung und
Verzögerung möglicherweise nicht vollständig ein. Der Überlappungsschnitt stellt sicher, dass der
Strahl an der Verbindungsstelle überlappt und einen sauberen, vollständig durchtrennten Schnitt
erzeugt. Dies ist besonders nützlich für:

- Dicke Materialien, bei denen der vollständige Durchtritt knapp ist
- Hochgeschwindigkeitsschnitte, bei denen Beschleunigungseffekte stärker ausgeprägt sind
- Teile, die ohne Nachbearbeitung herausfallen müssen

Der Überlappungsschnitt wird sowohl auf Außenkonturen als auch auf innere Löcher angewendet.

<!-- prettier-ignore-start -->
:::tip[Ein-/Auslauf vs. Überlappungsschnitt]
[Ein-/Auslauf](../lead-in-out.md) fügt Nullleistungs-An-
und Abfahrtsbewegungen vor und nach dem Schnittpfad hinzu. Der Überlappungsschnitt verlängert den
Schnittpfad selbst über die Verbindungsstelle hinaus. Beide können zusammen für optimale
Schnittqualität verwendet werden.
:::
<!-- prettier-ignore-end -->

#### Nachzeichnen mit benutzerdefiniertem Schwellenwert

Wenn du mit Bitmap-Bildern arbeitest, die in Vektoren konvertiert wurden, kannst du steuern, welche
Teile nachgezeichnet werden:

- **Inhalt erneut scannen**: Benutzerdefinierten Helligkeitsschwellenwert für das Nachzeichnen
  aktivieren
- **Nachzeichnungs-Schwellenwert (0.0-1.0)**: Helligkeits-Grenzwert, wenn das erneute Scannen
  aktiviert ist
  - Niedrigere Werte zeichnen nur dunklere Bereiche nach
  - Höhere Werte schließen hellere Bereiche ein

Dies ist nützlich, wenn das standardmäßige Nachzeichnen nicht das gewünschte Detailniveau erfasst.

### Laser-Einstellungen

![Laser-Einstellungen](/screenshots/step-settings-contour-laser.webp)

Leistung, Geschwindigkeit und Laser-Kopf-Auswahl befinden sich auf der Seite **Laser** des
Schritt-Einstellungen-Dialogs.

#### Leistung & Geschwindigkeit

**Leistung (%):**

- Laserintensität von 0-100%
- Höhere Leistung für dickere Materialien
- Niedrigere Leistung zum Ritzen oder Markieren

**Geschwindigkeit (mm/min):**

- Wie schnell sich der Laser bewegt
- Langsamer = mehr Energie = tieferer Schnitt
- Schneller = weniger Energie = leichterer Schnitt

#### Schnittbreiten-Kompensation

Schnittbreite ist die Breite des vom Laserstrahl entfernten Materials:

**Warum es wichtig ist:**

- Ein Kreis, der "auf Linie" geschnitten wird, wird etwas kleiner sein als entworfen
- Der Laser entfernt ~0.2-0.4mm Material (je nach Strahlbreite)

**Wie zu kompensieren:**

1. Schnittbreite auf Testschnitten messen
2. Pfad-Offset = Schnittbreite/2 verwenden
3. Für Teile: um Schnittbreite/2 **nach innen** offseten
4. Für Löcher: um Schnittbreite/2 **nach außen** offseten

Siehe [Schnittbreite](../kerf.md) für detaillierte Anleitung.

## Nachbearbeitung

![Kontur-Nachbearbeitungseinstellungen](/screenshots/step-settings-contour-post.webp)

Kontur-Operationen unterstützen mehrere Nachbearbeitungsoptionen:

- **[Pfad-Glättung](../smooth.md)** - Gezackte Kanten in Schneidepfaden reduzieren
- **[Halte-Laschen](../holding-tabs.md)** - Geschnittene Teile am Rohmaterial befestigt halten
- **[Auf Rohmaterial zuschneiden](../crop-to-stock.md)** - Schnitte auf Materialgrenze beschränken
- **[Pfad-Optimierung](../path-optimization.md)** - Verfahrdistanz zwischen Schnitten reduzieren
- **[Mehrfach-Durchgang](../multi-pass.md)** - Schnitte für dicke Materialien wiederholen
- **[Ein-/Auslauf](../lead-in-out.md)** - Nullleistungs-An- und Abfahrtsbewegungen für sauberere
  Schnittenden hinzufügen

### Mehrfach-Durchgang-Schneiden

Für Materialien dicker als ein einzelner Durchgang schneiden kann:

**Durchgänge:**

- Anzahl der Male, den Schnitt zu wiederholen
- Jeder Durchgang schneidet tiefer

**Durchgang-Tiefe (Z-Schritt):**

- Wie viel pro Durchgang die Z-Achse abzusenken ist (falls unterstützt)
- Erfordert Z-Achsen-Steuerung auf deiner Maschine
- Erzeugt echtes 2.5D-Schneiden
- Auf 0 setzen für gleich tiefe Mehrfach-Durchgänge

<!-- prettier-ignore-start -->
:::warning[Z-Achse erforderlich]
:::
<!-- prettier-ignore-end -->

Durchgang-Tiefe funktioniert nur, wenn deine Maschine über Z-Achsen-Steuerung verfügt. Für Maschinen
ohne Z-Achse verwende mehrere Durchgänge auf gleicher Tiefe.

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

- **Hinzufügen:** [Halte-Laschen](../holding-tabs.md)
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
3. Halte-Laschen hinzufügen (falls konfiguriert)
4. G-Code-Befehle generieren

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
- **[Halte-Laschen](../holding-tabs.md)** - Teile während des Schneidens sichern
- **[Schnittbreite](../kerf.md)** - Schnittgenauigkeit verbessern
- **[Materialtest-Raster](material-test-grid)** - Optimale Leistungs-/Geschwindigkeitseinstellungen
  finden
