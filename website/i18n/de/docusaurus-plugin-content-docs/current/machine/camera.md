# Kamera-Integration

Rayforge unterstützt die USB-Kamera-Integration für präzise Materialausrichtung und Positionierung. Die Kamera-Overlay-Funktion ermöglicht es Ihnen, genau zu sehen, wo Ihr Laser auf dem Material schneiden oder gravieren wird, was Rätselraten eliminiert und Materialabfall reduziert.

![Kameraeinstellungen](/screenshots/machine-camera.png)

## Übersicht

Die Kamera-Integration bietet:

- **Live-Video-Overlay** auf der Arbeitsfläche, das Ihr Material in Echtzeit zeigt
- **Bildausrichtung** zur Kalibrierung der Kameraposition relativ zum Laser
- **Visuelle Positionierung** um Jobs präzise auf unregelmäßigen oder vormarkierten Materialien zu platzieren
- **Materialvorschau** vor dem Ausführen von Jobs
- **Unterstützung mehrerer Kameras** für verschiedene Maschinen-Setups

:::tip Anwendungsfälle

- Schnitte auf vorgedruckten Materialien ausrichten
- Arbeiten mit unregelmäßig geformten Materialien
- Präzises Platzieren von Gravuren auf vorhandenen Objekten
- Reduzieren von Testschnitten und Materialabfall
  :::

---

## Kamera-Setup

### Hardware-Anforderungen

**Kompatible Kameras:**

- USB-Webcams (am häufigsten)
- Eingebaute Laptop-Kameras (wenn Rayforge auf einem Laptop in der Nähe der Maschine läuft)
- Jede Kamera, die von Video4Linux2 (V4L2) unter Linux oder DirectShow unter Windows unterstützt wird

**Empfohlenes Setup:**

- Kamera über dem Arbeitsbereich montiert mit klarer Sicht auf das Material
- Konsistente Lichtverhältnisse
- Kamera positioniert, um den Laser-Arbeitsbereich zu erfassen
- Sichere Befestigung, um Kamerabewegungen zu verhindern

### Eine Kamera hinzufügen

1. **Verbinden Sie Ihre Kamera** über USB mit Ihrem Computer

2. **Kameraeinstellungen öffnen:**
   - Navigieren Sie zu **Einstellungen Einstellungen Kamera**
   - Oder verwenden Sie die Kamera-Symbolleistenschaltfläche

3. **Eine neue Kamera hinzufügen:**
   - Klicken Sie auf die "+"
-Taste, um eine Kamera hinzuzufügen
   - Geben Sie einen beschreibenden Namen ein (z.B. "Obere Kamera", "Arbeitsbereich-Kamera")
   - Wählen Sie das Gerät aus dem Dropdown-Menü
     - Unter Linux: `/dev/video0`, `/dev/video1`, usw.
     - Unter Windows: Kamera 0, Kamera 1, usw.

4. **Kamera aktivieren:**
   - Schalten Sie den Kamera-Aktivierungsschalter um
   - Der Live-Feed sollte auf Ihrer Arbeitsfläche erscheinen

5. **Kameraeinstellungen anpassen:**
   - **Helligkeit:** Anpassen, wenn das Material zu dunkel/hell ist
   - **Kontrast:** Kantensichtbarkeit verbessern
   - **Transparenz:** Overlay-Deckkraft steuern (20-50% empfohlen)
   - **Weißabgleich:** Auto oder manuelle Kelvin-Temperatur

---

## Kamera-Ausrichtung

Die Kameraausrichtung kalibriert die Beziehung zwischen Kamerapixeln und realen Koordinaten und ermöglicht so präzises Positionieren.

### Warum Ausrichtung notwendig ist

Die Kamera sieht den Arbeitsbereich von oben, aber das Bild kann:

- Relativ zu den Maschinenachsen gedreht sein
- In X- und Y-Richtung unterschiedlich skaliert sein
- Durch Linsenperspektive verzerrt sein

Die Ausrichtung erstellt eine Transformationsmatrix, die Kamerapixel Maschinenkoordinaten zuordnet.

### Ausrichtungsprozedur

1. **Ausrichtungsdialog öffnen:**
   - Klicken Sie auf die Kamera-Ausrichtungstaste in der Symbolleiste
   - Oder gehen Sie zu **Kamera Kamera ausrichten**

2. **Ausrichtungsmarkierungen platzieren:**
   - Sie benötigen mindestens 3 Referenzpunkte (4 empfohlen für bessere Genauigkeit)
   - Ausrichtungspunkte sollten über den Arbeitsbereich verteilt sein
   - Verwenden Sie bekannte Positionen wie:
     - Maschinen-Home-Position
     - Lineal-Markierungen
     - Vorgeschnittene Ausrichtungslöcher
     - Kalibrierungsraster

3. **Bildpunkte markieren:**
   - Klicken Sie auf das Kamerabild, um einen Punkt an einer bekannten Position zu platzieren
   - Das Blasen-Widget erscheint und zeigt Punktkoordinaten an
   - Wiederholen Sie für jeden Referenzpunkt

4. **Weltkoordinaten eingeben:**
   - Geben Sie für jeden Bildpunkt die realen X/Y-Koordinaten in mm ein
   - Dies sind die tatsächlichen Maschinenkoordinaten, an denen sich jeder Punkt befindet
   - Messen Sie genau mit einem Lineal oder verwenden Sie bekannte Maschinenpositionen

5. **Ausrichtung anwenden:**
   - Klicken Sie auf "Anwenden", um die Transformation zu berechnen
   - Das Kamera-Overlay ist nun richtig ausgerichtet

6. **Ausrichtung überprüfen:**
   - Bewegen Sie den Laserkopf an eine bekannte Position
   - Überprüfen Sie, ob der Laserpunkt mit der erwarteten Position in der Kameraansicht übereinstimmt
   - Bei Bedarf durch Neuausrichtung feinabstimmen

### Ausrichtungs-Tipps

:::tip Best Practices
- Verwenden Sie Punkte an den Ecken Ihres Arbeitsbereichs für maximale Abdeckung
- Vermeiden Sie es, Punkte in einem Bereich zu clusteren
- Messen Sie Weltkoordinaten sorgfältig - die Genauigkeit hier bestimmt die gesamte Ausrichtungsqualität
- Richten Sie neu aus, wenn Sie die Kamera bewegen oder den Fokusabstand ändern
- Speichern Sie Ihre Ausrichtung - sie bleibt über Sitzungen hinweg erhalten
  :::

**Beispiel-Ausrichtungsworkflow:**

1. Laser zur Home-Position (0, 0) bewegen und in der Kamera markieren
2. Laser zu (100, 0) bewegen und in der Kamera markieren
3. Laser zu (100, 100) bewegen und in der Kamera markieren
4. Laser zu (0, 100) bewegen und in der Kamera markieren
5. Exakte Koordinaten für jeden Punkt eingeben
6. Anwenden und verifizieren

---

## Das Kamera-Overlay verwenden

Sobald ausgerichtet, hilft das Kamera-Overlay beim präzisen Positionieren von Jobs.

### Overlay Aktivieren/Deaktivieren

- **Kamera umschalten:** Klicken Sie auf das Kamera-Symbol in der Symbolleiste
- **Transparenz anpassen:** Verwenden Sie den Schieberegler in den Kameraeinstellungen (20-50% funktioniert gut)
- **Bild aktualisieren:** Kamera aktualisiert kontinuierlich während aktiviert

### Jobs mit der Kamera positionieren

**Workflow für präzises Platzieren:**

1. **Kamera-Overlay aktivieren** um Ihr Material zu sehen

2. **Ihr Design importieren** (SVG, DXF, usw.)

3. **Design auf der Arbeitsfläche positionieren:**
   - Ziehen Sie das Design, um es mit in der Kamera sichtbaren Merkmalen auszurichten
   - Verwenden Sie Zoom, um feine Details zu sehen
   - Nach Bedarf drehen/skalieren

4. **Ausrichtung überprüfen:**
   - Verwenden Sie den [Simulationsmodus](../features/simulation-mode), um zu visualisieren
   - Überprüfen Sie, dass Schnitte/Gravuren dort sein werden, wo Sie sie erwarten

5. **Job rahmen** um die Positionierung vor dem Ausführen zu verifizieren

6. **Job mit Zuversicht ausführen**

### Beispiel: Gravieren auf einer vorgedruckten Karte

1. Platzieren Sie die gedruckte Karte auf dem Laserbett
2. Kamera-Overlay aktivieren
3. Ihr Gravurdesign importieren
4. Design ziehen und positionieren, um mit gedruckten Merkmalen auszurichten
5. Position mit Pfeiltasten feinabstimmen
6. Rahmen um zu verifizieren
7. Job ausführen

---

## Kameraeinstellungen-Referenz

### Geräteeinstellungen

| Einstellung     | Beschreibung                  | Werte                                |
| --------------- | ----------------------------- | ------------------------------------ |
| **Name**        | Beschreibender Name für die Kamera | Beliebiger Text                      |
| **Geräte-ID**   | System-Gerätekennung          | `/dev/video0` (Linux), `0` (Windows) |
| **Aktiviert**   | Kamera-Aktivzustand           | Ein/Aus                              |

### Bildanpassung

| Einstellung         | Beschreibung               | Bereich                             |
| ------------------- | -------------------------- | ----------------------------------- |
| **Helligkeit**      | Gesamtbildhelligkeit       | -100 bis +100                       |
| **Kontrast**        | Kantendefinition und Kontrast | 0 bis 100                         |
| **Transparenz**     | Overlay-Deckkraft auf Arbeitsfläche | 0% (undurchsichtig) bis 100% (transparent) |
| **Weißabgleich**    | Farbtemperatur-Korrektur   | Auto oder 2000-10000K               |

### Ausrichtungsdaten

| Eigenschaft                    | Beschreibung                          |
| ------------------------------ | ------------------------------------ |
| **Bildpunkte**                 | Pixelkoordinaten im Kamerabild       |
| **Welpunkte**                  | Reale Maschinenkoordinaten (mm)      |
| **Transformationsmatrix**      | Berechnete Zuordnung (intern)        |

---

## Erweiterte Funktionen

### Kamera-Kalibrierung (Linsenverzerrungskorrektur)

Für präzise Arbeit können Sie die Kamera kalibrieren, um Tonnen-/Kissenverzerrung zu korrigieren:

1. **Drucken Sie ein Schachbrettmuster** (z.B. 8×6 Raster mit 25mm Quadraten)
2. **Erfassen Sie 10+ Bilder** des Musters aus verschiedenen Winkeln/Positionen
3. **Verwenden Sie OpenCV-Kalibrierungstools** um Kameramatrix und Verzerrungskoeffizienten zu berechnen
4. **Kalibrierung anwenden** in Rayforge (erweiterte Einstellungen)

:::note Wann kalibrieren
Linsenverzerrungskorrektur ist nur notwendig für:

- Weitwinkellinsen mit merklicher Tonnungsverzerrung
- Präzisionsarbeit, die <1mm Genauigkeit erfordert
- Große Arbeitsbereiche, in denen sich Verzerrung ansammelt

Die meisten Standard-Webcams funktionieren ohne Kalibrierung für typische Laserarbeit.
:::

### Mehrere Kameras

Rayforge unterstützt mehrere Kameras für verschiedene Ansichten oder Maschinen:

- Mehrere Kameras in den Einstellungen hinzufügen
- Jede Kamera kann unabhängige Ausrichtung haben
- Zwischen Kameras mit dem Kamera-Wähler wechseln
- Anwendungsfälle:
  - Draufsicht + Seitenansicht für 3D-Objekte
  - Verschiedene Kameras für verschiedene Maschinen
  - Weitwinkel + Detailkamera

---

## Fehlerbehebung

### Kamera nicht erkannt

**Problem:** Kamera erscheint nicht in der Geräteliste.

**Lösungen:**

**Linux:**
Überprüfen Sie, ob die Kamera vom System erkannt wird:

```bash
# Videogeräte auflisten
ls -l /dev/video*

# Kamera mit v4l2 überprüfen
v4l2-ctl --list-devices

# Mit einer anderen Anwendung testen
cheese  # oder VLC, usw.
```

**Für Snap-Benutzer:**

```bash
# Kamerazugriff gewähren
sudo snap connect rayforge:camera
```

**Windows:**

- Überprüfen Sie den Geräte-Manager für Kamera unter "Kameras" oder "Bildgebende Geräte"
- Stellen Sie sicher, dass keine andere Anwendung die Kamera verwendet (Zoom, Skype, usw. schließen)
- Versuchen Sie einen anderen USB-Port
- Kamera-Treiber aktualisieren

### Kamera zeigt schwarzen Bildschirm

**Problem:** Kamera erkannt, zeigt aber kein Bild.

**Mögliche Ursachen:**

1. **Kamera von anderer Anwendung verwendet** - Andere Video-Apps schließen
2. **Falsches Gerät ausgewählt** - Verschiedene Geräte-IDs ausprobieren
3. **Kamera-Berechtigungen** - Unter Linux Snap sicherstellen, dass Kamera-Schnittstelle verbunden ist
4. **Hardware-Problem** - Kamera mit anderer Anwendung testen

**Lösungen:**

```bash
# Linux: Kameragerät freigeben
sudo killall cheese  # oder andere Kamera-Apps

# Überprüfen, welcher Prozess die Kamera verwendet
sudo lsof /dev/video0
```

### Ausrichtung nicht genau

**Problem:** Kamera-Overlay stimmt nicht mit realer Laserposition überein.

**Diagnose:**

1. **Unzureichende Ausrichtungspunkte** - Mindestens 4 Punkte verwenden
2. **Messfehler** - Weltkoordinaten doppelt überprüfen
3. **Kamera bewegt** - Neu ausrichten, wenn Kameraposition geändert wurde
4. **Nichtlineare Verzerrung** - Kann Linsenkalibrierung erfordern

**Genauigkeit verbessern:**

- Mehr Ausrichtungspunkte verwenden (6-8 für sehr große Bereiche)
- Punkte über den gesamten Arbeitsbereich verteilen
- Weltkoordinaten sehr sorgfältig messen
- Maschinenbewegungsbefehle verwenden, um Laser präzise an bekannten Koordinaten zu positionieren
- Nach jeglichen Kamera-Anpassungen neu ausrichten

### Schlechte Bildqualität

**Problem:** Kamerabild ist unscharf, dunkel oder ausgewaschen.

**Lösungen:**

1. **Helligkeit/Kontrast anpassen** in den Kameraeinstellungen
2. **Beleuchtung verbessern** - Konsistente Arbeitsbereich-Beleuchtung hinzufügen
3. **Kameraobjektiv reinigen** - Staub und Ablagerungen reduzieren die Klarheit
4. **Fokus überprüfen** - Autofokus funktioniert möglicherweise nicht gut; manuell verwenden, falls möglich
5. **Transparenz vorübergehend reduzieren**, um Kamerabild deutlicher zu sehen
6. **Verschiedene Weißabgleich-Einstellungen** ausprobieren

### Kamera-Verzögerung oder Ruckeln

**Problem:** Live-Kamera-Feed ist abgehackt oder verzögert.

**Lösungen:**

- Kameraauflösung in den Geräteeinstellungen senken (falls zugänglich)
- Andere Anwendungen schließen, die CPU/GPU verwenden
- Grafiktreiber aktualisieren
- Unter Linux sicherstellen, dass V4L2-Backend verwendet wird (automatisch in Rayforge)
- Kamera deaktivieren, wenn nicht benötigt, um Ressourcen zu sparen

---

## Verwandte Seiten

- [Simulationsmodus](../features/simulation-mode) - Ausführungsvorschau mit Kamera-Overlay
- [3D-Vorschau](../ui/3d-preview) - Jobs in 3D visualisieren
- [Jobs rahmen](../features/framing-your-job) - Job-Position verifizieren
- [Allgemeine Einstellungen](general) - Maschinenkonfiguration
