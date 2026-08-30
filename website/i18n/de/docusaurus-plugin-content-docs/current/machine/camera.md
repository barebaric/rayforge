---
description:
  "Kamera-Kalibrierung in Rayforge für präzise Werkstückausrichtung einrichten. Verwenden Sie Ihre
  Kamera zur Vorschau und Positionierung von Designs auf Materialien."
---

# Kamera-Integration

Rayforge unterstützt die USB-Kamera-Integration für präzise Materialausrichtung und Positionierung.
Die Kamera-Overlay-Funktion ermöglicht es dir, genau zu sehen, wo dein Laser auf dem Material
schneiden oder gravieren wird, was Rätselraten eliminiert und Materialabfall reduziert.

![Kameraeinstellungen](/screenshots/machine-settings-camera.webp)

## Setup-Workflow

Die Einrichtung einer Kamera kann entweder über den geführten
[Kamera-Assistenten](#schritt-2-kamera-assistent) erfolgen — ein einziger Ablauf, der
Bildeinstellungen, Linsenkalibrierung und Ausrichtung abdeckt — oder indem du jeden Bereich manuell
über das Kameraeigenschaften-Panel konfigurierst. In jedem Fall umfasst die Einrichtung vier
Bereiche:

1. **Kamera hinzufügen** — Schließe deine Kamera an und füge sie der Maschinenkonfiguration hinzu
2. **Bildeinstellungen anpassen** — Optimiere Helligkeit, Kontrast, Weißabgleich und
   Rauschunterdrückung
3. **Linse kalibrieren** — Korrigiere Verzerrungen mit der automatischen Kalibrierung des
   Kamera-Assistenten oder manuellen Koeffizienten
4. **Kamera ausrichten** — Bilde Kamerapixel auf Maschinenkoordinaten ab für präzise Positionierung

Das Kameraeigenschaften-Panel zeigt Status-Symbole für Kalibrierung und Ausrichtung auf einen Blick:

- ✓ **Linsenkalibrierung** — Kalibrierung wurde durchgeführt
- ⚠ **Bildausrichtung** — Warnung wenn Ausrichtung wiederholt werden muss (z. B. nach
  Linsenkalibrierung)
- ✓ **Bildausrichtung** — Ausrichtung ist aktuell und gültig

---

## Schritt 1: Kamera hinzufügen

### Hardware-Anforderungen

**Kompatible Kameras:**

- USB-Webcams (am häufigsten)
- Eingebaute Laptop-Kameras (wenn Rayforge auf einem Laptop in der Nähe der Maschine läuft)
- Jede Kamera, die von Video4Linux2 (V4L2) unter Linux oder DirectShow unter Windows unterstützt
  wird

**Empfohlenes Setup:**

- Kamera über dem Arbeitsbereich montiert mit klarer Sicht auf das Material
- Konsistente Lichtverhältnisse
- Kamera positioniert, um den Laser-Arbeitsbereich zu erfassen
- Sichere Befestigung, um Kamerabewegungen zu verhindern

### Eine Kamera hinzufügen

1. **Verbinde deine Kamera** über USB mit deinem Computer

2. **Kameraeinstellungen öffnen:**
   - Navigiere zu **Einstellungen → Einstellungen → Kamera**
   - Oder verwende die Kamera-Symbolleistenschaltfläche

3. **Eine neue Kamera hinzufügen:**
   - Klicke auf die "+"-Taste, um eine Kamera hinzuzufügen
   - Gib einen beschreibenden Namen ein (z.B. "Obere Kamera", "Arbeitsbereich-Kamera")
   - Wähle das Gerät aus dem Dropdown-Menü
     - Unter Linux: `/dev/video0`, `/dev/video1`, usw.
     - Unter Windows: Kamera 0, Kamera 1, usw.

4. **Kamera aktivieren:**
   - Schalte den Kamera-Aktivierungsschalter um
   - Der Live-Feed sollte auf deiner Arbeitsfläche erscheinen

---

## Schritt 2: Kamera-Assistent

Der **Kamera-Assistent** führt die gesamte Kamera-Einrichtung in einem einzigen geführten Ablauf
durch und deckt dabei alle drei Bereiche der Reihe nach ab — Bildeinstellungen, Linsenkalibrierung
und Bildausrichtung. Er wird gestartet über:

- Die Zeile **Kamera-Assistent** im Kameraeigenschaften-Panel — klicke auf **Start**
- Automatisch aus dem [Konfigurations-Assistenten](../getting-started/first-time-setup.md), wenn du
  eine Kamera in dessen Kamera-Schritt aktivierst und fortfährst

### Schritt 2.1: Bildeinstellungen anpassen

![Bildeinstellungen Dialog](/screenshots/machine-settings-camera-image-settings.webp)

Die Bildeinstellungen sind die erste Stufe des Kamera-Assistenten — er öffnet sich dort und
ermöglicht dir, Auflösung, Weißabgleich, Helligkeit, Kontrast und Rauschunterdrückung einzustellen.
Wenn du den Assistenten nicht ausgeführt hast oder die von ihm gesetzten Werte anpassen möchtest,
klicke in den Kameraeigenschaften neben **Bildeinstellungen** auf **Configure**, um den
Bildeinstellungen-Dialog zu öffnen. Optimiere diese Parameter für eine klare Kameraansicht:

| Einstellung             | Beschreibung                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------- |
| **Helligkeit**          | Gesamtbildhelligkeit (-100 bis +100)                                                   |
| **Kontrast**            | Kantendefinition und Kontrast (0 bis 100)                                              |
| **YUYV bevorzugen**     | Unkomprimiertes YUYV statt MJPEG verwenden. Langsamer, aber kann einige Fehler beheben |
| **Transparenz**         | Overlay-Deckkraft auf Arbeitsfläche (0% undurchsichtig bis 100% transparent)           |
| **Weißabgleich**        | Farbtemperatur-Korrektur (Auto oder 2500-10000K)                                       |
| **Rauschunterdrückung** | Zeitliche Rauschreduzierung (0.0 bis 0.95)                                             |

Die YUYV-Option ist nützlich, wenn deine Kamera grünstichige Bilder im Standard-MJPEG-Format
erzeugt. Beachte, dass YUYV unkomprimiert ist und die verfügbare Auflösung oder Bildrate an
USB-2.0-Verbindungen reduzieren kann.

### Schritt 2.2: Linsenkalibrierung

Wenn deine Kamera ein Weitwinkelobjektiv hat oder schräg montiert ist, zeigt das Bild möglicherweise
sichtbare Krümmung — gerade Linien erscheinen gebogen, insbesondere in Richtung der Bildränder. Dies
nennt man Linsenverzerrung, und sie kann die Ausrichtung beeinträchtigen, selbst wenn deine
Ausrichtungspunkte sorgfältig gemessen wurden.

Die Linsenkalibrierung ist die zweite Stufe des Kamera-Assistenten. Du kannst wählen, wie die
Verzerrung korrigiert werden soll:

- **Automatic** — nimm Bilder einer gedruckten Kalibrierungskarte auf; der Assistent berechnet das
  Verzerrungsmodell für dich
- **Manual** — gib die radialen (k1–k3) und tangentialen (p1–p2) Koeffizienten von Hand ein
- **Skip** — lasse die Verzerrung unkorrigiert; du kannst später kalibrieren

#### Automatische Kalibrierung

Bei der **Automatic**-Kalibrierung führt dich der Assistent durch das Aufnehmen mehrerer Bilder
einer gedruckten Kalibrierungskarte von verschiedenen Positionen auf dem Bett und berechnet
anschließend automatisch ein Verzerrungsmodell.

![Assistent — Karteneinstellungen](/screenshots/machine-settings-camera-lens-calibration-wizard-card.webp)

1. Gib **Breite** und **Höhe** deiner gedruckten Karte ein. Die Vorschau aktualisiert sich in
   Echtzeit — die Karte sollte etwa 70% der Kameraansicht abdecken.
2. Klicke auf **Save to PDF**, um die Karte zum Drucken zu exportieren, drucke sie dann aus und lege
   sie auf das Laserbett.

![Assistent — Aufnahme](/screenshots/machine-settings-camera-lens-calibration-wizard-capture.webp)

3. Positioniere die Karte in der Kameraansicht an verschiedenen Stellen und Winkeln und klicke für
   jede Position auf **Capture Frame**. Strebe mindestens 8 Aufnahmen an, die das gesamte Bild
   abdecken, einschließlich Ecken und Kanten. Die Fortschrittsanzeige und Statusanzeigen zeigen die
   Aufnahmequalität.
4. Sobald genügend Aufnahmen gemacht wurden, berechnet der Assistent das Verzerrungsmodell und
   wendet es an — das Kamera-Overlay zeigt nun ein korrigiertes, gerades Bild.

#### Manuelle Kalibrierung

![Linsenkalibrierungsdialog](/screenshots/machine-settings-camera-lens-calibration.webp)

Für manuelle Koeffizienten oder zur Feinabstimmung des Ergebnisses nach einer automatischen
Kalibrierung öffne den Linsenkalibrierung-Dialog, indem du in den Kameraeigenschaften neben
**Linsenkalibrierung** auf **Configure** klickst. Hier kannst du die Verzerrungskoeffizienten
manuell anpassen — die radialen (k1–k3) und tangentialen (p1–p2) Parameter feinabstimmen.

### Schritt 2.3: Bildausrichtung

![Bildausrichtung Dialog](/screenshots/machine-settings-camera-image-alignment.webp)

Die Bildausrichtung ist die letzte Stufe des Kamera-Assistenten. Die Kameraausrichtung kalibriert
die Beziehung zwischen Kamerapixeln und realen Koordinaten und ermöglicht so präzises Positionieren.
Der Assistent verwendet dasselbe hier beschriebene Verfahren, und das Anwenden der Ausrichtung
schließt den Assistenten ab.

#### Warum Ausrichtung notwendig ist

Die Kamera sieht den Arbeitsbereich von oben, aber das Bild kann:

- Relativ zu den Maschinenachsen gedreht sein
- In X- und Y-Richtung unterschiedlich skaliert sein
- Durch Linsenperspektive verzerrt sein

Die Ausrichtung erstellt eine Transformationsmatrix, die Kamerapixel Maschinenkoordinaten zuordnet.

#### Ausrichtungsprozedur

1. **Ausrichtungsdialog öffnen:**
   - Klicke in den Kameraeigenschaften auf die **Configure**-Schaltfläche neben **Bildausrichtung**
   - Der Dialog zeigt das Kamerabild mit der aktuellen Ausrichtungsüberlagerung

2. **Ausrichtungsmarkierungen platzieren:**
   - Du benötigst mindestens 3 Referenzpunkte (4 empfohlen für bessere Genauigkeit)
   - Ausrichtungspunkte sollten über den Arbeitsbereich verteilt sein
   - Verwende bekannte Positionen wie:
     - Maschinen-Home-Position
     - Lineal-Markierungen
     - Vorgeschnittene Ausrichtungslöcher
     - Kalibrierungsraster

3. **Bildpunkte markieren:**
   - Klicke auf das Kamerabild, um einen Punkt an einer bekannten Position zu platzieren
   - Das Blasen-Widget erscheint und zeigt Punktkoordinaten an
   - Wiederhole für jeden Referenzpunkt

4. **Weltkoordinaten eingeben:**
   - Gib für jeden Bildpunkt die realen X/Y-Koordinaten in mm ein
   - Dies sind die tatsächlichen Maschinenkoordinaten, an denen sich jeder Punkt befindet
   - Miss genau mit einem Lineal oder verwende bekannte Maschinenpositionen

5. **Ausrichtung anwenden:**
   - Klicke auf **Anwenden**, um die Transformation zu berechnen
   - Das Kamera-Overlay ist nun richtig ausgerichtet

6. **Ausrichtung überprüfen:**
   - Bewege den Laserkopf an eine bekannte Position
   - Überprüfe, ob der Laserpunkt mit der erwarteten Position in der Kameraansicht übereinstimmt
   - Bei Bedarf durch Neuausrichtung feinabstimmen

#### Ausrichtungsstatus

Das Kameraeigenschaften-Panel zeigt den Ausrichtungsstatus mit einem Symbol:

- **Häkchen** — Ausrichtung ist aktuell und gültig
- **Warnung** — Ausrichtung muss wiederholt werden. Dies passiert, wenn die Linsenkalibrierung
  aktualisiert wird, da die Verzerrungskorrektur das Kamerabild verändert und die bestehende
  Ausrichtung ungültig macht. Deine Ausrichtungspunkte bleiben erhalten — öffne einfach den Dialog
  und klicke erneut auf **Anwenden**.

#### Beispiel-Workflow

1. Laser zur Home-Position (0, 0) bewegen und in der Kamera markieren
2. Laser zu (100, 0) bewegen und in der Kamera markieren
3. Laser zu (100, 100) bewegen und in der Kamera markieren
4. Laser zu (0, 100) bewegen und in der Kamera markieren
5. Exakte Koordinaten für jeden Punkt eingeben
6. Anwenden und verifizieren

:::tip Best Practices

- Verwende Punkte an den Ecken deines Arbeitsbereichs für maximale Abdeckung
- Vermeide es, Punkte in einem Bereich zu clusteren
- Miss Weltkoordinaten sorgfältig - die Genauigkeit hier bestimmt die gesamte Ausrichtungsqualität
- Richte neu aus, wenn du die Kamera bewegt oder den Fokusabstand geändert hast
- Richte nach der Aktualisierung der Linsenkalibrierung neu aus
- Speichere deine Ausrichtung - sie bleibt über Sitzungen hinweg erhalten :::

---

## Das Kamera-Overlay verwenden

Sobald ausgerichtet, hilft das Kamera-Overlay beim präzisen Positionieren von Jobs. Schalte es durch
Klicken auf das Kamerasymbol in der Hauptfenster-Symbolleiste ein oder aus.

---

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

**Linux:** Überprüfe, ob die Kamera vom System erkannt wird:

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

- Überprüfe den Geräte-Manager für Kamera unter "Kameras" oder "Bildgebende Geräte"
- Stelle sicher, dass keine andere Anwendung die Kamera verwendet (Zoom, Skype, usw. schließen)
- Versuche einen anderen USB-Port
- Kamera-Treiber aktualisieren

### Kamera zeigt schwarzen Bildschirm

**Problem:** Kamera erkannt, zeigt aber kein Bild.

**Mögliche Ursachen:**

1. **Kamera von anderer Anwendung verwendet** - Andere Video-Apps schließen
2. **Falsches Gerät ausgewählt** - Verschiedene Geräte-IDs ausprobieren
3. **Kamera-Berechtigungen** - Unter Linux Snap sicherstellen, dass Kamera-Schnittstelle verbunden
   ist
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
4. **Fokus überprüfen** - Autofokus funktioniert möglicherweise nicht gut; manuell verwenden, falls
   möglich
5. **Transparenz vorübergehend reduzieren**, um Kamerabild deutlicher zu sehen
6. **Verschiedene Weißabgleich-Einstellungen** ausprobieren
7. **Rauschunterdrückung anpassen**, wenn das Bild körnig erscheint

### Kamera-Verzögerung oder Ruckeln

**Problem:** Live-Kamera-Feed ist abgehackt oder verzögert.

**Lösungen:**

- Kameraauflösung in den Geräteeinstellungen senken (falls zugänglich)
- Andere Anwendungen schließen, die CPU/GPU verwenden
- Grafiktreiber aktualisieren

---

## Verwandte Seiten

- [3D-Vorschau](../ui/3d-preview.md) — Vorschau der Ausführung mit Kamera-Overlay
- [Jobs rahmen](../features/framing-your-job.md) — Job-Position verifizieren
- [Allgemeine Einstellungen](general) — Maschinenkonfiguration
