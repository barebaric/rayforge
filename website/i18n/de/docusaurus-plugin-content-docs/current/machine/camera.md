---
description: "Kamera-Kalibrierung in Rayforge für präzise Werkstückausrichtung einrichten. Verwenden Sie Ihre Kamera zur Vorschau und Positionierung von Designs auf Materialien."
---

# Kamera-Integration

Rayforge unterstützt die USB-Kamera-Integration für präzise
Materialausrichtung und Positionierung. Die Kamera-Overlay-Funktion ermöglicht
es dir, genau zu sehen, wo dein Laser auf dem Material schneiden oder
gravieren wird, was Rätselraten eliminiert und Materialabfall reduziert.

![Kameraeinstellungen](/screenshots/machine-camera.png)

## Setup-Workflow

Die Einrichtung einer Kamera erfolgt in vier Schritten:

1. **Kamera hinzufügen** — Schließe deine Kamera an und füge sie der
   Maschinenkonfiguration hinzu
2. **Bildeinstellungen anpassen** — Optimiere Helligkeit, Kontrast,
   Weißabgleich und Rauschunterdrückung
3. **Linse kalibrieren** — Korrigiere Verzerrungen mit dem
   Kalibrierungsassistenten oder manuellen Koeffizienten
4. **Kamera ausrichten** — Bilde Kamerapixel auf Maschinenkoordinaten ab für
   präzise Positionierung

Die Schritte 2–4 werden über das Kameraeigenschaften-Panel aufgerufen, wo
Status-Symbole den Fortschritt auf einen Blick zeigen:

- ✓ **Linsenkalibrierung** — Kalibrierung wurde durchgeführt
- ⚠ **Bildausrichtung** — Warnung wenn Ausrichtung wiederholt werden muss (z.
  B. nach Linsenkalibrierung)
- ✓ **Bildausrichtung** — Ausrichtung ist aktuell und gültig

---

## Schritt 1: Kamera hinzufügen

### Hardware-Anforderungen

**Kompatible Kameras:**

- USB-Webcams (am häufigsten)
- Eingebaute Laptop-Kameras (wenn Rayforge auf einem Laptop in der Nähe
  der Maschine läuft)
- Jede Kamera, die von Video4Linux2 (V4L2) unter Linux oder DirectShow
  unter Windows unterstützt wird

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
   - Gib einen beschreibenden Namen ein (z.B. "Obere Kamera",
     "Arbeitsbereich-Kamera")
   - Wähle das Gerät aus dem Dropdown-Menü
     - Unter Linux: `/dev/video0`, `/dev/video1`, usw.
     - Unter Windows: Kamera 0, Kamera 1, usw.

4. **Kamera aktivieren:**
   - Schalte den Kamera-Aktivierungsschalter um
   - Der Live-Feed sollte auf deiner Arbeitsfläche erscheinen

---

## Schritt 2: Bildeinstellungen anpassen

![Bildeinstellungen Dialog](/screenshots/camera-image-settings.png)

Klicke auf **Konfigurieren** neben **Bildeinstellungen** in den
Kameraeigenschaften, um den Bildeinstellungen-Dialog zu öffnen. Optimiere
diese Parameter für eine klare Kameraansicht:

| Einstellung             | Beschreibung                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------- |
| **Helligkeit**          | Gesamtbildhelligkeit (-100 bis +100)                                                   |
| **Kontrast**            | Kantendefinition und Kontrast (0 bis 100)                                              |
| **YUYV bevorzugen**     | Unkomprimiertes YUYV statt MJPEG verwenden. Langsamer, aber kann einige Fehler beheben |
| **Transparenz**         | Overlay-Deckkraft auf Arbeitsfläche (0% undurchsichtig bis 100% transparent)           |
| **Weißabgleich**        | Farbtemperatur-Korrektur (Auto oder 2500-10000K)                                       |
| **Rauschunterdrückung** | Zeitliche Rauschreduzierung (0.0 bis 0.95)                                             |

Die YUYV-Option ist nützlich, wenn deine Kamera grünstichige Bilder im
Standard-MJPEG-Format erzeugt. Beachte, dass YUYV unkomprimiert ist und die
verfügbare Auflösung oder Bildrate an USB-2.0-Verbindungen reduzieren kann.

---

## Schritt 3: Linsenkalibrierung

Wenn deine Kamera ein Weitwinkelobjektiv hat oder schräg montiert ist,
zeigt das Bild möglicherweise sichtbare Krümmung — gerade Linien
erscheinen gebogen, insbesondere in Richtung der Bildränder. Dies nennt
man Linsenverzerrung, und sie kann die Ausrichtung beeinträchtigen,
selbst wenn deine Ausrichtungspunkte sorgfältig gemessen wurden.

Rayforge enthält einen geführten Kalibrierungsassistenten, der diese
Verzerrung automatisch korrigiert. Du kannst auch die
Verzerrungskoeffizienten manuell anpassen.

### Linsenkalibrierungsdialog

![Linsenkalibrierungsdialog](/screenshots/camera-lens-calibration.png)

Öffne den Linsenkalibrierung-Dialog, indem du auf **Konfigurieren**
neben **Linsenkalibrierung** in den Kameraeigenschaften klickst. Hier
kannst du:

- **Verzerrungskoeffizienten manuell anpassen** — Radiale (k1–k3) und
  tangentiale (p1–p2) Verzerrungsparameter feinabstimmen
- **Kalibrierungsassistenten starten** — Klicke auf **Assistent** für
  eine geführte automatische Kalibrierung

Manuelle Anpassungen sind nützlich für die Feinabstimmung, nachdem der
Assistent eine erste Lösung berechnet hat, oder wenn du die ungefähren
Verzerrungswerte für dein Objektiv kennst.

### Kalibrierungsassistent

Der Kalibrierungsassistent führt dich durch das Aufnehmen mehrerer Bilder
einer gedruckten Kalibrierungskarte von verschiedenen Positionen auf dem
Bett. Er berechnet dann automatisch ein Verzerrungsmodell.

**Schritt 1: Kalibrierungskarte konfigurieren**

![Assistent —
Karteneinstellungen](/screenshots/camera-lens-calibration-wizard-card.png)

1. Klicke auf **Assistent** im Linsenkalibrierung-Dialog, um zu starten
2. Gib **Breite** und **Höhe** deiner gedruckten Karte ein
3. Die Vorschau aktualisiert sich in Echtzeit — die Karte sollte etwa
   70% der Kameraansicht abdecken
4. Klicke auf **Als PDF speichern**, um die Karte zum Drucken zu exportieren
5. Drucke die Karte aus und lege sie auf das Laserbett

**Schritt 2: Bilder aufnehmen**

![Assistent —
Aufnahme](/screenshots/camera-lens-calibration-wizard-capture.png)

1. Klicke auf **Weiter**, um in den Aufnahmemodus zu wechseln
2. Positioniere die Kalibrierungskarte an verschiedenen Stellen und
   Winkeln in der Kameraansicht
3. Klicke für jede Position auf **Bild aufnehmen**
4. Strebe mindestens 8 Aufnahmen an, die das gesamte Bild abdecken,
   einschließlich Ecken und Kanten
5. Die Fortschrittsanzeige und Statusanzeigen zeigen die Aufnahmequalität

**Schritt 3: Kalibrierung anwenden**

1. Sobald genügend Aufnahmen gemacht wurden, klicke auf **Kalibrieren**
2. Die berechneten Verzerrungskoeffizienten werden automatisch auf die
   Kamera angewendet
3. Das Kamera-Overlay zeigt nun ein korrigiertes, gerades Bild

---

## Schritt 4: Bildausrichtung

![Bildausrichtung Dialog](/screenshots/camera-image-alignment.png)

Die Kameraausrichtung kalibriert die Beziehung zwischen Kamerapixeln und
realen Koordinaten und ermöglicht so präzises Positionieren.

### Warum Ausrichtung notwendig ist

Die Kamera sieht den Arbeitsbereich von oben, aber das Bild kann:

- Relativ zu den Maschinenachsen gedreht sein
- In X- und Y-Richtung unterschiedlich skaliert sein
- Durch Linsenperspektive verzerrt sein

Die Ausrichtung erstellt eine Transformationsmatrix, die Kamerapixel
Maschinenkoordinaten zuordnet.

### Ausrichtungsprozedur

1. **Ausrichtungsdialog öffnen:**
   - Klicke auf die **Konfigurieren** Taste neben **Bildausrichtung** in den
     Kameraeigenschaften
   - Der Dialog zeigt das Kamerabild mit der aktuellen
     Ausrichtungsüberlagerung

2. **Ausrichtungsmarkierungen platzieren:**
   - Du benötigst mindestens 3 Referenzpunkte (4 empfohlen für bessere
     Genauigkeit)
   - Ausrichtungspunkte sollten über den Arbeitsbereich verteilt sein
   - Verwende bekannte Positionen wie:
     - Maschinen-Home-Position
     - Lineal-Markierungen
     - Vorgeschnittene Ausrichtungslöcher
     - Kalibrierungsraster

3. **Bildpunkte markieren:**
   - Klicke auf das Kamerabild, um einen Punkt an einer bekannten Position zu
     platzieren
   - Das Blasen-Widget erscheint und zeigt Punktkoordinaten an
   - Wiederhole für jeden Referenzpunkt

4. **Weltkoordinaten eingeben:**
   - Gib für jeden Bildpunkt die realen X/Y-Koordinaten in mm ein
   - Dies sind die tatsächlichen Maschinenkoordinaten, an denen sich jeder
     Punkt befindet
   - Miss genau mit einem Lineal oder verwende bekannte Maschinenpositionen

5. **Ausrichtung anwenden:**
   - Klicke auf **Anwenden**, um die Transformation zu berechnen
   - Das Kamera-Overlay ist nun richtig ausgerichtet

6. **Ausrichtung überprüfen:**
   - Bewege den Laserkopf an eine bekannte Position
   - Überprüfe, ob der Laserpunkt mit der erwarteten Position in der
     Kameraansicht übereinstimmt
   - Bei Bedarf durch Neuausrichtung feinabstimmen

### Ausrichtungsstatus

Das Kameraeigenschaften-Panel zeigt den Ausrichtungsstatus mit einem Symbol:

- **Häkchen** — Ausrichtung ist aktuell und gültig
- **Warnung** — Ausrichtung muss wiederholt werden. Dies passiert, wenn die
  Linsenkalibrierung aktualisiert wird, da die Verzerrungskorrektur das
  Kamerabild verändert und die bestehende Ausrichtung ungültig macht. Deine
  Ausrichtungspunkte bleiben erhalten — öffne einfach den Dialog und klicke
  erneut auf **Anwenden**.

### Beispiel-Workflow

1. Laser zur Home-Position (0, 0) bewegen und in der Kamera markieren
2. Laser zu (100, 0) bewegen und in der Kamera markieren
3. Laser zu (100, 100) bewegen und in der Kamera markieren
4. Laser zu (0, 100) bewegen und in der Kamera markieren
5. Exakte Koordinaten für jeden Punkt eingeben
6. Anwenden und verifizieren

:::tip Best Practices

- Verwende Punkte an den Ecken deines Arbeitsbereichs für maximale Abdeckung
- Vermeide es, Punkte in einem Bereich zu clusteren
- Miss Weltkoordinaten sorgfältig - die Genauigkeit hier bestimmt die
  gesamte Ausrichtungsqualität
- Richte neu aus, wenn du die Kamera bewegt oder den Fokusabstand geändert hast
- Richte nach der Aktualisierung der Linsenkalibrierung neu aus
- Speichere deine Ausrichtung - sie bleibt über Sitzungen hinweg erhalten
  :::

---

## Das Kamera-Overlay verwenden

Sobald ausgerichtet, hilft das Kamera-Overlay beim präzisen Positionieren
von Jobs. Schalte es durch Klicken auf das Kamerasymbol in der
Hauptfenster-Symbolleiste ein oder aus.

---

### Mehrere Kameras

Rayforge unterstützt mehrere Kameras für verschiedene Ansichten oder
Maschinen:

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
Überprüfe, ob die Kamera vom System erkannt wird:

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

- Überprüfe den Geräte-Manager für Kamera unter "Kameras" oder
  "Bildgebende Geräte"
- Stelle sicher, dass keine andere Anwendung die Kamera verwendet (Zoom,
  Skype, usw. schließen)
- Versuche einen anderen USB-Port
- Kamera-Treiber aktualisieren

### Kamera zeigt schwarzen Bildschirm

**Problem:** Kamera erkannt, zeigt aber kein Bild.

**Mögliche Ursachen:**

1. **Kamera von anderer Anwendung verwendet** - Andere Video-Apps schließen
2. **Falsches Gerät ausgewählt** - Verschiedene Geräte-IDs ausprobieren
3. **Kamera-Berechtigungen** - Unter Linux Snap sicherstellen, dass
   Kamera-Schnittstelle verbunden ist
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
- Maschinenbewegungsbefehle verwenden, um Laser präzise an bekannten
  Koordinaten zu positionieren
- Nach jeglichen Kamera-Anpassungen neu ausrichten

### Schlechte Bildqualität

**Problem:** Kamerabild ist unscharf, dunkel oder ausgewaschen.

**Lösungen:**

1. **Helligkeit/Kontrast anpassen** in den Kameraeinstellungen
2. **Beleuchtung verbessern** - Konsistente Arbeitsbereich-Beleuchtung
   hinzufügen
3. **Kameraobjektiv reinigen** - Staub und Ablagerungen reduzieren die Klarheit
4. **Fokus überprüfen** - Autofokus funktioniert möglicherweise nicht
   gut; manuell verwenden, falls möglich
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
