# Werkstückpositionierung

Dieser Leitfaden behandelt alle Methoden, die in Rayforge für die genaue
Positionierung deines Werkstücks und die Ausrichtung deiner Designs vor dem
Schneiden oder Gravieren verfügbar sind.

## Übersicht

Genaue Werkstückpositionierung ist wichtig für:

- **Abfallvermeidung**: Verhindern, dass an der falschen Stelle geschnitten
  wird
- **Präzise Ausrichtung**: Designs auf vorgedruckten Materialien positionieren
- **Wiederholbare Ergebnisse**: Denselben Job mehrmals konsistent ausführen
- **Mehrteilige Jobs**: Mehrere Teile auf einem einzigen Blatt ausrichten

Rayforge bietet mehrere sich ergänzende Werkzeuge zur Positionierung:

| Methode            | Zweck                       | Am besten für                               |
| ------------------ | --------------------------- | ------------------------------------------- |
| **Fokusmodus**     | Laserposition sehen         | Schnelle visuelle Ausrichtung               |
| **Einrahmen**      | Job-Grenzen vorschauen      | Überprüfen, ob Design auf Material passt    |
| **WKS-Nullpunkt**  | Koordinatenursprung setzen  | Wiederholbare Positionierung                |
| **Kamera-Overlay** | Visuelle Design-Platzierung | Präzise Ausrichtung auf bestehende Merkmale |

---

## Fokusmodus (Laserzeiger)

Der Fokusmodus schaltet den Laser mit niedriger Leistungsstufe ein und fungiert
als "Laserzeiger", um dir zu helfen, genau zu sehen, wo der Laserkopf
positioniert ist.

### Fokusmodus aktivieren

1. **Mit deiner Maschine verbinden**
2. **Auf die Fokus-Taste klicken** in der Symbolleiste (Laser-Symbol)
3. Der Laser schaltet sich mit der konfigurierten Fokus-Leistungsstufe ein
4. **Laserkopf bewegen** um die Strahlposition auf deinem Material zu sehen
5. **Fokus-Taste erneut klicken** zum Ausschalten wenn fertig

:::warning Sicherheit
Selbst bei niedriger Leistung kann der Laser die Augen beschädigen. Schau
niemals direkt in den Strahl oder richte ihn auf reflektierende
Oberflächen. Trage geeigneten Augenschutz.
:::

### Fokus-Leistung konfigurieren

Die Fokus-Leistung bestimmt, wie hell der Laserpunkt erscheint:

1. Gehe zu **Einstellungen → Maschine → Laser**
2. Finde die Einstellung **Fokus-Leistung**
3. Setze einen Wert, der den Punkt sichtbar macht, ohne dein Material zu
   markieren
   - Typische Werte: 1-5% für die meisten Materialien
   - Auf 0 setzen, um die Funktion zu deaktivieren

:::tip Die richtige Leistung finden
Beginne mit 1% und erhöhe schrittweise. Der Punkt sollte sichtbar
sein, aber keine Marke auf deinem Material hinterlassen. Dunklere Materialien
benötigen möglicherweise höhere Leistung, um den Punkt deutlich zu sehen.
:::

### Wann den Fokusmodus verwenden

- **Schnelle Ausrichtungskontrollen**: Sehen, ob der Laser ungefähr dort ist,
  wo du erwartest
- **Materialkanten finden**: Zu Ecken bewegen, um Materialplatzierung zu
  verifizieren
- **WKS-Ursprung setzen**: Laser an gewünschten Nullpunkt positionieren vor
  dem WKS-Setzen
- **Home-Position verifizieren**: Überprüfen, dass das Referenzieren korrekt
  funktioniert hat

---

## Einrahmen

Einrahmen zeichnet das Begrenzungsrechteck deines Jobs bei niedriger (oder
keiner) Leistung nach und zeigt genau, wo dein Design geschnitten oder graviert
wird.

### Wie man einrahmt

1. **Design laden und positionieren** in Rayforge
2. **Klicke Maschine → Rahmen** oder drücke `Strg+F`
3. Der Laserkopf zeichnet den Begrenzungsrahmen deines Jobs nach
4. **Den Umriss verifizieren**, dass er in dein Material passt

### Rahmen-Einstellungen

Rahmen-Verhalten konfigurieren in **Einstellungen → Maschine → Laser**:

- **Rahmen-Geschwindigkeit**: Wie schnell sich der Kopf während des Einrahmens
  bewegt (langsamer = leichter zu sehen)
- **Rahmen-Leistung**: Laserleistung während des Einrahmens
  - Auf 0 für Luft-Einrahmen setzen (Laser aus, nur Bewegung)
  - Auf 1-5% für eine sichtbare Spur auf dem Material setzen

:::tip Luft-Einrahmen vs. Niedrigleistung

- **Luft-Einrahmen (0% Leistung)**: Sicher für jedes Material, aber du siehst
  nur die Kopfbewegung
- **Niedrigleistung-Einrahmen**: Hinterlässt eine schwache sichtbare Marke,
  nützlich für präzise Ausrichtung auf dunklen Materialien
  :::

### Wann einrahmen

- **Vor jedem Job**: Schnelle Verifizierung, dass Design passt
- **Nach Positionsänderungen**: Neue Platzierung bestätigen
- **Teure Materialien**: Doppelt prüfen vor dem Ausführen
- **Mehrteilige Jobs**: Verifizieren, dass alle Teile auf das Material passen

Siehe [Deinen Job einrahmen](framing-your-job) für weitere Details.

---

## WKS-Nullpunkt setzen (Arbeitskoordinatensystem)

Arbeitskoordinatensysteme (WKS) ermöglichen es dir, benutzerdefinierte
"Nullpunkte" für deine Jobs zu definieren. Dies erleichtert die Ausrichtung von
Jobs an deine Materialposition.

### Schnelle WKS-Einrichtung

1. **Laserkopf bewegen** zur Ecke deines Materials (oder gewünschter
   Ursprungspunkt)
2. **Kontrollpanel öffnen** (`Strg+L`)
3. **Ein WKS auswählen** (G54 ist das Standard-Arbeitskoordinatensystem)
4. **Auf Null X und Null Y klicken** um aktuelle Position als Ursprung zu
   setzen
5. Der (0,0)-Punkt deines Designs wird nun an dieser Position ausgerichtet

### Koordinatensysteme verstehen

Rayforge verwendet mehrere Koordinatensysteme:

| System      | Beschreibung                                            |
| ----------- | ------------------------------------------------------- |
| **G53**     | Maschinenkoordinaten (fest, kann nicht geändert werden) |
| **G54**     | Arbeitskoordinatensystem 1 (Standard)                   |
| **G55-G59** | Zusätzliche Arbeitskoordinatensysteme                   |

:::tip Mehrere Arbeitsbereiche
Verwende verschiedene WKS-Slots für verschiedene Vorrichtungspositionen.
Zum Beispiel:

- G54 für die linke Seite deines Arbeitsbereichs
- G55 für die rechte Seite
- G56 für eine Rotationsvorrichtung
  :::

### Wann WKS-Nullpunkt setzen

- **Neue Materialplatzierung**: Ursprung an Materialecke ausrichten
- **Vorrichtungsarbeit**: Ursprung auf Vorrichtungsreferenzpunkt setzen
- **Wiederholbare Jobs**: Gleicher Job, verschiedene Positionen
- **Produktionsläufe**: Konsistente Positionierung über mehrere Stücke

Siehe [Arbeitskoordinatensysteme](../general-info/coordinate-systems) für
vollständige Dokumentation.

---

## Kamerabasierte Positionierung

Das Kamera-Overlay zeigt eine Live-Ansicht deines Materials mit deinem Design
überlagert und ermöglicht präzise visuelle Ausrichtung.

### Kamera einrichten

1. **USB-Kamera anschließen** über deinem Arbeitsbereich
2. Gehe zu **Einstellungen → Kamera** und füge dein Kameragerät hinzu
3. **Kamera aktivieren** um das Overlay auf deiner Arbeitsfläche zu sehen
4. **Kamera ausrichten** mit dem Ausrichtungsverfahren (erforderlich für
   genaue Positionierung)

### Kamera-Ausrichtung

Die Kamera-Ausrichtung ordnet Kamerapixel den realen Koordinaten zu:

1. Öffne **Kamera → Kamera ausrichten**
2. Platziere Ausrichtungsmarkierungen an bekannten Positionen (mindestens
   4 Punkte)
3. Gib die realen X/Y-Koordinaten für jeden Punkt ein
4. Klicke auf **Anwenden** um die Transformation zu berechnen

:::tip Ausrichtungsgenauigkeit

- Verwende Punkte, die über deinen gesamten Arbeitsbereich verteilt sind
- Miss Weltkoordinaten sorgfältig mit einem Lineal
- Verwende Maschinenpositionen (zu bekannten Koordinaten bewegen) für
  beste Genauigkeit
  :::

### Positionierung mit Kamera-Overlay

1. **Kamera-Overlay aktivieren** um dein Material zu sehen
2. **Dein Design importieren**
3. **Design ziehen** um es mit im Kamerabild sichtbaren Merkmalen auszurichten
4. **Feinabstimmung** mit Pfeiltasten für pixelgenaue Platzierung
5. **Einrahmen zur Verifizierung** vor dem Ausführen des Jobs

### Wann Kamera-Positionierung verwenden

- **Vorgedruckte Materialien**: Schnitte an bestehenden Drucke ausrichten
- **Unregelmäßige Materialien**: Auf nicht-rechteckigen Stücken positionieren
- **Präzise Platzierung**: Sub-Millimeter-Genauigkeitsanforderungen
- **Komplexe Layouts**: Mehrere Elemente mit spezifischem Abstand

Siehe [Kamera-Integration](../machine/camera) für vollständige Dokumentation.

---

## Empfohlene Arbeitsabläufe

### Grundlegender Positionierungs-Workflow

Für einfache Jobs auf rechteckigen Materialien:

1. **Material platzieren** auf dem Laserbett
2. **Fokusmodus aktivieren** und bewegen um Materialposition zu verifizieren
3. **WKS-Nullpunkt setzen** an der Materialecke
4. **Design positionieren** in der Arbeitsfläche
5. **Job einrahmen** um Platzierung zu verifizieren
6. **Job ausführen**

### Präzisionsausrichtungs-Workflow

Für genaue Platzierung auf vorgedruckten oder markierten Materialien:

1. **Kamera einrichten und ausrichten** (einmalige Einrichtung)
2. **Material platzieren** auf dem Laserbett
3. **Kamera-Overlay aktivieren** um das Material zu sehen
4. **Design importieren und positionieren** visuell auf dem Kamerabild
5. **Kamera deaktivieren** und einrahmen zur Verifizierung
6. **Job ausführen**

### Produktions-Workflow

Für mehrere identische Jobs:

1. **Vorrichtung einrichten** auf dem Laserbett
2. **WKS-Nullpunkt setzen** ausgerichtet auf die Vorrichtung (z.B. G54)
3. **Design laden und konfigurieren**
4. **Einrahmen zur Verifizierung** der Ausrichtung mit der Vorrichtung
5. **Job ausführen**
6. **Material ersetzen** und wiederholen (WKS bleibt gleich)

### Multi-Position-Workflow

Für denselben Job an verschiedenen Positionen:

1. **Mehrere WKS-Positionen einrichten**:
   - Zu Position 1 bewegen, G54-Null setzen
   - Zu Position 2 bewegen, G55-Null setzen
   - Zu Position 3 bewegen, G56-Null setzen
2. **Design laden** (gleiches Design für alle Positionen)
3. **G54 auswählen**, einrahmen und ausführen
4. **G55 auswählen**, einrahmen und ausführen
5. **G56 auswählen**, einrahmen und ausführen

---

## Fehlerbehebung

### Laserpunkt im Fokusmodus nicht sichtbar

- **Fokus-Leistung erhöhen** in den Lasereinstellungen
- **Dunkle Materialien** benötigen möglicherweise höhere Leistung (5-10%)
- **Laserverbindung überprüfen** und sicherstellen, dass Maschine reagiert
- **Fokus-Leistung verifizieren** dass sie nicht auf 0 gesetzt ist

### Kamera-Overlay falsch ausgerichtet

- **Kamera-Ausrichtung erneut ausführen** mit mehr Referenzpunkten
- **Kamerabefestigung überprüfen** - sie könnte sich bewegt haben
- **Weltkoordinaten verifizieren** wurden genau gemessen
- **Siehe Kamera-Fehlerbehebung** in der Kamera-Integration-Dokumentation

---

## Verwandte Themen

- [Deinen Job einrahmen](framing-your-job) - Detaillierte Rahmen-Dokumentation
- [Arbeitskoordinatensysteme](../general-info/coordinate-systems) - WKS-Referenz
- [Kamera-Integration](../machine/camera) - Kamera-Setup und Ausrichtung
- [Kontrollpanel](../ui/bottom-panel) - Bewegungssteuerung und WKS-Verwaltung
- [Schnellstart-Anleitung](../getting-started/quick-start) - Grundlegender Workflow
