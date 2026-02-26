# Arbeitskoordinatensysteme (WCS)

Arbeitskoordinatensysteme (WCS) ermöglichen es dir, mehrere Referenzpunkte auf dem Arbeitsbereich deiner Maschine zu definieren. Dies erleichtert es, denselben Job an verschiedenen Positionen auszuführen, ohne Werkstücke neu zu entwerfen oder neu zu positionieren.

## Koordinatenräume

Rayforge verwendet drei Koordinatenräume, die zusammenarbeiten:

| Raum         | Beschreibung                                                                                                     |
| ------------ | ---------------------------------------------------------------------------------------------------------------- |
| **MACHINE**  | Absolute Koordinaten relativ zur Home-Position der Maschine. Ursprung ist durch die Hardware festgelegt.         |
| **WORKAREA** | Der nutzbare Bereich innerhalb deiner Maschine, unter Berücksichtigung der Ränder um das Bett.                   |
| **WCS**      | Das Koordinatensystem deines Auftrags. Benutzerkonfigurierbarer Ursprung für Design- und Auftragspositionierung. |

:::note Hinweis für Entwickler
Intern verwendet Rayforge ein normalisiertes Koordinatensystem namens WORLD-Raum.
Der WORLD-Raum beschreibt denselben physischen Raum wie der MACHINE-Raum, aber mit einer
festen Konvention: Y-aufwärts mit Ursprung unten-links. Dies vereinfacht interne
Berechnungen und Rendering. Benutzer müssen nicht direkt mit dem WORLD-Raum interagieren.
:::

### MACHINE-Raum

Der MACHINE-Raum ist das absolute Koordinatensystem relativ zur Home-Position
deiner Maschine. Der Ursprung (0,0) wird durch die Referenzfahrt-Konfiguration
deiner Maschine bestimmt.

- **Ursprung**: Home-Position der Maschine (0,0,0) - durch Hardware festgelegt
- **Zweck**: Referenz für alle anderen Koordinatensysteme
- **Fest**: Kann nicht durch Software geändert werden

Die Koordinatenrichtung hängt von deiner Maschinenkonfiguration ab:

- **Ursprungs-Ecke**: Kann oben-links, unten-links, oben-rechts oder unten-rechts sein
- **Achsenrichtung**: X- und Y-Achsen können basierend auf dem Hardware-Setup umgekehrt sein

### WORKAREA-Raum

Der WORKAREA-Raum definiert den nutzbaren Bereich innerhalb deiner Maschine,
unter Berücksichtigung beliebiger Ränder um die Kanten deines Bettes.

- **Ursprung**: Gleiche Ecke wie der MACHINE-Raum-Ursprung
- **Zweck**: Definiert den tatsächlichen Bereich, in dem Jobs ausgeführt werden können
- **Ränder**: Können Ränder haben (links, oben, rechts, unten)

Zum Beispiel, wenn deine Maschine 400×300mm groß ist, aber einen 10mm Rand auf allen Seiten hat,
würde die WORKAREA 380×280mm betragen, beginnend bei Position (10, 10) im MACHINE-Raum.

## WCS verstehen

Denke an WCS als anpassbare "Nullpunkte" für deine Arbeit. Während deine Maschine eine feste Home-Position hat (bestimmt durch Endschalter), ermöglicht es WCS dir zu definieren, wo deine Arbeit beginnen soll.

### Warum WCS verwenden?

- **Mehrere Vorrichtungen**: Mehrere Arbeitsbereiche auf deinem Bett einrichten und zwischen ihnen wechseln
- **Wiederholbare Positionierung**: Denselben Job an verschiedenen Orten ausführen
- **Schnelle Ausrichtung**: Einen Referenzpunkt basierend auf deinem Material oder Werkstück setzen
- **Produktions-Workflows**: Mehrere Jobs über deinen Arbeitsbereich organisieren

## WCS-Typen

Rayforge unterstützt die folgenden Koordinatensysteme:

| System  | Typ      | Beschreibung                                                     |
| ------- | -------- | ---------------------------------------------------------------- |
| **G53** | Maschine | Absolute Maschinenkoordinaten (fest, kann nicht geändert werden) |
| **G54** | Arbeit 1 | Erstes Arbeitskoordinatensystem (Standard)                       |
| **G55** | Arbeit 2 | Zweites Arbeitskoordinatensystem                                 |
| **G56** | Arbeit 3 | Drittes Arbeitskoordinatensystem                                 |
| **G57** | Arbeit 4 | Viertes Arbeitskoordinatensystem                                 |
| **G58** | Arbeit 5 | Fünftes Arbeitskoordinatensystem                                 |
| **G59** | Arbeit 6 | Sechstes Arbeitskoordinatensystem                                |

### Maschinenkoordinaten (G53)

G53 repräsentiert die absolute Position deiner Maschine, mit Null an der Home-Position der Maschine. Dies ist von deiner Hardware festgelegt und kann nicht geändert werden.

**Wann verwenden:**

- Referenzieren und Kalibrieren
- Absolute Positionierung relativ zu Maschinen-Grenzen
- Wenn du auf die physische Maschinenposition verweisen musst

### Arbeitskoordinaten (G54-G59)

Dies sind Offset-Koordinatensysteme, die du definieren kannst. Jedes hat seinen eigenen Nullpunkt, den du überall auf deinem Arbeitsbereich setzen kannst.

**Wann verwenden:**

- Einrichten mehrerer Arbeitsvorrichtungen
- An Material-Positionen ausrichten
- Denselben Job an verschiedenen Orten ausführen

## WCS in der Oberfläche visualisieren

### 2D-Arbeitsfläche

Die 2D-Arbeitsfläche zeigt deinen WCS-Ursprung mit einer grünen Markierung:

- **Grüne Linien**: Zeigen die aktuelle WCS-Ursprungs- (0, 0)-Position
- **Raster-Ausrichtung**: Rasterlinien sind am WCS-Ursprung ausgerichtet, nicht am Maschinenursprung

Die Ursprungsmarkierung bewegt sich, wenn du das aktive WCS oder seinen Offset änderst, und zeigt dir genau, wo deine Arbeit beginnen wird.

### 3D-Vorschau

In der 3D-Vorschau wird WCS anders angezeigt:

- **Raster und Achsen**: Das gesamte Raster erscheint, als ob der WCS-Ursprung der Weltursprung ist
- **Isolierte Ansicht**: Der WCS wird "isoliert" angezeigt - es sieht aus, als ob das Raster am WCS zentriert ist, nicht an der Maschine
- **Beschriftungen**: Koordinatenbeschriftungen sind relativ zum WCS-Ursprung

Dies erleichtert es zu visualisieren, wo dein Job relativ zum ausgewählten Arbeitskoordinatensystem laufen wird.

## WCS auswählen und ändern

### Über die Symbolleiste

1. Das WCS-Dropdown in der Haupt-Symbolleiste lokalisieren (standardmäßig "G53" beschriftet)
2. Klicken, um die verfügbaren Koordinatensysteme zu sehen
3. Das gewünschte WCS auswählen

### Über das Steuerungsfeld

1. Das Steuerungsfeld öffnen (Ansicht → Steuerungsfeld oder Strg+L)
2. Das WCS-Dropdown im Maschinenstatus-Abschnitt finden
3. Das gewünschte WCS aus dem Dropdown auswählen

## WCS-Offsets setzen

Du kannst definieren, wo sich jeder WCS-Ursprung auf deiner Maschine befindet.

### Null an aktueller Position setzen

1. Mit deiner Maschine verbinden
2. Das WCS auswählen, das du konfigurieren möchtest (z.B. G54)
3. Den Laserkopf an die Position verfahren, die (0, 0) sein soll
4. Im Steuerungsfeld auf die Null-Tasten klicken:
   - **X nullen**: Setzt aktuelle X-Position als 0 für das aktive WCS
   - **Y nullen**: Setzt aktuelle Y-Position als 0 für das aktive WCS
   - **Z nullen**: Setzt aktuelle Z-Position als 0 für das aktive WCS

Die Offsets werden im Controller deiner Maschine gespeichert und bleiben über Sitzungen hinweg erhalten.

### Aktuelle Offsets anzeigen

Das Steuerungsfeld zeigt die aktuellen Offsets für das aktive WCS:

- **Aktuelle Offsets**: Zeigt den (X, Y, Z)-Offset vom Maschinenursprung
- **Aktuelle Position**: Zeigt die Position des Laserkopfs im aktiven WCS

## WCS in deinen Jobs

Wenn du einen Job ausführst, verwendet Rayforge das aktive WCS zur Positionierung deiner Arbeit:

1. Designe deinen Job in der Arbeitsfläche
2. Wähle das WCS, das du verwenden möchtest
3. Führe den Job aus - er wird entsprechend dem WCS-Offset positioniert

Derselbe Job kann an verschiedenen Positionen ausgeführt werden, indem einfach das aktive WCS geändert wird.

## Praktische Workflows

### Workflow 1: Mehrere Vorrichtungs-Positionen

Du hast ein großes Bett und möchtest drei Arbeitsbereiche einrichten:

1. **Maschine referenzieren** um eine Referenz zu erstellen
2. **Zum ersten Arbeitsbereich verfahren** und G54-Offset setzen (X nullen, Y nullen)
3. **Zum zweiten Arbeitsbereich verfahren** und G55-Offset setzen
4. **Zum dritten Arbeitsbereich verfahren** und G56-Offset setzen
5. Jetzt kannst du zwischen G54, G55 und G56 wechseln, um Jobs in jedem Bereich auszuführen

### Workflow 2: An Material ausrichten

Du hast ein Materialstück irgendwo auf deinem Bett platziert:

1. **Den Laserkopf verfahren** zur Ecke deines Materials
2. **G54 auswählen** (oder dein bevorzugtes WCS)
3. **X nullen und Y nullen klicken** um die Materialecke als (0, 0) zu setzen
4. **Deinen Job designen** mit (0, 0) als Ursprung
5. **Den Job ausführen** - er wird von der Materialecke starten

### Workflow 3: Produktions-Raster

Du musst dasselbe Teil 10-mal an verschiedenen Positionen schneiden:

1. **Ein Teil designen** in Rayforge
2. **G54-G59-Offsets** für deine gewünschten Positionen einrichten
3. **Den Job ausführen** mit aktivem G54
4. **Zu G55 wechseln** und erneut ausführen
5. **Wiederholen** für jede WCS-Position

## Wichtige Hinweise

### WCS-Einschränkungen

- **G53 kann nicht geändert werden**: Maschinenkoordinaten sind durch Hardware festgelegt
- **Offsets bleiben erhalten**: WCS-Offsets werden im Controller deiner Maschine gespeichert
- **Verbindung erforderlich**: Du musst mit einer Maschine verbunden sein, um WCS-Offsets zu setzen

---

**Verwandte Seiten:**

- [Steuerungsfeld](../ui/control-panel) - Manuelle Steuerung und WCS-Verwaltung
- [Maschineneinrichtung](../machine/general) - Deine Maschine konfigurieren
- [3D-Vorschau](../ui/3d-preview) - Deine Jobs visualisieren
