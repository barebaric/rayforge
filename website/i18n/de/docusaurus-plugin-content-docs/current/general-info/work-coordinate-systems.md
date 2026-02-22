# Arbeitskoordinatensysteme (WCS)

Arbeitskoordinatensysteme (WCS) ermöglichen es Ihnen, mehrere Referenzpunkte auf dem Arbeitsbereich Ihrer Maschine zu definieren. Dies erleichtert es, denselben Job an verschiedenen Positionen auszuführen, ohne Werkstücke neu zu entwerfen oder neu zu positionieren.

## WCS verstehen

Denken Sie an WCS als anpassbare "Nullpunkte" für Ihre Arbeit. Während Ihre Maschine eine feste Home-Position hat (bestimmt durch Endschalter), ermöglicht es WCS Ihnen zu definieren, wo Ihre Arbeit beginnen soll.

### Warum WCS verwenden?

- **Mehrere Vorrichtungen**: Mehrere Arbeitsbereiche auf Ihrem Bett einrichten und zwischen ihnen wechseln
- **Wiederholbare Positionierung**: Denselben Job an verschiedenen Orten ausführen
- **Schnelle Ausrichtung**: Einen Referenzpunkt basierend auf Ihrem Material oder Werkstück setzen
- **Produktions-Workflows**: Mehrere Jobs über Ihren Arbeitsbereich organisieren

## WCS-Typen

Rayforge unterstützt die folgenden Koordinatensysteme:

| System   | Typ     | Beschreibung                                               |
| -------- | ------- | ---------------------------------------------------------- |
| **G53**  | Maschine| Absolute Maschinenkoordinaten (fest, kann nicht geändert werden) |
| **G54**  | Arbeit 1| Erstes Arbeitskoordinatensystem (Standard)                 |
| **G55**  | Arbeit 2| Zweites Arbeitskoordinatensystem                            |
| **G56**  | Arbeit 3| Drittes Arbeitskoordinatensystem                            |
| **G57**  | Arbeit 4| Viertes Arbeitskoordinatensystem                            |
| **G58**  | Arbeit 5| Fünftes Arbeitskoordinatensystem                            |
| **G59**  | Arbeit 6| Sechstes Arbeitskoordinatensystem                           |

### Maschinenkoordinaten (G53)

G53 repräsentiert die absolute Position Ihrer Maschine, mit Null an der Home-Position der Maschine. Dies ist von Ihrer Hardware festgelegt und kann nicht geändert werden.

**Wann verwenden:**

- Referenzieren und Kalibrieren
- Absolute Positionierung relativ zu Maschinen-Grenzen
- Wenn Sie auf die physische Maschinenposition verweisen müssen

### Arbeitskoordinaten (G54-G59)

Dies sind Offset-Koordinatensysteme, die Sie definieren können. Jedes hat seinen eigenen Nullpunkt, den Sie überall auf Ihrem Arbeitsbereich setzen können.

**Wann verwenden:**

- Einrichten mehrerer Arbeitsvorrichtungen
- An Material-Positionen ausrichten
- Denselben Job an verschiedenen Orten ausführen

## WCS in der Oberfläche visualisieren

### 2D-Arbeitsfläche

Die 2D-Arbeitsfläche zeigt Ihren WCS-Ursprung mit einer grünen Markierung:

- **Grüne Linien**: Zeigen die aktuelle WCS-Ursprungs- (0, 0)-Position
- **Raster-Ausrichtung**: Rasterlinien sind am WCS-Ursprung ausgerichtet, nicht am Maschinenursprung

Die Ursprungsmarkierung bewegt sich, wenn Sie das aktive WCS oder seinen Offset ändern, und zeigt Ihnen genau, wo Ihre Arbeit beginnen wird.

### 3D-Vorschau

In der 3D-Vorschau wird WCS anders angezeigt:

- **Raster und Achsen**: Das gesamte Raster erscheint, als ob der WCS-Ursprung der Weltursprung ist
- **Isolierte Ansicht**: Der WCS wird "isoliert" angezeigt - es sieht aus, als ob das Raster am WCS zentriert ist, nicht an der Maschine
- **Beschriftungen**: Koordinatenbeschriftungen sind relativ zum WCS-Ursprung

Dies erleichtert es zu visualisieren, wo Ihr Job relativ zum ausgewählten Arbeitskoordinatensystem laufen wird.

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

Sie können definieren, wo sich jeder WCS-Ursprung auf Ihrer Maschine befindet.

### Null an aktueller Position setzen

1. Mit Ihrer Maschine verbinden
2. Das WCS auswählen, das Sie konfigurieren möchten (z.B. G54)
3. Den Laserkopf an die Position verfahren, die (0, 0) sein soll
4. Im Steuerungsfeld auf die Null-Tasten klicken:
   - **X nullen**: Setzt aktuelle X-Position als 0 für das aktive WCS
   - **Y nullen**: Setzt aktuelle Y-Position als 0 für das aktive WCS
   - **Z nullen**: Setzt aktuelle Z-Position als 0 für das aktive WCS

Die Offsets werden im Controller Ihrer Maschine gespeichert und bleiben über Sitzungen hinweg erhalten.

### Aktuelle Offsets anzeigen

Das Steuerungsfeld zeigt die aktuellen Offsets für das aktive WCS:

- **Aktuelle Offsets**: Zeigt den (X, Y, Z)-Offset vom Maschinenursprung
- **Aktuelle Position**: Zeigt die Position des Laserkopfs im aktiven WCS

## WCS in Ihren Jobs

Wenn Sie einen Job ausführen, verwendet Rayforge das aktive WCS, um Ihre Arbeit zu positionieren:

1. Ihren Job in der Arbeitsfläche entwerfen
2. Das WCS auswählen, das Sie verwenden möchten
3. Den Job ausführen - er wird gemäß dem WCS-Offset positioniert

Derselbe Job kann an verschiedenen Positionen ausgeführt werden, einfach durch Ändern des aktiven WCS.

## Praktische Workflows

### Workflow 1: Mehrere Vorrichtungs-Positionen

Sie haben ein großes Bett und möchten drei Arbeitsbereiche einrichten:

1. **Maschine referenzieren**, um eine Referenz zu etablieren
2. **Zum ersten Arbeitsbereich verfahren** und G54-Offset setzen (X nullen, Y nullen)
3. **Zum zweiten Arbeitsbereich verfahren** und G55-Offset setzen
4. **Zum dritten Arbeitsbereich verfahren** und G56-Offset setzen
5. Jetzt können Sie zwischen G54, G55 und G56 wechseln, um Jobs in jedem Bereich auszuführen

### Workflow 2: An Material ausrichten

Sie haben ein Stück Material irgendwo auf Ihrem Bett platziert:

1. **Den Laserkopf verfahren** an die Ecke Ihres Materials
2. **G54 auswählen** (oder Ihr bevorzugtes WCS)
3. **X nullen und Y nullen klicken**, um die Materialecke als (0, 0) zu setzen
4. **Ihren Job entwerfen** mit (0, 0) als Ursprung
5. **Den Job ausführen** - er wird von der Materialecke starten

### Workflow 3: Produktions-Raster

Sie müssen dasselbe Teil 10-mal an verschiedenen Orten schneiden:

1. **Ein Teil entwerfen** in Rayforge
2. **G54-G59-Offsets einrichten** für Ihre gewünschten Positionen
3. **Den Job ausführen** mit aktivem G54
4. **Zu G55 wechseln** und erneut ausführen
5. **Wiederholen** für jede WCS-Position

## Wichtige Hinweise

### WCS-Einschränkungen

- **G53 kann nicht geändert werden**: Maschinenkoordinaten sind von der Hardware festgelegt
- **Offsets bleiben erhalten**: WCS-Offsets werden im Controller Ihrer Maschine gespeichert
- **Verbindung erforderlich**: Sie müssen mit einer Maschine verbunden sein, um WCS-Offsets zu setzen

### WCS und Job-Ursprung

WCS funktioniert unabhängig von Ihren Job-Ursprungs-Einstellungen. Der Job-Ursprung bestimmt, wo auf der Arbeitsfläche Ihr Job beginnt, während WCS bestimmt, wo diese Arbeitsflächen-Position auf Ihrer Maschine abgebildet wird.

### Maschinen-Kompatibilität

Nicht alle Maschinen unterstützen alle WCS-Funktionen:

- **GRBL (v1.1+)**: Volle Unterstützung für G53-G59
- **Smoothieware**: Unterstützt G54-G59 (Offset-Lesen kann eingeschränkt sein)
- **Benutzerdefinierte Controller**: Variiert je nach Implementierung

---

**Verwandte Seiten:**

- [Koordinatensysteme](coordinate-systems) - Koordinatensysteme verstehen
- [Steuerungsfeld](../ui/control-panel) - Manuelle Steuerung und WCS-Verwaltung
- [Maschinen-Setup](../machine/general) - Ihre Maschine konfigurieren
- [3D-Vorschau](../ui/3d-preview) - Ihre Jobs visualisieren
