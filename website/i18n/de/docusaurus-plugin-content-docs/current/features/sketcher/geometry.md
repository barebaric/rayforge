---
description: "Erfahre, wie du Linien, Bezier-Kurven, Bögen, Ellipsen, Rechtecke und andere 2D-Geometrie im Rayforge-Sketcher erstellst."
---

# 2D-Geometrie erstellen

Der Sketcher unterstützt das Erstellen der folgenden grundlegenden geometrischen Elemente:

- **Pfade (Linien und Bezier-Kurven)**: Gerade Linien und glatte Bezier-Kurven mit dem vereinheitlichten Pfad-Werkzeug zeichnen. Klicke um Punkte zu setzen, ziehe um Bezier-Griffpunkte zu erstellen.
- **Bögen**: Bögen durch Angeben eines Mittelpunkts, Startpunkts und Endpunkts zeichnen
- **Ellipsen**: Erstelle Ellipsen (und Kreise) mit zwei Klicks: Der erste
  setzt den Mittelpunkt, der zweite den Randpunkt. Du kannst auch am
  Mittelpunkt drücken, ziehen und am Rand loslassen - beide Gesten
  funktionieren gleichwertig. Halte `Strg` gedrückt, um auf einen perfekten
  Kreis zu beschränken, und `Shift`, um den Startpunkt als Mittelpunkt der
  Ellipse zu verwenden.
- **Rechtecke**: Rechtecke durch Angeben von zwei gegenüberliegenden Ecken
  zeichnen, oder an der ersten Ecke drücken, ziehen und an der
  gegenüberliegenden Ecke loslassen. Jedes Rechteck erstellt automatisch
  einen Mittelpunkt (auf den geometrischen Mittelpunkt beschränkt), sodass
  du eine Dimension daran anbringen oder daran einrasten kannst. Halte
  `Shift` beim Zeichnen gedrückt, um das Rechteck symmetrisch um den
  Startpunkt zu platzieren, und `Strg`, um es auf ein Quadrat zu beschränken.
- **Abgerundete Rechtecke**: Rechtecke mit abgerundeten Ecken zeichnen - mit
  denselben Gesten und Zusatztasten wie beim Rechteck-Werkzeug: zwei Klicks
  oder Klicken-und-Ziehen, mit `Shift` zur Zentrierung auf dem Startpunkt
  und `Strg` zur Beschränkung auf ein Quadrat. Der Eckradius lässt sich
  durch Eingabe von Maßen festlegen (`0-9`, Felder W, H und R).
- **Textfelder**: Textelemente zu deiner Skizze hinzufügen. Der Textinhalt
  unterstützt parametrische Vorlagenausdrücke (siehe
  [Textvorlagen](../text.md)).
- **Füllungen**: Geschlossene Bereiche füllen, um feste Bereiche zu erstellen

Diese Elemente bilden die Grundlage deiner 2D-Designs und können kombiniert werden, um komplexe Formen zu erstellen. Füllungen sind besonders nützlich, um feste Bereiche zu erstellen, die als ein Stück graviert oder geschnitten werden.

## Zwei Klicks oder Ziehen

Die Form-Erstellungswerkzeuge (Ellipse, Rechteck, abgerundetes Rechteck)
akzeptieren zwei Gesten gleichwertig: Klicke den ersten Punkt, bewege und
klicke den zweiten Punkt, oder drücke am ersten Punkt, ziehe und lasse am
zweiten los. Ein kurzer Klick ohne Bewegung aktiviert das Werkzeug lediglich
und wartet auf den zweiten Punkt, sodass versehentliche Klicke keine
degenerierte Geometrie hinterlassen. Solange eine Vorschau aktiv ist, zeigt
die Statusleiste die verfügbaren Zusatztasten an, und `Esc` bricht die
Vorschau ab.

## Arbeiten mit Bezier-Kurven

Das Pfad-Werkzeug unterstützt Bezier-Kurven zum Erstellen glatter, organischer Formen:

### Bezier-Kurven zeichnen

1. Wähle das Pfad-Werkzeug aus dem Kreismenü oder verwende den Tastatur-Kurzbefehl
2. Klicke um Punkte zu setzen - jeder Klick erstellt einen neuen Punkt
3. Ziehe nach dem Klicken um Bezier-Griffpunkte für glatte Kurven zu erstellen
4. Füge weitere Punkte hinzu um deinen Pfad zu erweitern
5. Drücke Escape oder doppelklicke um den Pfad abzuschließen

### Bezier-Kurven bearbeiten

- **Punkte verschieben**: Klicke und ziehe einen beliebigen Punkt um ihn neu zu positionieren
- **Griffpunkte anpassen**: Ziehe die Griff-Endpunkte um die Kurvenform zu ändern
- **Mit existierenden Punkten verbinden**: Beim Bearbeiten eines Pfades kannst du an existierende Punkte in deiner Skizze einrasten
- **Glatt/Symmetrisch machen**: Punkte, die durch eine Koinzident-Einschränkung verbunden sind, können glatt (kontinuierliche Tangente) oder symmetrisch (gespiegelte Griffpunkte) gemacht werden

### Kurven zu Linien konvertieren

Verwende das **Glätten-Werkzeug** um Bezier-Kurven zurück in gerade Linien zu konvertieren.
Dies ist nützlich wenn du saubere, einfache Geometrie benötigst. Wähle die Bezier-Segmente
die du konvertieren möchtest und wende die Glätten-Aktion an.
