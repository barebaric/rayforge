---
description: "Erfahre, wie du Linien, Bezier-Kurven, Bögen, Ellipsen, Rechtecke und andere 2D-Geometrie im Rayforge-Sketcher erstellst."
---

# 2D-Geometrie erstellen

Der Sketcher unterstützt das Erstellen der folgenden grundlegenden geometrischen Elemente:

- **Pfade (Linien und Bezier-Kurven)**: Gerade Linien und glatte Bezier-Kurven mit dem vereinheitlichten Pfad-Werkzeug zeichnen. Klicke um Punkte zu setzen, ziehe um Bezier-Griffpunkte zu erstellen.
- **Bögen**: Bögen durch Angeben eines Mittelpunkts, Startpunkts und Endpunkts zeichnen
- **Ellipsen**: Erstelle Ellipsen (und Kreise) durch Definieren eines Mittelpunkts
  und Ziehen, um Größe und Seitenverhältnis festzulegen. Halte `Strg` während
  des Ziehens gedrückt, um auf einen perfekten Kreis zu beschränken.
- **Rechtecke**: Rechtecke durch Angeben von zwei gegenüberliegenden Ecken zeichnen.
  Jedes Rechteck erstellt automatisch einen Mittelpunkt (auf den geometrischen
  Mittelpunkt beschränkt), sodass du eine Dimension daran anbringen oder daran
  einrasten kannst. Halte `Shift` beim Zeichnen gedrückt, um das Rechteck
  symmetrisch um den Startpunkt zu platzieren, ähnlich wie beim Ellipsen-Werkzeug.
- **Abgerundete Rechtecke**: Rechtecke mit abgerundeten Ecken zeichnen
- **Textfelder**: Textelemente zu deiner Skizze hinzufügen. Der Textinhalt
  unterstützt parametrische Vorlagenausdrücke (siehe
  [Textvorlagen](../text.md)).
- **Füllungen**: Geschlossene Bereiche füllen, um feste Bereiche zu erstellen

Diese Elemente bilden die Grundlage deiner 2D-Designs und können kombiniert werden, um komplexe Formen zu erstellen. Füllungen sind besonders nützlich, um feste Bereiche zu erstellen, die als ein Stück graviert oder geschnitten werden.

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
