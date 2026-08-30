---
description:
  "Zeichne Rechtecke und abgerundete Rechtecke im Rayforge-Sketcher, mit Mittelpunkten, Zusatztasten
  und Maß-Eingabe."
---

# Rechteck und abgerundetes Rechteck

Der Sketcher bietet zwei Rechteck-Werkzeuge, die dieselben Gesten und Zusatztasten teilen: das
**Rechteck**-Werkzeug (`G+R`) und das **Abgerundetes Rechteck**-Werkzeug (`G+O`).

![Ein Rechteck und ein abgerundetes Rechteck](/screenshots/addons-sketcher-tool-rectangle.webp)

## Rechtecke zeichnen

Zeichne ein Rechteck, indem du zwei gegenüberliegende Ecken angibst, oder drücke an der ersten Ecke,
ziehe und lasse an der gegenüberliegenden Ecke los. Die Zusatztasten funktionieren bei beiden
Werkzeugen gleich:

- Halte `Shift` gedrückt, um das Rechteck symmetrisch um den Startpunkt zu platzieren.
- Halte `Ctrl` gedrückt, um es auf ein Quadrat zu beschränken.

Jedes Rechteck erstellt automatisch einen **Mittelpunkt**, der auf den geometrischen Mittelpunkt
beschränkt ist, sodass du die Mitte der Form bemaßen oder einrasten kannst.

Während eine Vorschau aktiv ist, kannst du die exakte Größe eingeben: Die Statusleiste zeigt die
Felder `W` und `H` (plus `R` für den Eckradius abgerundeter Rechtecke). Gib einen Wert ein, drücke
`Tab`, um zwischen den Feldern zu wechseln, und `Enter`, um anzuwenden. Beide Werkzeuge akzeptieren
die Zwei-Klick- und die Klicken-und-Ziehen-Geste gleichwertig; `Esc` bricht die Vorschau ab.

Der Eckradius des abgerundeten Rechtecks lässt sich auch später durch Bearbeiten seiner
Einschränkungen ändern — die Ecken sind vollständig eingeschränkt, daher bleibt der Radius
anpassbar.
