---
description:
  "Zeichne gerade Linien und glatte Bezier-Kurven mit dem Pfad-Werkzeug im Rayforge-Sketcher."
---

# Pfad-Werkzeug

Das Pfad-Werkzeug (`G+P` oder `G+L`) zeichnet in einem einheitlichen Workflow verbundene Ketten aus
geraden Linien und glatten Bezier-Kurven. Es ist das vielseitigste Zeichenwerkzeug im Sketcher:
Klicke, um Punkte zu setzen, und ziehe, um das Segment zu einer Kurve zu biegen.

![Ein Pfad aus zwei Linien, verbunden durch ein Bezier-Segment, mit seinen Wegpunkten und Griffpunkten](/screenshots/addons-sketcher-tool-path.webp)

## Pfade zeichnen

1. Wähle das Pfad-Werkzeug aus dem Kreismenü, dem Menü **Sketch** oder mit `G+P`.
2. Klicke, um den ersten Punkt zu setzen. Eine Live-Vorschau folgt dem Cursor.
3. Klicke erneut ohne zu ziehen, um ein gerades Segment abzuschließen — das nächste Segment beginnt
   sofort an diesem Punkt.
4. Drücke an einem Punkt und ziehe vor dem Loslassen, um das Segment in eine Bezier-Kurve zu
   verwandeln. Der Zug steuert die „Wölbung" der Kurve.
5. Füge weiter Punkte hinzu, um deinen Pfad aufzubauen.
6. Drücke `Escape` oder doppelklicke, um den Pfad abzuschließen.

Während eine Vorschau aktiv ist, listet die Statusleiste die geltenden Zusatztasten auf, und `Esc`
bricht sie ab.

## Arbeiten mit Bezier-Kurven

Bezier-Kurven erzeugen glatte, organische Formen:

- **Griffpunkte anpassen**: Wähle eine Bezier aus und ziehe die runden Griff-Endpunkte, um die
  Kurvenform zu ändern. Jeder Griffpunkt biegt die Kurve auf seiner Seite des Wegpunkts.
- **Mit vorhandenen Punkten verbinden**: Beim Zeichnen heftet das magnetische Einrasten neue
  Segmente an vorhandene Punkte in deiner Skizze, und die passende Einschränkung wird automatisch
  erstellt.

### Wegpunkt-Typen

Der Punkt, an dem zwei Segmente eines Pfades aufeinandertreffen, ist ein _Wegpunkt_. Der
Wegpunkt-Typ steuert, wie die Kurve durch ihn hindurchläuft:

- **Spitz**: Die Griffpunkte auf beiden Seiten sind unabhängig und erzeugen eine Ecke.
- **Glatt**: Die Griffpunkte teilen sich eine Tangente und erzeugen einen kontinuierlichen,
  abgerundeten Übergang.
- **Symmetrisch**: Wie Glatt, nur dass die Griffpunkte zusätzlich gespiegelt sind, sodass beide
  Seiten gleich stark biegen.

Um den Typ eines Wegpunkts zu ändern, klicke ihn (oder das angrenzende Bezier-Segment) mit der
rechten Maustaste an und wähle den Typ aus dem Kreismenü. Neu gezeichnete Bezier-Wegpunkte sind
symmetrisch.

![Das Kreismenü auf einem ausgewählten Bezier-Wegpunkt mit den Werkzeugen Glätten, Spitz, Glatt und Symmetrisch](/screenshots/addons-sketcher-tool-path-pie-menu.webp)

### Kurven in Linien umwandeln

Das **Glätten**-Werkzeug aus demselben Kreismenü wandelt Bezier-Kurven zurück in gerade Linien, was
nützlich ist, wenn du saubere, einfache Geometrie brauchst. Wähle die Bezier-Segmente aus, die du
umwandeln möchtest, und wende die Glätten-Aktion an. Die Segmente ziehen sich auf die gerade
Verbindung zwischen ihren Endpunkten zusammen.

## Automatische Einschränkungen

Das Pfad-Werkzeug nimmt wie jedes andere Zeichenwerkzeug am magnetischen Einrasten teil. Zeigen die
Einrast-Hilfslinien während des Zeichnens eine Ausrichtung an, werden passende horizontale und
vertikale Einschränkungen automatisch erstellt, was deine Skizze von Anfang an ordentlich hält,
anstatt sie nachträglich in Ordnung zu bringen. Halte `Shift` gedrückt, um das neue Segment auf die
nächstgelegene Achse zu beschränken. Die vollständige Liste der Einrast-Indikatoren findest du unter
[Raster und Einrasten](index.md#grid-and-snapping).
