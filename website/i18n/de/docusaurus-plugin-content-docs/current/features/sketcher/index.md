---
description:
  "Der integrierte parametrische 2D-Sketcher von Rayforge ermöglicht es dir, einschränkungsbasierte,
  maßorientierte Designs zu zeichnen, die bearbeitbar und präzise bleiben."
---

# Parametrischer 2D-Sketcher

Rayforge enthält einen parametrischen 2D-Sketcher, mit dem du Teile direkt in der Anwendung zeichnen
kannst. Statt fertige Grafiken aus einem anderen Programm zu importieren, skizzierst du Linien,
Kurven und Formen auf einer unendlichen Leinwand und verbindest sie mit Einschränkungen. Das
Ergebnis ist ein Design, das präzise bleibt, egal wie oft du deine Meinung über seine Abmessungen
änderst.

![Der Sketch-Editor](/screenshots/addons-sketcher-editor.webp)

## Was „parametrisch" hier bedeutet

Eine Skizze ist mehr als eine Zeichnung — sie ist ein kleines Modell mit Regeln. Die Regeln sind
**Einschränkungen**: Aussagen wie „diese beiden Linien sind parallel", „diese Ecke ist ein rechter
Winkel" oder „diese Kante ist genau 100 mm lang". Nach jeder Änderung ordnet ein Solver die
Geometrie neu an, sodass alle Regeln wieder gelten.

Das hat eine praktische Konsequenz: Du kannst deine Design-Absicht einmal festhalten und dann weiter
bearbeiten. Erhöhe die Abstands-Einschränkung von 100 mm auf 130 mm, und das ganze Teil folgt.
Dimensionale Einschränkungen akzeptieren auch Ausdrücke — ein Radius von `width/2` bleibt die Hälfte
der Breite, egal wie breit das Teil wird.

Wenn jeder verbleibende Freiheitsgrad durch eine Einschränkung festgelegt ist, ist die Skizze
_vollständig eingeschränkt_. Der Editor zeigt dir über Farben, wo du stehst: Geometrie, die von
Einschränkungen gehalten wird, wird grün gezeichnet, uneingeschränkte Punkte schwarz, und sobald
eine Skizze vollständig eingeschränkt ist, wird das Grün dunkler. Einschränkungen, die sich
widersprechen, werden rot markiert und im Konflikte-Panel in der Seitenleiste aufgelistet, wo du sie
untersuchen oder löschen kannst.

![Eine Skizze mit Bemaßungen](/screenshots/addons-sketcher-constraints.webp)

Eine unvollständig eingeschränkte Skizze ist kein Fehler — sie ist oft genau das, was du beim
Experimentieren willst. Die Seite [Einschränkungen](constraints.md) erklärt jeden verfügbaren
Einschränkungstyp im Detail.

## Der Sketch-Editor

Skizzen liegen wie jedes andere Werkstück im Dokument. Erstelle eine mit der Schaltfläche **Neue
Skizze** im unteren Panel (oder klicke mit der rechten Maustaste auf die Leinwand und wähle
denselben Eintrag aus dem Kontextmenü), und der Sketch-Editor übernimmt das Fenster: die Leinwand in
der Mitte, ein Eigenschaften-Panel mit dem Skizzen-Namen und seinen Parametern links und eine
Symbolleiste oben.

Die Symbolleiste bündelt die Sitzungs-Werkzeuge — Rückgängig und Wiederholen, Umschalter für die
Sichtbarkeit von Einschränkungen und Konstruktionsgeometrie, Füll- und Linienfarben, Spiegeln —
sowie die Schaltflächen **Fertigstellen** und **Abbrechen**. **Fertigstellen** speichert die Skizze
zurück in das Dokument; **Abbrechen** verwirft die in dieser Sitzung gemachten Änderungen. Um eine
bestehende Skizze später erneut zu bearbeiten, doppelklicke sie im Hauptarbeitsbereich, oder wähle
sie aus und wähle **Skizze bearbeiten** aus dem Kontextmenü.

Der Editor ist tastaturorientiert. Die Statusleiste unten listet immer die Kurzbefehle auf, die für
das aktuelle Werkzeug und die aktuelle Auswahl gelten, sodass die relevanten Tasten genau dann auf
dem Bildschirm stehen, wenn du sie brauchst. Vollständiges Rückgängig und Wiederholen ist für jede
Operation verfügbar.

## Das Kreismenü

Ein Rechtsklick irgendwo im Sketch-Editor öffnet das Kreismenü — ein radiales Menü, das jedes
Zeichen- und Bearbeitungswerkzeug einen Klick entfernt bereitstellt. Das Menü ist kontextsensitiv:
Ein Rechtsklick auf freier Fläche bietet die Zeichenwerkzeuge an, während ein Rechtsklick auf eine
ausgewählte Linie die Einschränkungen und Modifikationen anbietet, die für eine Linie sinnvoll sind.
Verwandte Werkzeuge sind zu Gruppen zusammengefasst; fahre mit der Maus über eine Gruppe, um ihre
Einträge auszufächern. Klicke erneut mit der rechten Maustaste, um das Menü zu schließen oder es an
anderer Stelle wieder zu öffnen.

![Das Kreismenü, geöffnet auf einer ausgewählten Linie](/screenshots/addons-sketcher-pie-menu.webp)

## Tastatur-Kurzbefehle

Der Sketcher wird über die Tastatur bedient, und die Statusleiste unten listet immer die Kurzbefehle
auf, die für das aktuelle Werkzeug und die aktuelle Auswahl gelten. Diese allgemeinen Kurzbefehle
funktionieren überall im Editor:

| Aktion                                    | Kurzbefehl                           |
| ----------------------------------------- | ------------------------------------ |
| Auswahl-Werkzeug                          | `Space`                              |
| Rückgängig / Wiederholen                  | `Ctrl+Z` / `Ctrl+Y` (`Ctrl+Shift+Z`) |
| Auswahl duplizieren                       | `Ctrl+D`                             |
| Auswahl löschen                           | `Delete`                             |
| Auswahl verschieben                       | `Pfeiltasten` (`Shift`: größer)      |
| Auswahl vertikal / horizontal spiegeln    | `M+V` / `M+H`                        |
| Konstruktionsmodus umschalten             | `G+N`                                |
| Operation abbrechen oder Auswahl aufheben | `Escape`                             |
| Ansicht an Inhalt anpassen                | `1`                                  |

Das Spiegeln erfolgt vor Ort über die Mitte des Begrenzungsrahmens der Auswahl; Einschränkungen, die
die Auswahlgrenze überschreiten, werden entfernt, interne Einschränkungen bleiben erhalten.
Duplikate bekommen neue IDs und neu zugewiesene interne Einschränkungen; Rückgängig entfernt sie.

Jedes Zeichen- und Bearbeitungswerkzeug hat zusätzlich einen Kurzbefehl aus zwei Tasten, der auf
seiner Seite dokumentiert ist:

| Werkzeug                              | Kurzbefehl |
| ------------------------------------- | ---------- |
| [Pfad](path.md)                       | `G+P`      |
| [Bogen](arc-ellipse.md)               | `G+A`      |
| [Ellipse](arc-ellipse.md)             | `G+C`      |
| [Rechteck](rectangle.md)              | `G+R`      |
| [Abgerundetes Rechteck](rectangle.md) | `G+O`      |
| [Bereiche füllen](fill.md)            | `G+F`      |
| [Textfeld](text-box.md)               | `G+T`      |
| [Kreisförmiges Array](arrays.md)      | `G+Y`      |
| [Array entlang Kurve](arrays.md)      | `G+W`      |
| [Raster](grid.md)                     | `G+G`      |
| [Versatz](offset.md)                  | `O+F`      |
| [Fase](chamfer-fillet.md)             | `C+H`      |
| [Verrundung](chamfer-fillet.md)       | `C+F`      |

Die Einschränkungs-Kurzbefehle sind auf der Seite [Einschränkungen](constraints.md) aufgelistet.

## Raster und Einrasten {#grid-and-snapping}

Die Leinwand zeigt ein adaptives Raster, dessen Abstand sich an die Zoomstufe anpasst und das
entlang der Achsen in deiner bevorzugten Einheit beschriftet ist, sodass es auch als Lineal dient:
Du kannst Größen und Positionen direkt von der Leinwand ablesen.

Während du zeichnest oder ziehst, zieht _magnetisches Einrasten_ den Cursor zu nahegelegenen
Referenzpunkten. Die Leinwand markiert, wozu der Cursor angezogen wird:

- ein **blauer Kreis** markiert einen vorhandenen Punkt (Endpunkt),
- **grüne Pfeile** markieren einen Mittelpunkt,
- eine **rosa Hervorhebung** bedeutet, dass sich der Cursor über einer Kante befindet,
- **gestrichelte Linien** über die Leinwand sind Ausrichtungshilfslinien, die angezeigt werden, wenn
  der Cursor horizontal oder vertikal mit einem anderen Punkt fluchtet,
- weitere Indikatoren decken Sonderfälle ab, wie gleichmäßige Abstände (orange), Tangentialität
  (lila) und Mittelpunkte (rot).

Einrasten ist nicht nur eine visuelle Hilfe — wenn du Geometrie auf einem Einrast-Ziel festlegst,
wird die passende Einschränkung automatisch erstellt. Beendest du eine Linie auf einem vorhandenen
Endpunkt, werden die beiden koinzident; Einrasten auf einen Mittelpunkt erzeugt eine
Symmetrie-Einschränkung; Ausrichtungshilfslinien werden zu horizontalen oder vertikalen
Einschränkungen. Wenn du eine freie Platzierung bevorzugst, schaltet `Tab` das magnetische Einrasten
aus. Halte beim Ziehen `Shift` gedrückt, um die Bewegung auf die nächstgelegene Achse zu
beschränken.

![Ausrichtungshilfslinien und der Gleichabstand-Einrastindikator beim Zeichnen](/screenshots/addons-sketcher-snap.webp)

## Konstruktionsgeometrie {#construction-geometry}

Jede Entität kann als Konstruktionsgeometrie markiert werden. Konstruktions-Entitäten werden
gestrichelt gezeichnet, wirken für den Solver wie jede andere Geometrie als Layout-Hilfslinien und
werden beim Fertigen der Skizze von den Werkzeugpfaden ausgeschlossen. Sie sind praktisch für
Mittellinien, Konstruktionskreise und das Gerüst hinter symmetrischen Designs. Wähle eine oder
mehrere Entitäten aus und drücke `G+N` (oder verwende den Konstruktion-Eintrag im Kreismenü), um die
Markierung umzuschalten; der Konstruktions-Umschalter in der Symbolleiste blendet sie aus, wenn sie
im Weg sind.

## Wo es weitergeht

Die Zeichenwerkzeuge sind jeweils auf ihrer eigenen Seite dokumentiert: [Pfad](path.md) (Linien und
Bezier-Kurven), [Bogen und Ellipse](arc-ellipse.md), [Rechteck](rectangle.md) (und abgerundete
Rechtecke), [Bereiche füllen](fill.md), [Textfeld](text-box.md) und [Raster](grid.md).
Modifikationen wie [Versatz](offset.md) und [Fase und Verrundung](chamfer-fillet.md) formen
bestehende Geometrie um, [Arrays](arrays.md) kopiert sie entlang eines Kreises oder einer Kurve, und
[Ausdrücke](expressions.md) erklärt Parameter, Ausdrücke und parametrische Textfelder. Skizzen
können mit allen Einschränkungen intakt gespeichert und erneut importiert werden — siehe
[Import und Export](import-export.md).
