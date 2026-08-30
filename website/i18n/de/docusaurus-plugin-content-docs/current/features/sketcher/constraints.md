---
description:
  "Wie Einschränkungen im Rayforge-Sketcher funktionieren: Hinzufügen, Bearbeiten, Auswählen und
  Löschen sowie Auflösen von Konflikten."
---

# Einschränkungen

Einschränkungen sind die Regeln, die eine Skizze zusammenhalten. Jede ist eine kleine Aussage über
die Geometrie — „diese beiden Punkte sind ein und dasselbe", „diese Linie ist genau 80 mm lang" —
und nach jeder Bearbeitung ordnet der Solver die Skizze neu an, sodass alle Aussagen gleichzeitig
gelten. Geometrie ohne Einschränkungen kann frei driften; jede Einschränkung, die du hinzufügst,
legt einen Freiheitsgrad fest.

Es gibt zwei Familien. **Geometrische Einschränkungen** erfassen Beziehungen ohne Messwert:
Koinzidenz, Horizontalität, Tangentialität, Symmetrie. **Dimensionale Einschränkungen** hängen eine
Zahl an die Geometrie: einen Abstand, einen Radius, einen Winkel. Dimensionale Werte akzeptieren
Ausdrücke (siehe [unten](#dimensionale-werte-bearbeiten)), und genau dort passiert das
„Parametrische" im parametrischen Skizzieren.

Der Solver meldet seinen Zustand über Farben. Von Einschränkungen gehaltene Geometrie wird grün
gezeichnet, uneingeschränkte Punkte schwarz, und eine vollständig eingeschränkte Skizze färbt das
Grün dunkler. Gültige Einschränkungs-Markierungen sind grün, ausdrucksbasierte Markierungen orange,
und Markierungen von Einschränkungen, die der Solver nicht erfüllen kann, werden rot (siehe
[Konflikte](#wenn-einschränkungen-konfligieren)).

## Eine Einschränkung hinzufügen

Wähle die Geometrie aus, auf die die Einschränkung angewendet werden soll, und drücke entweder den
Tastatur-Kurzbefehl oder wähle die Einschränkung aus dem Kreismenü — geometrische Einschränkungen
liegen in der Gruppe **Einschränken**, dimensionale in der Gruppe **Bemessen**. Jede Einschränkung
verlangt eine bestimmte Auswahl:

| Einschränkung               | Auswählen                        | Kurzbefehl   |
| --------------------------- | -------------------------------- | ------------ |
| Horizontal / Vertikal       | 2 Punkte oder beliebige Linien   | `H` / `V`    |
| Koinzident / Punkt auf Form | 2 Punkte oder Punkt + Form       | `O` oder `C` |
| Senkrecht                   | 2 Formen                         | `N`          |
| Tangential                  | 1 Linie + 1 Bogen oder Kreis     | `T`          |
| Symmetrie                   | 3 Punkte oder 2 Punkte + 1 Linie | `S`          |
| Gleiche Länge               | 2 oder mehr Formen               | `E`          |
| Abstand                     | 2 Punkte oder 1 Linie            | `K+D`        |
| Durchmesser                 | 1 Kreis                          | `K+O`        |
| Radius                      | 1 Bogen oder Kreis               | `K+R`        |
| Winkel                      | 2 Linien                         | `K+A`        |
| Seitenverhältnis            | 2 Linien                         | `K+X`        |

Die Reihenfolge einer Auswahl spielt nie eine Rolle, mit einer Ausnahme: Bei drei ausgewählten
Punkten verwendet Symmetrie den **letzten** Punkt als Spiegelzentrum. Ein Kurzbefehl wird nur
ausgelöst, wenn die aktuelle Auswahl zur Einschränkung passt — alles andere wird auch aus dem
Kreismenü herausgefiltert.

Einschränkungen entstehen auch von selbst, während du zeichnest: Einrasten auf einen Endpunkt
erzeugt eine Koinzident-Einschränkung, und Ausrichtungshilfslinien werden zu horizontalen oder
vertikalen Einschränkungen (siehe [die Sketcher-Übersicht](index.md#grid-and-snapping)).

## Geometrische Einschränkungen

Eine **Koinzident**-Einschränkung führt zwei getrennte Punkte an einer Position zusammen. Wähle die
beiden Punkte aus, und beide werden zusammengezogen; die Markierung ist ein Ring um den vereinigten
Punkt. Zeichnest du eine Linie, die genau auf einem vorhandenen Endpunkt endet, wird diese
Einschränkung automatisch erstellt.

![Zwei Linien, durch eine Koinzident-Einschränkung verbunden](/screenshots/addons-sketcher-constraint-coincident.webp)

**Horizontal** und **Vertikal** drehen die ausgewählte Linie oder das Paar ausgewählter Punkte auf
eine Achse. Die Markierungen sind kleine Balken — horizontal bzw. vertikal — die neben der Geometrie
gezeichnet werden.

![Eine horizontale Einschränkung](/screenshots/addons-sketcher-constraint-horizontal.webp)

![Eine vertikale Einschränkung](/screenshots/addons-sketcher-constraint-vertical.webp)

**Senkrecht** zwingt zwei Formen, sich im rechten Winkel zu treffen. Es funktioniert für zwei
Linien, eine Linie und einen Bogen oder Kreis sowie für zwei Bögen und Kreise. Die Markierung ist
ein rechter-Winkel-Bogen am Schnittpunkt.

![Zwei Linien, die sich im rechten Winkel treffen](/screenshots/addons-sketcher-constraint-perpendicular.webp)

**Tangential** glättet den Übergang, wo eine Linie auf einen Bogen oder Kreis trifft: Die Linie wird
gedreht, sodass sie die Kurve berührt, ohne sie zu schneiden. Ihre Markierung ist ein kleines „T" am
Berührungspunkt.

![Eine Linie, tangential zu einem Kreis](/screenshots/addons-sketcher-constraint-tangent.webp)

**Punkt auf Form** heftet einen Punkt an eine Linie, einen Bogen oder einen Kreis — ohne ihn mit
einem bestimmten Punkt zu verschmelzen, wie es Koinzident tut. Wähle einen Punkt und eine Form; die
Markierung ist ein Ring um den eingeschränkten Punkt. Ist die Form eine Kurve (Bezier), wird der
Punkt darauf beschränkt, entlang ihr zu gleiten.

![Ein Linienendpunkt, der auf einer anderen Linie aufliegt](/screenshots/addons-sketcher-constraint-point-on-line.webp)

**Symmetrie** spiegelt zwei Punkte über ein Zentrum oder eine Achse und kommt in den beiden bereits
erwähnten Modi vor: Wähle drei Punkte, und der letzte wird zum Zentrum, um das die ersten beiden
gespiegelt werden, oder wähle zwei Punkte und eine Linie, um über diese Linie zu spiegeln. Die
Markierung ist ein Paar entgegengesetzter Pfeilspitzen in der Mitte zwischen den gespiegelten
Punkten.

![Zwei Punkte, über eine Linie gespiegelt](/screenshots/addons-sketcher-constraint-symmetry.webp)

Eine siebte geometrische Einschränkung, **Kollinear**, zwingt Punkte auf eine unendliche Linie. Sie
hat keine Markierung auf der Leinwand und kann nicht von Hand angewendet werden — die Fasen- und
Verrundungs-Werkzeuge erstellen sie, um die modifizierte Ecke ausgerichtet zu halten.

## Dimensionale Einschränkungen

Die **Abstand**-Einschränkung legt den Abstand zwischen zwei Punkten oder die Länge einer Linie
fest. Ihr Label zeigt den aktuellen Wert in der Mitte des gemessenen Bereichs; wenn die beiden
Punkte nicht bereits durch eine Linie verbunden sind, macht eine gestrichelte Hilfslinie klar, was
gemessen wird.

![Eine Abstands-Einschränkung von 80 mm](/screenshots/addons-sketcher-constraint-distance.webp)

Kreise und Bögen bekommen ihre eigenen Bemaßungen. **Durchmesser** beschriftet die volle Breite
eines Kreises mit einem `Ø`-Präfix, **Radius** beschriftet den Abstand vom Zentrum eines Bogens oder
Kreises mit einem `R`-Präfix, und beide platzieren das Label knapp außerhalb der Form mit einer
kurzen Hilfslinie.

![Eine Durchmesser-Einschränkung](/screenshots/addons-sketcher-constraint-diameter.webp)

![Eine Radius-Einschränkung](/screenshots/addons-sketcher-constraint-radius.webp)

Die **Winkel**-Einschränkung setzt den Winkel zwischen zwei ausgewählten Linien. Sie zeichnet einen
Bogen zwischen den beiden Richtungen an deren Schnittpunkt, beschriftet mit dem Wert in Grad.

![Eine Winkel-Einschränkung von 45 Grad](/screenshots/addons-sketcher-constraint-angle.webp)

**Seitenverhältnis** koppelt die Längen zweier Linien: Die Länge der ersten geteilt durch die Länge
der zweiten muss dem angegebenen Wert entsprechen. Ihre Markierung, ein Paar entgegengesetzter
Eckklammern, sitzt an der Stelle, an der sich die Linien treffen.

![Eine Seitenverhältnis-Einschränkung zwischen zwei Linien](/screenshots/addons-sketcher-constraint-aspect-ratio.webp)

Schließlich sorgt **Gleiche Länge**, angewendet auf zwei oder mehr Linien, Bögen, Kreise oder
Ellipsen, dafür, dass alle dieselbe Länge oder denselben Radius teilen, wobei jede Form mit einem
`=`-Zeichen markiert wird. Der Solver verwendet intern auch eine Gleicher-Abstand-Variante dieser
Einschränkung — zum Beispiel, um einen Kreis rund oder die beiden Seiten einer Fase symmetrisch zu
halten —, die dieselbe `=`-Markierung trägt, aber nicht von Hand angewendet werden kann.

![Zwei Linien gleicher Länge](/screenshots/addons-sketcher-constraint-equal-length.webp)

## Dimensionale Werte bearbeiten

Doppelklicke auf das Label einer dimensionalen Einschränkung, um sie zu bearbeiten. Der Dialog
akzeptiert eine einfache Zahl oder einen Ausdruck: Skizzen-Parameter und Eingabevariablen können
namentlich referenziert werden, und mathematische Funktionen sind verfügbar — ein Radius von
`width/2` folgt dem Breiten-Parameter, wohin er auch geht. Sobald eine Einschränkung von einem
Ausdruck angetrieben wird, färbt sich ihre Markierung orange als Erinnerung daran, dass die Zahl
berechnet und nicht eingegeben ist. Die vollständige Syntax zusammen mit den Skizzen-Parametern, auf
die sie verweisen kann, ist in [Ausdrücke](expressions.md) beschrieben.

Ein Doppelklick auf eine noch nicht bemaßte Linie, einen Bogen oder Kreis bietet an, die passende
Bemaßung direkt zu erstellen (Abstand, Radius oder Durchmesser).

## Auswählen und Löschen

Einschränkungs-Markierungen nehmen an der Auswahl teil wie alles andere: Beim Überfahren mit der
Maus erscheint eine gelbe Hervorhebung und ein Tooltip mit dem Namen der Einschränkung, und ein
Klick wählt sie aus, wobei sie blau gezeichnet wird. Drücken von `Entf` entfernt die ausgewählte
Einschränkung und gibt die Geometrie frei, die sie gehalten hat. Beim Löschen von Geometrie gehen
ihre Einschränkungen mit. Bei dimensionalen Einschränkungen hat der oben beschriebene
Bearbeitungsdialog keinen Löschen-Button — das Entfernen einer Bemaßung ist ein normales Löschen der
ausgewählten Markierung.

## Wenn Einschränkungen konfligieren

Einschränkungen, die sich widersprechen — etwa ein Dreieck, dessen Seiten nicht alle gleichzeitig
stimmen können —, können die Skizze nicht kaputtmachen: Der Solver gibt sein Bestes und markiert,
was er nicht erfüllen konnte. Konfligierende Einschränkungen werden rot, sowohl ihre Markierungen
als auch die Geometrie, die sie halten, sodass der beschädigte Bereich auf einen Blick sichtbar ist.

![Konfligierende Abstands-Einschränkungen, in der Seitenleiste markiert](/screenshots/addons-sketcher-conflicts.webp)

Die Seitenleiste listet jeden Konflikt unter **Konfligierende Einschränkungen** auf, wobei jede
Zeile die Einschränkung und die Punkte, die sie berührt, nennt. Die Zeilen sind interaktiv: Beim
Überfahren mit der Maus wird die Einschränkung auf der Leinwand hervorgehoben, ein Klick wählt sie
aus, und der Löschen-Button rechts entfernt sie. Typischerweise ist der schnellste Weg aus einem
Konflikt, die Einschränkung zu löschen oder neu zu bewerten, die die veraltete Absicht ausdrückt —
die Liste existiert genau deshalb, weil der Solver nicht erraten kann, welche der widersprüchlichen
Regeln die falsche ist.

## Wo es weitergeht

Jedes Zeichenwerkzeug ist auf seiner eigenen Seite dokumentiert — siehe [Pfad](path.md),
[Bogen und Ellipse](arc-ellipse.md) und [Rechteck](rectangle.md), um zu erfahren, wie man die Formen
zeichnet, an die diese Einschränkungen angeheftet werden.
