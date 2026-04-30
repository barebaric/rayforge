# Parametrischer 2D-Sketcher

Der Parametrische 2D-Sketcher ist eine leistungsstarke Funktion in Rayforge, mit der du präzise, einschränkungsbasierte 2D-Designs direkt in der Anwendung erstellen und bearbeiten kannst. Diese Funktion ermöglicht es dir, benutzerdefinierte Teile von Grund auf zu entwerfen, ohne externe CAD-Software zu benötigen.

## Übersicht

Der Sketcher bietet einen vollständigen Satz von Werkzeugen zum Erstellen geometrischer Formen und zum Anwenden parametrischer Einschränkungen, um präzise Beziehungen zwischen Elementen zu definieren. Dieser Ansatz stellt sicher, dass deine Designs ihre beabsichtigte Geometrie beibehalten, auch wenn Abmessungen geändert werden.

## Skizzen erstellen und bearbeiten

### Eine neue Skizze erstellen

1. Auf die "Neue Skizze"-Taste in der Symbolleiste klicken oder das Hauptmenü verwenden
2. Ein neuer leerer Sketch-Arbeitsbereich öffnet sich mit der Sketch-Editor-Oberfläche
3. Mit dem Erstellen von Geometrie beginnen, indem du die Zeichenwerkzeuge aus dem Kreismenü oder Tastatur-Kurzbefehlen verwendest
4. Einschränkungen anwenden, um Beziehungen zwischen Elementen zu definieren
5. Auf "Skizze fertigstellen" klicken, um deine Arbeit zu speichern und zum Hauptarbeitsbereich zurückzukehren

### Bestehende Skizzen bearbeiten

1. Auf ein skizzenbasiertes Werkstück im Hauptarbeitsbereich doppelklicken
2. Alternativ eine Skizze auswählen und "Skizze bearbeiten" aus dem Kontextmenü wählen
3. Modifikationen mit denselben Werkzeugen und Einschränkungen vornehmen
4. Auf "Skizze fertigstellen" klicken, um Änderungen zu speichern, oder "Skizze abbrechen", um sie zu verwerfen

## 2D-Geometrie erstellen

Der Sketcher unterstützt das Erstellen der folgenden grundlegenden geometrischen Elemente:

- **Pfade (Linien und Bezier-Kurven)**: Gerade Linien und glatte Bezier-Kurven mit dem vereinheitlichten Pfad-Werkzeug zeichnen. Klicke um Punkte zu setzen, ziehe um Bezier-Griffpunkte zu erstellen.
- **Bögen**: Bögen durch Angeben eines Mittelpunkts, Startpunkts und Endpunkts zeichnen
- **Ellipsen**: Erstelle Ellipsen (und Kreise) durch Definieren eines Mittelpunkts
  und Ziehen, um Größe und Seitenverhältnis festzulegen. Halte `Strg` während
  des Ziehens gedrückt, um auf einen perfekten Kreis zu beschränken.
- **Rechtecke**: Rechtecke durch Angeben von zwei gegenüberliegenden Ecken zeichnen
- **Abgerundete Rechtecke**: Rechtecke mit abgerundeten Ecken zeichnen
- **Textfelder**: Textelemente zu deiner Skizze hinzufügen. Der Textinhalt
  unterstützt parametrische Vorlagenausdrücke (siehe
  [Textvorlagen](#textvorlagen) unten).
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

## Parametrisches Einschränkungssystem

Das Einschränkungssystem ist der Kern des parametrischen Sketchers und ermöglicht es dir, präzise geometrische Beziehungen zu definieren:

### Geometrische Einschränkungen

- **Koinzident**: Zwingt zwei Punkte, dieselbe Position einzunehmen
- **Vertikal**: Schränkt eine Linie ein, perfekt vertikal zu sein
- **Horizontal**: Schränkt eine Linie ein, perfekt horizontal zu sein
- **Tangential**: Macht eine Linie tangential zu einem Kreis oder Bogen
- **Senkrecht**: Zwingt zwei Linien, eine Linie und einen Bogen/Kreis, oder zwei Bögen/Kreise, sich in einem 90-Grad-Winkel zu treffen
- **Punkt auf Linie/Form**: Schränkt einen Punkt ein, auf einer Linie, einem Bogen oder einem Kreis zu liegen
- **Symmetrie**: Erzeugt symmetrische Beziehungen zwischen Elementen. Unterstützt zwei Modi:
  - **Punkt-Symmetrie**: 3 Punkte auswählen (der erste ist das Zentrum)
  - **Linien-Symmetrie**: 2 Punkte und 1 Linie auswählen (die Linie ist die Achse)

### Dimensionale Einschränkungen

- **Abstand**: Setzt den exakten Abstand zwischen zwei Punkten oder entlang einer Linie
- **Durchmesser**: Definiert den Durchmesser eines Kreises
- **Radius**: Setzt den Radius eines Kreises oder Bogens
- **Winkel**: Erzwingt einen spezifischen Winkel zwischen zwei Linien
- **Seitenverhältnis**: Zwingt das Verhältnis zwischen zwei Abständen, gleich einem angegebenen Wert zu sein
- **Gleiche Länge/Gleicher Radius**: Zwingt mehrere Elemente (Linien, Bögen oder Kreise), dieselbe Länge oder denselben Radius zu haben
- **Gleicher Abstand**: Zwingt den Abstand zwischen zwei Punkt-Paaren, gleich zu sein

## Kreismenü-Oberfläche

Der Sketcher verfügt über ein kontextsensitives Kreismenü, das schnellen Zugriff auf alle Zeichen- und Einschränkungswerkzeuge bietet. Dieses Radialmenü erscheint, wenn du im Sketch-Arbeitsbereich rechtsklickst, und passt sich basierend auf deinem aktuellen Kontext und deiner Auswahl an.

Die Kreismenü-Elemente zeigen dynamisch verfügbare Optionen basierend darauf, was du ausgewählt hast. Wenn du beispielsweise auf leeren Raum klickst, siehst du Zeichenwerkzeuge. Wenn du auf ausgewählte Geometrie klickst, siehst du anwendbare Einschränkungen.

![Sketcher-Kreismenü](/screenshots/sketcher-pie-menu.png)

## Tastatur-Kurzbefehle

Der Sketcher bietet Tastatur-Kurzbefehle für effizienten Workflow:

### Werkzeug-Kurzbefehle
- `Leertaste`: Auswahl-Werkzeug
- `G+P`: Pfad-Werkzeug (Linien und Bezier-Kurven)
- `G+A`: Bogen-Werkzeug
- `G+C`: Kreis-Werkzeug
- `G+R`: Rechteck-Werkzeug
- `G+O`: Abgerundetes Rechteck-Werkzeug
- `G+F`: Bereich füllen-Werkzeug
- `G+T`: Textfeld-Werkzeug
- `G+G`: Raster-Werkzeug (Raster-Sichtbarkeit umschalten)
- `G+N`: Konstruktionsmodus auf Auswahl umschalten

### Aktions-Kurzbefehle
- `C+H`: Fase-Ecke hinzufügen
- `C+F`: Verrundungs-Ecke hinzufügen
- `C+S`: Ausgewählte Bezier-Kurven zu Linien glätten

### Einschränkungs-Kurzbefehle
- `H`: Horizontale Einschränkung anwenden
- `V`: Vertikale Einschränkung anwenden
- `N`: Senkrechte Einschränkung anwenden
- `T`: Tangentiale Einschränkung anwenden
- `E`: Gleiche-Einschränkung anwenden
- `O` oder `C`: Ausrichtungs-Einschränkung anwenden (Koinzident)
- `S`: Symmetrie-Einschränkung anwenden
- `K+D`: Abstands-Einschränkung anwenden
- `K+R`: Radius-Einschränkung anwenden
- `K+O`: Durchmesser-Einschränkung anwenden
- `K+A`: Winkel-Einschränkung anwenden
- `K+X`: Seitenverhältnis-Einschränkung anwenden

### Allgemeine Kurzbefehle
- `Strg+Z`: Rückgängig
- `Strg+Y` oder `Strg+Umschalt+Z`: Wiederholen
- `Entf`: Ausgewählte Elemente löschen
- `Escape`: Aktuelle Operation abbrechen oder Auswahl aufheben
- `F`: Ansicht an Inhalt anpassen

## Konstruktionsmodus

Der Konstruktionsmodus ermöglicht es dir, Entitäten als "Konstruktionsgeometrie" zu markieren - Hilfselemente, die verwendet werden, um dein Design zu leiten, aber nicht Teil der endgültigen Ausgabe sind. Konstruktions-Entitäten werden anders angezeigt (typischerweise als gestrichelte Linien) und werden nicht eingeschlossen, wenn die Skizze zum Laserschneiden oder Gravieren verwendet wird.

Um den Konstruktionsmodus umzuschalten:
- Eine oder mehrere Entitäten auswählen
- `N` oder `G+N` drücken, oder die Konstruktionsoption im Kreismenü verwenden

Konstruktions-Entitäten sind nützlich für:
- Erstellen von Referenzlinien und -kreisen
- Definieren temporärer Geometrie zur Ausrichtung
- Aufbauen komplexer Formen aus einem Rahmen von Hilfslinien

## Raster, Einrasten und Sichtbarkeits-Steuerung

### Raster-Werkzeug

Das Raster-Werkzeug bietet eine visuelle Referenz für Ausrichtung und
Größenbestimmung:

- Raster ein/ausschalten mit dem Raster-Werkzeug-Button oder `G+G`
- Das Raster passt sich an deinen Zoom-Level an für konsistente Abstände

### Magnetisches Einrasten

Beim Erstellen oder Verschieben von Geometrie zieht Rayforge deinen Cursor
automatisch zu nahegelegenen Elementen — Endpunkten, Linienmittelpunkten,
Schnittpunkten und anderen Referenzpunkten. Dies macht es einfach, Formen präzise
zu verbinden, ohne jeden Punkt manuell zu platzieren. Der Einrast-Indikator wird
hervorgehoben, wenn dein Cursor nahe an einem Einrast-Ziel ist.

### Automatische Einschränkung bei Erstellung

Viele Zeichenwerkzeuge wenden automatisch Einschränkungen an, während du Geometrie
erstellst. Wenn du beispielsweise eine Linie nahe der Horizontalen oder Vertikalen
zeichnest, bietet der Sketcher an, sie an Ort und Stelle zu fixieren. Dies hilft,
deine Skizze von Anfang an ordentlich zu halten, anstatt nachträglich Korrekturen
vorzunehmen.

### Anzeigen/Verbergen-Steuerung

Die Sketcher-Symbolleiste enthält Umschalt-Buttons zur Sichtbarkeitssteuerung:

- **Konstruktionsgeometrie anzeigen/verbergen**: Sichtbarkeit von
  Konstruktions-Entitäten umschalten
- **Einschränkungen anzeigen/verbergen**: Sichtbarkeit von
  Einschränkungs-Markierungen umschalten

Diese Steuerungen helfen, visuelle Unordnung bei der Arbeit an komplexen Skizzen
zu reduzieren.

### Achsenbeschränkte Bewegung

Beim Ziehen von Punkten oder Geometrie, halte `Umschalt` um die Bewegung auf die
nächstgelegene Achse (horizontal oder vertikal) zu beschränken. Dies ist nützlich,
um die Ausrichtung bei Anpassungen beizubehalten.

## Fase und Verrundung

Der Sketcher bietet Werkzeuge zum Modifizieren von Ecken deiner Geometrie:

- **Fase**: Ersetzt eine scharfe Ecke durch eine abgeschrägte Kante. Einen Verbindungspunkt auswählen (wo sich zwei Linien treffen) und die Fasen-Aktion anwenden.
- **Verrundung**: Ersetzt eine scharfe Ecke durch eine abgerundete Kante. Einen Verbindungspunkt auswählen (wo sich zwei Linien treffen) und die Verrundungs-Aktion anwenden.

Fase oder Verrundung verwenden:
1. Einen Verbindungspunkt auswählen, wo sich zwei Linien treffen
2. `C+H` für Fase oder `C+F` für Verrundung drücken
3. Das Kreismenü oder Tastatur-Kurzbefehle verwenden, um die Modifikation anzuwenden

## Textvorlagen

Textfelder unterstützen Vorlagenausdrücke in geschweiften Klammern. Diese
werden zum Lösungszeitpunkt mit den aktuellen Parameterwerten aufgelöst,
sodass sich der Text automatisch aktualisiert, wenn du eine Dimension oder
Eingabevariable änderst.

### Variablensubstitution

Referenziere beliebige Skizzenparameter oder Eingabevariablen namentlich:

- `{width}` — der aktuelle Wert des Parameters „width"
- `{name}` — der Wert eines String-Eingabeparameters
- `{count:.0f}` — formatiert mit einem Python-Formatbezeichner (keine Dezimalen)

### Mathematische Ausdrücke

Du kannst mathematische Funktionen in Vorlagen verwenden:

- `{sqrt(area):.2f}` — Quadratwurzel von „area", formatiert auf 2 Dezimalen
- `{width * 2}` — arithmetische Ausdrücke

Die Standard-Mathematikfunktionen (`sqrt`, `sin`, `cos`, `tan`, `pi` usw.)
sind verfügbar.

### Eingebaute Funktionen

- `{today()}` — das heutige Datum (z.B. `2026-05-01`)
- `{now()}` — aktuelles Datum und Uhrzeit
- `{uuid4()}` — eine eindeutige 8-stellige hexadezimale Zeichenkette, bei
  jeder Lösung neu generiert

Dies ist nützlich zum Datumstempeln von Teilen oder zum Erzeugen eindeutiger
Seriennummern für die Produktionskennzeichnung.

### Anwendungsbeispiele

- `Part #{uuid4()}` — eindeutige Seriennummer bei jeder Lösung
- `W={width:.1f} H={height:.1f}` — live Maßbeschriftungen
- `Datum: {today()}` — jedes Teil datumsstempeln
- `{name} - {count:.0f}Stk` — String- und numerische Parameter kombinieren

## Import und Export

### Objekte exportieren

Du kannst jedes ausgewählte Werkstück in verschiedene Vektorformate exportieren:

1. Ein Werkstück auf der Arbeitsfläche auswählen
2. **Objekt → Objekt exportieren...** wählen (oder rechtsklicken und aus Kontextmenü auswählen)
3. Das Exportformat wählen:
   - **RFS (.rfs)**: Rayforge's natives parametrisches Sketch-Format - behält alle Einschränkungen und kann zum Bearbeiten re-importiert werden
   - **SVG (.svg)**: Standard-Vektorformat - weit kompatibel mit Design-Software
   - **DXF (.dxf)**: CAD-Austauschformat - kompatibel mit den meisten CAD-Anwendungen

### Skizzen speichern

Du kannst deine 2D-Skizzen in Dateien speichern, um sie in anderen Projekten wiederzuverwenden. Alle parametrischen Einschränkungen werden beim Speichern beibehalten, was sicherstellt, dass deine Designs ihre geometrischen Beziehungen beibehalten.

### Skizzen importieren

Gespeicherte Skizzen können in jeden Arbeitsbereich importiert werden, was es dir ermöglicht, eine Bibliothek häufig verwendeter Designelemente zu erstellen. Der Importprozess behält alle Einschränkungen und dimensionalen Beziehungen bei.

## Workflow-Tipps

1. **Mit grober Geometrie beginnen**: Zuerst Basisformen erstellen, dann mit Einschränkungen verfeinern
2. **Einschränkungen früh verwenden**: Einschränkungen beim Aufbau anwenden, um Design-Absicht beizubehalten
3. **Einschränkungsstatus überprüfen**: Das System zeigt an, wann Skizzen vollständig eingeschränkt sind
4. **Auf Konflikte achten**: Einschränkungen, die miteinander in Konflikt stehen, werden rot hervorgehoben und im Einschränkungen-Panel angezeigt
5. **Symmetrie nutzen**: Symmetrie-Einschränkungen können komplexe Designs erheblich beschleunigen
6. **Raster verwenden**: Raster für präzise Ausrichtung aktivieren, und Strg zum Einrasten verwenden
7. **Iterieren und verfeinern**: Zögere nicht, Einschränkungen zu ändern, um das gewünschte Ergebnis zu erzielen

## Bearbeitungsfunktionen

- **Vollständige Rückgängig/Wiederholen-Unterstützung**: Der gesamte Skizzenzustand wird mit jeder Operation gespeichert
- **Dynamischer Cursor**: Der Cursor ändert sich, um das aktive Zeichenwerkzeug zu reflektieren
- **Einschränkungs-Visualisierung**: Angewendete Einschränkungen werden in der Oberfläche klar angezeigt
- **Echtzeit-Updates**: Änderungen an Einschränkungen aktualisieren die Geometrie sofort
- **Doppelklick-Bearbeitung**: Doppelklick auf dimensionale Einschränkungen (Abstand, Radius, Durchmesser, Winkel, Seitenverhältnis) öffnet einen Dialog zum Bearbeiten ihrer Werte
- **Parametrische Ausdrücke**: Dimensionale Einschränkungen unterstützen Ausdrücke, die es Werten ermöglichen, aus anderen Parametern berechnet zu werden (z.B. `breite/2` für einen Radius, der die Hälfte der Breite ist)
