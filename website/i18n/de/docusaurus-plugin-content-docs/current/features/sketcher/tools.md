---
description: "Sketcher-Werkzeuge, Tastatur-Kurzbefehle, Kreismenü, Konstruktionsmodus, Raster, Einrasten, Versatz, Fase und Verrundung in Rayforge."
---

# Sketcher-Werkzeuge

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
- `G+Y`: Kreisförmiges Array-Werkzeug
- `G+W`: Array-entlang-Kurve-Werkzeug
- `G+G`: Raster-Werkzeug (Raster von Kopien aus der Auswahl erstellen)
- `G+N`: Konstruktionsmodus auf Auswahl umschalten

### Aktions-Kurzbefehle

- `O+F`: Kontur versetzen
- `C+H`: Fase-Ecke hinzufügen
- `C+F`: Verrundungs-Ecke hinzufügen
- `C+S`: Ausgewählte Bezier-Kurven zu Linien glätten
- `M+V`: Auswahl vertikal spiegeln
- `M+H`: Auswahl horizontal spiegeln
- `Strg+D`: Auswahl vor Ort duplizieren

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
- `Strg+D`: Ausgewählte Elemente duplizieren
- `Entf`: Ausgewählte Elemente löschen
- `Pfeiltasten`: Ausgewählte Entitäten verschieben (`Umschalt` für größeren Schritt)
- `Escape`: Aktuelle Operation abbrechen oder Auswahl aufheben
- `F`: Ansicht an Inhalt anpassen

## Spiegeln, Duplizieren und Verschieben

Mehrere Transformationstools arbeiten mit der aktuellen Auswahl:

- **Vertikal / Horizontal spiegeln** (`M+V` / `M+H`): Spiegelt die
  Auswahl vor Ort über die Mitte ihres Begrenzungsrahmens. Einschränkungen,
  die die Auswahlgrenze überschreiten, werden entfernt; interne
  Einschränkungen bleiben erhalten.
- **Duplizieren** (`Strg+D`): Kopiert die Auswahl vor Ort. Die Kopien
  bekommen neue IDs und zugewiesene interne Einschränkungen; danach bleiben
  nur die Kopien ausgewählt. Rückgängig entfernt sie.
- **Verschieben**: Mit ausgewählten Entitäten bewegen die **Pfeiltasten**
  die Auswahl. Halte `Umschalt` für einen größeren Verschiebeschritt.

Diese sind über die Symbolleiste und das **Sketch**-Menü verfügbar.

## Konstruktionsmodus

Der Konstruktionsmodus ermöglicht es dir, Entitäten als "Konstruktionsgeometrie" zu markieren - Hilfselemente, die verwendet werden, um dein Design zu leiten, aber nicht Teil der endgültigen Ausgabe sind. Konstruktions-Entitäten werden anders angezeigt (typischerweise als gestrichelte Linien) und werden nicht eingeschlossen, wenn die Skizze zum Laserschneiden oder Gravieren verwendet wird.

Um den Konstruktionsmodus umzuschalten:

- Eine oder mehrere Entitäten auswählen
- `N` oder `G+N` drücken, oder die Konstruktionsoption im Kreismenü verwenden

Konstruktions-Entitäten sind nützlich für:

- Erstellen von Referenzlinien und -kreisen
- Definieren temporärer Geometrie zur Ausrichtung
- Aufbauen komplexer Formen aus einem Rahmen von Hilfslinien

## Sichtbarkeits-Steuerung

Das Raster passt sich der Zoomstufe an und steht stets als Größenreferenz
zur Verfügung; wie das Einrasten funktioniert, ist in der
[Sketcher-Übersicht](index.md#raster-und-einrasten) beschrieben.

Die Sketcher-Symbolleiste enthält Umschalt-Buttons zur Sichtbarkeitssteuerung:

- **Konstruktionsgeometrie anzeigen/verbergen**: Sichtbarkeit von
  Konstruktions-Entitäten umschalten
- **Einschränkungen anzeigen/verbergen**: Sichtbarkeit von
  Einschränkungs-Markierungen umschalten

Diese Steuerungen helfen, visuelle Unordnung bei der Arbeit an komplexen Skizzen
zu reduzieren.

### Automatische Einschränkung bei Erstellung

Viele Zeichenwerkzeuge wenden automatisch Einschränkungen an, während du
Geometrie erstellst. Das Pfad-Werkzeug erstellt horizontale und vertikale
Einschränkungen, wenn Einrast-Hilfslinien während des Zeichnens eine
Ausrichtung anzeigen, was hilft, deine Skizze von Anfang an ordentlich zu
halten, anstatt nachträglich Korrekturen vorzunehmen.

### Achsenbeschränkte Bewegung

Beim Ziehen von Punkten oder Geometrie, halte `Umschalt` um die Bewegung auf die
nächstgelegene Achse (horizontal oder vertikal) zu beschränken. Dies ist nützlich,
um die Ausrichtung bei Anpassungen beizubehalten.

## Kontur versetzen

Das Versatz-Werkzeug vergrößert oder verkleinert eine ausgewählte Kontur um
einen angegebenen Abstand, oder erweitert einen offenen Pfad zu einem Schlitz.
Wähle die Entitäten aus, die eine Kontur bilden (oder verwende Doppelklick, um
verbundene Geometrie auszuwählen), drücke dann `O+F` oder verwende den
**Versatz**-Eintrag im Kreismenü.

![Kontur-versetzen-Dialog](/screenshots/addons-sketcher-offset-dialog.webp)

Der Dialog fragt nach dem Versatzabstand und zeigt während der Eingabe eine
Live-Vorschau des Ergebnisses auf der Leinwand:

- **Geschlossene Konturen** wachsen bei einem positiven Abstand und schrumpfen
  bei einem negativen. Ein Versatz, über den die Kontur kollabieren würde,
  wird abgelehnt.
- **Offene Pfade** werden zu einer geschlossenen Schlitzkontur der angegebenen
  Breite mit abgerundeten Endkappen.

![Bezier-Kontur](/screenshots/addons-sketcher-offset-before.webp)
![Bezier zu einem Schlitz versetzt](/screenshots/addons-sketcher-offset-after.webp)

Beim Versetzen wird die ausgewählte Kontur durch das Ergebnis ersetzt:

- Einzelne Kreise, Bögen und Ellipsen behalten ihren Entitätstyp und werden
  direkt aktualisiert, sodass sie wie zuvor bearbeitbar und einschränkbar
  bleiben.
- Ketten verbundener Segmente (einschließlich Beziers) werden durch eine
  Polygon-Entität ersetzt. Das Polygon wird als Ganzes bearbeitet: Ziehe den
  Mittelpunkt, um es zu verschieben, und den Griffpunkt, um es zu drehen oder
  gleichmäßig zu skalieren.

Enthält die Auswahl mehrere getrennte Konturen, wird jede in einem einzigen
Schritt unabhängig versetzt.

## Fase und Verrundung

Der Sketcher bietet Werkzeuge zum Modifizieren von Ecken deiner Geometrie:

- **Fase**: Ersetzt eine scharfe Ecke durch eine abgeschrägte Kante. Einen Verbindungspunkt auswählen (wo sich zwei Linien treffen) und die Fasen-Aktion anwenden.
- **Verrundung**: Ersetzt eine scharfe Ecke durch eine abgerundete Kante. Einen Verbindungspunkt auswählen (wo sich zwei Linien treffen) und die Verrundungs-Aktion anwenden.

Fase oder Verrundung verwenden:

1. Einen Verbindungspunkt auswählen, wo sich zwei Linien treffen
2. `C+H` für Fase oder `C+F` für Verrundung drücken
3. Das Kreismenü oder Tastatur-Kurzbefehle verwenden, um die Modifikation anzuwenden
