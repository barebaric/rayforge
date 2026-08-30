---
description: "Fülle geschlossene Skizzenbereiche mit einfarbigen Flächen oder Verlaufsfüllungen im Rayforge-Sketcher."
---

# Bereiche füllen

Das Füll-Werkzeug (`G+F`) füllt geschlossene Bereiche einer Skizze mit
einer durchgehenden Fläche. Füllungen sind nützlich für Bereiche, die
als ein Stück graviert werden sollen.

![Ein gefülltes Rechteck](/screenshots/addons-sketcher-tool-fill.webp)

## Füllungen erstellen und entfernen

1. Zeichne ein oder mehrere geschlossene Konturen (zum Beispiel mit dem
   [Rechteck](rectangle.md)- oder [Pfad](path.md)-Werkzeug).
2. Wähle das Füll-Werkzeug aus dem Kreismenü, dem Menü **Sketch** oder
   drücke `G+F`.
3. Klicke irgendwo innerhalb eines geschlossenen Bereichs, um ihn zu
   füllen.
4. Klicke erneut auf einen gefüllten Bereich, um seine Füllung zu
   entfernen.

Ein Klick innerhalb eines Textfelds schaltet stattdessen die Füllung
der Textglyphen um, statt eine Bereichsfüllung zu erstellen.

## Füllfarbe

Die Füllfarbe für neue Füllungen wählst du mit der Schaltfläche
**Füllfarbe** in der Sketcher-Symbolleiste. Vorhandene Füllungen
behalten ihre Farbe, bis sie entfernt und neu erstellt werden.

Wie alles im Sketcher ist eine Füllung an ihre Begrenzung gebunden:
Verändere die Größe der umgebenden Geometrie, und die Füllung folgt.
