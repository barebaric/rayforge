---
description: "Erfahre mehr über geometrische und dimensionale Einschränkungen im parametrischen 2D-Sketcher von Rayforge."
---

# Parametrisches Einschränkungssystem

Das Einschränkungssystem ist der Kern des parametrischen Sketchers und ermöglicht es dir, präzise geometrische Beziehungen zu definieren:

## Geometrische Einschränkungen

- **Koinzident**: Zwingt zwei Punkte, dieselbe Position einzunehmen
- **Vertikal**: Schränkt eine Linie ein, perfekt vertikal zu sein
- **Horizontal**: Schränkt eine Linie ein, perfekt horizontal zu sein
- **Tangential**: Macht eine Linie tangential zu einem Kreis oder Bogen
- **Senkrecht**: Zwingt zwei Linien, eine Linie und einen Bogen/Kreis, oder zwei Bögen/Kreise, sich in einem 90-Grad-Winkel zu treffen
- **Punkt auf Linie/Form**: Schränkt einen Punkt ein, auf einer Linie, einem Bogen oder einem Kreis zu liegen
- **Kollinear**: Zwingt zwei oder mehrere Linien, auf derselben unendlichen Linie zu liegen
- **Symmetrie**: Erzeugt symmetrische Beziehungen zwischen Elementen. Unterstützt zwei Modi:
  - **Punkt-Symmetrie**: 3 Punkte auswählen (der erste ist das Zentrum)
  - **Linien-Symmetrie**: 2 Punkte und 1 Linie auswählen (die Linie ist die Achse)

## Dimensionale Einschränkungen

- **Abstand**: Setzt den exakten Abstand zwischen zwei Punkten oder entlang einer Linie
- **Durchmesser**: Definiert den Durchmesser eines Kreises
- **Radius**: Setzt den Radius eines Kreises oder Bogens
- **Winkel**: Erzwingt einen spezifischen Winkel zwischen zwei Linien
- **Seitenverhältnis**: Zwingt das Verhältnis zwischen zwei Abständen, gleich einem angegebenen Wert zu sein
- **Gleiche Länge/Gleicher Radius**: Zwingt mehrere Elemente (Linien, Bögen, Ellipsen oder Kreise), dieselbe Länge oder denselben Radius zu haben
- **Gleicher Abstand**: Zwingt den Abstand zwischen zwei Punkt-Paaren, gleich zu sein
