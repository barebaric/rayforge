---
description: "Verwende den integrierten parametrischen 2D-Sketcher von Rayforge, um maßgeschneiderte laserfertige Designs mit Linien, Kreisen, Bezier-Kurven und Einschränkungen zu erstellen."
---

# Parametrischer 2D-Sketcher

Der Parametrische 2D-Sketcher ist eine leistungsstarke Funktion in Rayforge, mit der du präzise, einschränkungsbasierte 2D-Designs direkt in der Anwendung erstellen und bearbeiten kannst. Diese Funktion ermöglicht es dir, benutzerdefinierte Teile von Grund auf zu entwerfen, ohne externe CAD-Software zu benötigen.

## Übersicht

Der Sketcher bietet einen vollständigen Satz von Werkzeugen zum Erstellen geometrischer Formen und zum Anwenden parametrischer Einschränkungen, um präzise Beziehungen zwischen Elementen zu definieren. Dieser Ansatz stellt sicher, dass deine Designs ihre beabsichtigte Geometrie beibehalten, auch wenn Abmessungen geändert werden.

## Skizzen erstellen und bearbeiten

### Eine neue Skizze erstellen

1. Öffne das untere Panel und klicke auf die Schaltfläche **Neue Skizze**, oder
   rechtsklicke auf die Arbeitsfläche und wähle **Neue Skizze** aus dem
   Kontextmenü.
2. Ein neuer leerer Sketch-Arbeitsbereich öffnet sich mit der Sketch-Editor-Oberfläche
3. Mit dem Erstellen von Geometrie beginnen, indem du die Zeichenwerkzeuge aus dem Kreismenü oder Tastatur-Kurzbefehlen verwendest
4. Einschränkungen anwenden, um Beziehungen zwischen Elementen zu definieren
5. Auf "Skizze fertigstellen" klicken, um deine Arbeit zu speichern und zum Hauptarbeitsbereich zurückzukehren

### Bestehende Skizzen bearbeiten

1. Auf ein skizzenbasiertes Werkstück im Hauptarbeitsbereich doppelklicken
2. Alternativ eine Skizze auswählen und "Skizze bearbeiten" aus dem Kontextmenü wählen
3. Modifikationen mit denselben Werkzeugen und Einschränkungen vornehmen
4. Auf "Skizze fertigstellen" klicken, um Änderungen zu speichern, oder "Skizze abbrechen", um sie zu verwerfen

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
