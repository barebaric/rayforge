---
description: "Erstelle kreisförmige Arrays und Arrays entlang einer Kurve im parametrischen Sketcher von Rayforge."
---

# Arrays

Der Sketcher bietet zwei Array-Werkzeuge zum Erstellen parametrischer Arrays:
**Kreisförmiges Array** und **Array entlang Kurve**.

## Kreisförmige Arrays

Das **Kreisförmige Array**-Werkzeug (`G+Y`) erstellt ein parametrisches
Polarmuster aus der aktuellen Auswahl:

1. Wähle die Entitäten aus, die du musterhaft anordnen möchtest.
2. Aktiviere das Werkzeug über die Symbolleiste, das Menü **Sketch → Arrays**
   oder `G+Y`.
3. Ein Hilfskreis erscheint auf der Arbeitsfläche und ein nicht-modaler Dialog
   öffnet sich mit einer Live-Vorschau.
4. Setze die **Anzahl** und den **Gesamtwinkel**. Kopien werden parametrisch
   um den Mittelpunkt des Hilfskreises erzeugt.
5. Ziehe den Mittelpunkt des Hilfskreises, um das Array umzupositionieren,
   oder ziehe die ursprüngliche Entität, um den Radius zu ändern — die
   Dialogfelder aktualisieren sich live.
6. Die **Radius-Dimension** des Hilfskreises skaliert das gesamte Array.
   **Doppelklicke** auf den Hilfskreis, um den Bearbeitungsdialog erneut zu
   öffnen und fehlende Elemente zu erzeugen oder die Verteilung zu ändern.

Kopien sind statisch gebackene Geometrie ohne Solver-Einschränkungen:
Sie werden bei der Bearbeitung des Arrays aus der Vorlage neu erzeugt.
Das Löschen eines Elements entfernt nur dessen eigene Geometrie und
verteilt die Übrigen nie um.

## Array entlang Kurve

Das **Array entlang Kurve**-Werkzeug verteilt Kopien eines oder mehrerer
Entitäten entlang eines Hilfspfads (einer Linie, eines Bogens oder einer
Bezier-Kurve). Die Kopien werden direkt auf dem Pfad platziert und folgen
seiner Tangente an jeder Position.

### Ein Array entlang Kurve erstellen

1. Zeichne die Form, die du verteilen möchtest (die Vorlage), und den
   Hilfspfad, dem du folgen möchtest.
2. Wähle beide aus: Zuerst den **Hilfspfad** anklicken, dann mit Umschalt
   die **Vorlagen-Entitäten** auswählen.
3. Aktiviere das Werkzeug über die Symbolleiste, das Menü **Sketch → Arrays**
   oder `G+W`.
4. Ein nicht-modaler Dialog öffnet sich mit einer Live-Vorschau mit Kopien,
   die entlang des Pfads verteilt sind.
5. Passe die **Anzahl** an (Gesamtzahl inklusive Vorlage am Pfadanfang) oder
   setze einen **Abstandswert**, um die Anzahl automatisch aus der
   Pfadlänge abzuleiten.
6. Aktiviere optional **An Tangente ausrichten**, damit sich jede Kopie an die
   Pfadrichtung an ihrer Position ausrichtet.
7. Verwende **Versatz vom Start**, um einen führenden Abschnitt des Pfads
   zu überspringen, bevor die erste Kopie platziert wird.

### Ein Array entlang Kurve bearbeiten

- **Doppelklicke** auf den Hilfspfad (oder klicke auf **Bearbeiten** in der
  Symbolleiste), um den Dialog erneut zu öffnen und Anzahl, Abstand,
  Versatz oder Ausrichtungseinstellungen zu ändern.
- **Ziehe** an einem Endpunkt des Hilfspfads, um ihn umzuformen. Beim
  Loslassen werden alle Kopien automatisch entlang der neuen Pfadgeometrie
  umverteilt — einschließlich Rotationsaktualisierungen, wenn *An Tangente
  ausrichten* aktiviert ist.
- Die Vorlage kann wie jede andere Sketch-Geometrie bearbeitet werden;
  Änderungen werden beim nächsten Update auf alle Kopien übertragen.

### Funktionsweise

Kopien sind statisch gebackene Geometrie — sie sind nicht über
Solver-Einschränkungen mit der Vorlage verknüpft. Wenn der Hilfspfad
bearbeitet wird, erkennt `sync_arrays` die Änderung und
regeneriert alle Kopien von Grund auf mit der aktuellen Pfadgeometrie.
Dies hält Updates schnell und vermeidet Solver-Overhead.

Die Vorlage (Slot 0) wird am Pfadanfang platziert. Ihre Position und
Ausrichtung aktualisieren sich automatisch, wenn der Pfad bearbeitet
wird. Die ursprünglichen Vorlagen-Entitäten werden beim Erstellen des
Arrays entfernt; Rückgängig stellt sie wieder her.
