# Pfade bereinigen

Logos und Zeichnungen aus anderen Programmen sind nicht immer sauber. Ein Umriss kann einen
Bruchteil eines Millimeters vor seinem Startpunkt enden, eine Form kann aus mehreren offenen Teilen
bestehen, oder derselbe Pfad steht zweimal in der Datei. Rayforge schneidet solche Pfade genau so,
wie sie sind: Eine Form, die nur geschlossen aussieht, wird als offen behandelt, und ein doppelter
Pfad wird zweimal geschnitten.

Die Werkzeuge zum Bereinigen von Pfaden beheben das bei den ausgewählten Werkstücken. Du findest sie
im Menü **Objekt** und im Kontextmenü der Arbeitsfläche unter **Pfade bereinigen**. Jedes Werkzeug
ist ein einzelner Rückgängig-Schritt.

## Pfade schließen

**Pfade schließen…** schließt jeden offenen Pfad, dessen Anfangs- und Endpunkt näher beieinander
liegen als die eingegebene Toleranz. Die Lücke wird mit einer geraden Linie zwischen den beiden
vorhandenen Punkten überbrückt, sodass sich die Form weder verschiebt noch ihre Größe ändert.

## Offene Pfade verbinden

**Offene Pfade verbinden…** verbindet offene Pfade, deren Endpunkte näher beieinander liegen als die
Toleranz, zu längeren Pfaden. Teile werden bei Bedarf umgedreht, die Reihenfolge und Richtung, in
der sie gezeichnet wurden, spielt also keine Rolle. Treffen sich danach die Enden eines verbundenen
Pfads, wird er außerdem geschlossen.

Beide Werkzeuge merken sich die zuletzt verwendete Toleranz für die laufende Sitzung. Für die
meisten importierten Dateien passt eine Toleranz zwischen 0,05 mm und 0,2 mm.

## Duplikate löschen

**Duplikate löschen** entfernt Pfade, die genau auf einem anderen Pfad im selben Werkstück liegen,
unabhängig von ihrer Richtung oder ihrem Startpunkt. Wenn du mehrere Werkstücke auswählst, wird auch
ein Werkstück entfernt, das eine exakte Kopie eines anderen ausgewählten Werkstücks an derselben
Position ist.

Überlappende Pfade, die nur einen Teil ihrer Länge gemeinsam haben, bleiben erhalten. Um diese
kümmert sich der Postprozessor [Linien zusammenführen](merge-lines), wenn der Auftrag erzeugt wird.

## Zerlegen

**Zerlegen** macht aus jedem Pfad eines Werkstücks ein eigenes Werkstück, einschließlich Löchern und
offener Pfade. Das unterscheidet sich von **Teilen**, das eine Insel mit ihren Löchern zusammenhält.

Das Bereinigen von Pfaden funktioniert nur bei importierten und nachgezeichneten Werkstücken.
Werkstücke aus einer Skizze bearbeitest du stattdessen im [Sketcher](sketcher/index).

## Verwandte Seiten

- [Linien zusammenführen](merge-lines) - Überlappende Segmente nur einmal schneiden
- [Canvas-Werkzeuge](../ui/canvas-tools) - Einzelne Segmente auswählen und löschen
- [Dateien importieren](../files/importing) - Designs in Rayforge laden
