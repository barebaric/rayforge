---
description:
  "Erstelle ein Konstruktionsraster aus Zeilen und Spalten als Zeichen-Gerüst im Rayforge-Sketcher."
---

# Raster

Das Raster-Werkzeug (`G+G`) erstellt ein gleichmäßiges Raster aus Konstruktionslinien — Zeilen und
Spalten gleichmäßig verteilter Hilfslinien, die als Zeichen-Gerüst dienen, zum Beispiel um ein
Perforationsmuster anzulegen oder wiederholte Elemente auszurichten.

![Ein 4x6-Konstruktionsraster](/screenshots/addons-sketcher-tool-grid.webp)

1. Wähle das Raster-Werkzeug aus dem Kreismenü, dem Menü **Sketch** oder mit `G+G`.
2. Ein Dialog fragt nach der Anzahl der **Zeilen** und **Spalten**.
3. Bestätige, um das Raster am Skizzen-Ursprung mit 10-mm-Zellen zu erstellen.

Das Raster besteht aus Konstruktionsgeometrie: Es wird gestrichelt gezeichnet, dient wie jede andere
Geometrie als Einrast- und Ausrichtungsreferenz und wird beim Fertigen der Skizze von den
Werkzeugpfaden ausgeschlossen (siehe [Konstruktionsgeometrie](index.md#construction-geometry)).
Einzelne Linien können wie jede andere Geometrie verschoben oder gelöscht werden, und wenn du sie
auswählst und den Konstruktionsmodus mit `G+N` umschaltest, wird das Gerüst zu echter Geometrie.
