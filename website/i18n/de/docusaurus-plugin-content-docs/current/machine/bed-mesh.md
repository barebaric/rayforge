---
description:
  "Tastet die Arbeitsfläche deines Lasers rasterförmig ab und gleicht eine unebene Plattform
  automatisch aus. Hält den Fokuspunkt auch auf großen oder gewellten Betten im Material."
---

# Bettnetz

Große Arbeitsflächen sind selten perfekt eben. Wenn die Betthöhe stärker schwankt als die
Schärfentiefe deines Lasers, werden Schnitte über das Material hinweg uneinheitlich. Die
Bettnetz-Funktion tastet die Oberflächenhöhe an einem Raster ab und gleicht die Z-Achse des
Werkzeugwegs an die tatsächliche Oberfläche an.

![Bettnetz](/screenshots/machine-settings-bed-mesh.webp)

## Voraussetzungen

Das Bettnetz benötigt eine Maschine mit Z-Achse und einen Treiber mit Tastenunterstützung (GRBL,
Marlin, Smoothie und OctoPrint derzeit). Die Seite **Bettnetz** erscheint nur unter **Einstellungen
→ Maschine**, wenn beides zutrifft. Sie befindet sich nach der Seite „Gerät“.

## Das Bett abtasten

Öffne **Einstellungen → Maschine** und navigiere zur Seite **Bettnetz**.

1. **Tastraster**: Lege den zu tastenden Bereich fest (X/Y-Ursprung, Breite, Höhe) sowie die
   Rasterdichte (Spalten und Zeilen). Die Seite zeigt die Anzahl der Tastpunkte und eine geschätzte
   Dauer. Feinere Raster folgen der Oberfläche genauer, dauern aber länger.
2. **Tasten**: Stelle die Tastgeschwindigkeit ein, wie weit der Kopf an jedem Punkt nach unten
   suchen darf (Maximaler Hub) und die sichere Z-Höhe für Bewegungen zwischen den Punkten.
3. Klicke auf **Abtastung starten**. Die Maschine fährt jeden Rasterpunkt im Schlangenlinienmuster
   an und berührt die Oberfläche; die 3D-Ansicht füllt sich während der Messung live. Du kannst den
   Lauf jederzeit stoppen; das Netz wird erst gespeichert, wenn das komplette Raster abgeschlossen
   ist.

Das Netz wird mit dem Maschinenprofil gespeichert und als farbige 3D-Fläche dargestellt: blaue
Bereiche sind tiefer, rote höher. Mit **Netz löschen** entfernst du das Netz wieder, wenn du keine
Höhenkompensation mehr möchtest.

:::note Beim Abtasten bewegt sich der Kopf über das gesamte Rastergebiet. Räume das Bett von
Objekten, die den Taster blockieren könnten, und stelle sicher, dass dein Taster (oder
Laser-Fadenkreuz) die Oberfläche an jedem Rasterpunkt tatsächlich erreichen kann. :::

## Das Netz anwenden

Das Aufnehmen des Netzes ändert nur die Höhenkarte — die Anwendung wird pro Auftrag gesteuert:

- **Aufträge standardmäßig anwenden**: Aktiviere diese Option auf der Bettnetz-Seite, um die Z-Achse
  jedes Auftrags auf dieser Maschine zu kompensieren.
- **Pro-Ebene-Überschreibung**: In den Ebenen-Einstellungen (Ebeneneinstellungen → Nachbearbeitung)
  erscheint der Maschinenstandard mit den Aktionen **Anpassen** und **Deaktivieren**. Anpassen
  kopiert die Korrektur in die Ebene, wo du den Z-Versatz ändern kannst (z. B. um etwas über der
  Oberfläche zu fokussieren); Deaktivieren schaltet die Korrektur nur für diese Ebene aus. Entfernt
  man den Eintrag der Ebene, greift wieder der Maschinenstandard.

Die Korrektur gilt für Schneide- und Bewegungsbewegungen gleichermaßen, und Bögen bleiben Bögen (sie
werden zu Helixbewegungen). Sie läuft auf dem fertigen Werkzeugweg in Maschinenkoordinaten und ist
daher von Arbeitskoordinatensystemen oder der Ursprungsecke der Maschine unbeeinflusst.

---

## Verwandte Seiten## Verwandte Seiten

- [Hardware-Einstellungen](hardware) - Maschinenabmessungen und Achsenkonfiguration
- [Geräteeinstellungen](device) - Verbindung und Controller-Optionen
- [3D-Ansicht](../ui/3d-preview.md) - 3D-Werkzeugweg-Visualisierung
