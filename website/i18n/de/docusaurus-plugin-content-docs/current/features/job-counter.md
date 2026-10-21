---
description:
  "Ein Auftragszähler in Textvorlagen: eine Zahl, der nach jedem Auftrag weiterzählt, mit
  fortlaufenden Nummern für Array-Kopien."
---

# Auftragszähler

Textfelder lösen `{...}`-Vorlagen auf (siehe
[Parameter und Ausdrücke](sketcher/expressions.md#template-expressions-in-text-boxes)). Zusätzlich
zu den Skizzenparametern und den eingebauten Funktionen hat das Projekt einen Auftragszähler, der
sich von Auftrag zu Auftrag ändert. So gravierst du nummerierte Schilder, Anhänger und Etiketten,
ohne den Text jedes Mal zu bearbeiten.

## Der Auftragszähler

Verwende `{counter}` in einem Textfeld. Es ist eine ganze Zahl, daher bestimmen Formatangaben ihr
Aussehen:

- `{counter}` — `7`
- `{counter:04d}` — `0007`, vier Stellen mit führenden Nullen
- `No. {counter:03d}` — `No. 007`

Öffne **Werkzeuge → Auftragszähler…**, um den Zähler einzurichten:

- **Aktueller Wert** — die Nummer, mit der der nächste Auftrag beginnt.
- **Startwert** — wohin **Zurücksetzen** zurückkehrt.
- **Erhöhen um** — die Schrittweite pro Kopie und pro Auftrag.
- **Am Endwert umbrechen** / **Endwert** — nach dem Endwert macht der Zähler beim Startwert weiter.
- **Nach jedem Auftrag erhöhen** — wenn ein an die Maschine gesendeter Auftrag fertig ist, zählt der
  Zähler weiter. Rahmen, Zeiger-Probeläufe und abgebrochene Aufträge lassen ihn unverändert.
- **Nach G-Code-Export erhöhen** — auch beim Exportieren von G-Code weiterzählen, für Aufträge, die
  von einer Speicherkarte laufen.

Die Gruppe **Nächster Auftrag** zeigt eine Vorschau der Nummern, die der nächste Auftrag verwendet.
**Zurück** und **Weiter** verschieben den Zähler um einen Auftrag, **Zurücksetzen** kehrt zum
Startwert zurück. Jede Änderung, auch das automatische Weiterzählen, ist Teil des
Rückgängig-Verlaufs und wird im Projekt gespeichert.

## Kopien nummerieren

Jedes Werkstück hat einen **Variablen-Versatz**. Eine Kopie mit Versatz _n_ zeigt die Nummer _n_
Kopien nach der aktuellen. Ein [Gitter- oder Kreis-Array](arrays.md) gibt seinen Kopien automatisch
steigende Versätze, sodass ein Array aus 20 Schildern 20 fortlaufende Nummern graviert und der
nächste Auftrag mit der 21. weitermacht.

Arrays innerhalb einer Skizze sind anders: Ihre Kopien gehören zu einem Werkstück und zeigen alle
dieselbe Nummer. Erstelle das Array stattdessen auf der Leinwand, um die Kopien zu nummerieren.

Für Kopien, die du selbst platziert hast, wähle sie alle aus und wähle **Anordnen → Array → Kopien
nummerieren**: Die Versätze werden in Leserichtung vergeben, Zeilen von oben nach unten und jede
Zeile von links nach rechts.

Der Zähler erhöht sich um die Anzahl der Kopien im Auftrag: den höchsten Variablen-Versatz plus
eins.
