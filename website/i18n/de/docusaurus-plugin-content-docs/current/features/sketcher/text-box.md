---
description:
  "Platziere gravierten Text, Beschriftungen und Seriennummern mit dem Textfeld-Werkzeug im
  Rayforge-Sketcher auf einer Skizze."
---

# Textfeld

Das Textfeld-Werkzeug (`G+T`) platziert Text auf der Skizze als bearbeitbare Geometrie — gravierten
Text, Beschriftungen und Seriennummern. Textfelder sind vollständig parametrisch: Die Glyphen
befinden sich innerhalb eines durch Einschränkungen festgelegten Rahmens, sodass sie sich neu lösen,
sobald der Rahmen bewegt oder bemaßt wird.

![Ein Wortzeichen und eine Teilebeschriftung](/screenshots/addons-sketcher-tool-text-box.webp)

## Text erstellen und bearbeiten

1. Wähle das Textfeld-Werkzeug aus dem Kreismenü, dem Menü **Sketch** oder mit `G+T`.
2. Klicke an die Stelle, an der der Text beginnen soll: Am Klickpunkt erscheint ein Textfeld, und
   das Werkzeug wechselt direkt in den Bearbeitungsmodus.
3. Tippe den Text ein — das Feld passt seine Größe beim Tippen automatisch an.
4. Drücke `Enter` oder `Escape`, um die Bearbeitung zu beenden.

Um ein vorhandenes Textfeld zu bearbeiten, klicke hinein. Ein Doppelklick wählt ein Wort aus, ein
Dreifachklick die ganze Zeile, und Text kann wie in jedem Texteditor ausgewählt und ersetzt werden —
inklusive `Ctrl+C`/`Ctrl+V`, Rückgängig/Wiederholen und Einfügen während der Bearbeitung.

## Schriftarteigenschaften

![Das Schriftarteigenschaften-Bedienfeld](/screenshots/addons-sketcher-tool-text-box-font-properties.webp)

Das Bedienfeld **Schriftarteigenschaften** in der Seitenleiste steuert das Erscheinungsbild des auf
der Leinwand ausgewählten Textfelds:

- **Schriftfamilie** — wähle aus den installierten Systemschriften.
- **Schriftgröße** — in Punkt.
- **Fett**- und **Kursiv**-Schalter.

## Ein parametrischer Rahmen

Ein Textfeld ist kein Rasterbild: Seine Glyphen sind echte Skizzengeometrie, die innerhalb eines
Rahmens aus einem Ursprung sowie Breiten- und Höhenpunkten angeordnet ist. Der Rahmen wird als
Konstruktionsgeometrie gestrichelt gezeichnet, dient also als Layout-Referenz und landet beim
Fertigen der Skizze nie in den Werkzeugpfaden. Wie alles andere im Sketcher ist auch der Rahmen mit
Einschränkungen versehen, sodass er wie jede andere Geometrie bemaßt werden kann — ändere die
Breiteneinschränkung, und der Text wird neu gelöst, um das Feld zu füllen.

Ein Klick innerhalb eines Textfelds mit dem [Füll-Werkzeug](fill.md) schaltet stattdessen die
Füllung der Textglyphen um, statt eine Bereichsfüllung zu erzeugen.

## Vorlagenausdrücke

Textfelder akzeptieren **Vorlagenausdrücke**: Alles in geschweiften Klammern wird beim Lösen der
Skizze ausgewertet, sodass Beschriftungen Live-Werte wie Maße, Daten oder eindeutige Seriennummern
anzeigen können. Einzelheiten und die eingebauten Funktionen findest du unter
[Vorlagenausdrücke in Textfeldern](expressions.md#template-expressions-in-text-boxes).
