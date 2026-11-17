---
description:
  "Zeichne regelmäßige Polygone und Sterne im Rayforge-Sketcher. Sie bestehen aus gewöhnlichen
  Punkten und Linien, die von Einschränkungen regelmäßig gehalten werden."
---

# Polygon und Stern

Das **Polygon**-Werkzeug (`G+X`) zeichnet ein regelmäßiges Polygon, etwa ein Sechseck für einen
Untersetzer, oder einen Stern.

![Ein Sechseck und ein fünfzackiger Stern](/screenshots/addons-sketcher-tool-polygon.webp)

## Ein Polygon zeichnen

Klicke dorthin, wo der Mittelpunkt liegen soll, bewege den Zeiger und klicke erneut, oder drücke am
Mittelpunkt, ziehe und lasse los. Der Zeiger legt die erste Ecke fest und bestimmt damit sowohl die
Größe als auch die Drehung. Ein neues Polygon hat 6 Ecken.

- Halte `Ctrl` gedrückt, um stattdessen einen Stern zu zeichnen: Zwischen je zwei äußeren Ecken wird
  eine innere Ecke auf dem halben äußeren Radius eingefügt.

Während die Vorschau aktiv ist, kannst du exakte Werte eingeben: Die Statusleiste zeigt die Felder
`R`, `N` und `Star`. `R` ist der Radius vom Mittelpunkt zu den Ecken, `N` die Anzahl der Ecken (3
bis 100) und `Star` der innere Radius als Bruchteil des äußeren (zwischen 0 und 1). Die Eingabe
eines `Star`-Werts macht aus dem Polygon einen Stern. Gib einen Wert ein, drücke `Tab`, um zum
nächsten Feld zu wechseln, und `Enter`, um die Form zu erstellen; `Esc` bricht die Vorschau ab. Das
Werkzeug merkt sich die Anzahl der Ecken für das nächste Polygon.

## Was du erhältst

Ein Polygon ist kein spezielles Objekt: Es besteht aus gewöhnlichen Punkten und Linien, sodass
Verrundung, Fase, Versatz und boolesche Operationen damit wie mit jeder anderen Form funktionieren.

Einschränkungen halten es regelmäßig. Alle Kanten sind gleich lang, und die Ecken liegen auf einem
Konstruktionskreis um den Mittelpunkt; ein Stern hat einen zweiten Konstruktionskreis für seine
inneren Ecken und behält das Verhältnis zwischen den beiden Radien bei. Wenn du einen Radius
eingegeben hast, legt eine Radius-Einschränkung ihn fest. Du kannst dann den Mittelpunkt ziehen, um
das Polygon zu verschieben, oder eine Ecke ziehen, um es zu skalieren und zu drehen, und es bleibt
regelmäßig. Um die Größe später zu ändern, bearbeite die Radius-Einschränkung oder füge dem
Konstruktionskreis eine hinzu.
