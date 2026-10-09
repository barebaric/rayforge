---
description:
  "Gestrichelte Schnitte in Rayforge mit dem Perforation-Nachbearbeitungsschritt: Schnittlänge und
  Auslasslänge für Falzlinien, Abreißlinien und Nählöcher festlegen."
---

# Perforation

Der Nachbearbeitungsschritt **Perforation** macht aus einem durchgehenden Schnitt einen
gestrichelten. Der Laser schneidet über die **Schnittlänge**, fährt dann mit ausgeschaltetem Laser
über die **Auslasslänge** und wiederholt das entlang jeder Kontur des Werkstücks.

## Wann verwenden

- Falzlinien in Karton und Wellpappe
- Abreißlinien (Tickets, Gutscheine, Verpackungen)
- Nählöcher in Leder
- Dekorative gestrichelte Linien

## Einstellungen

Perforation ist in den Nachbearbeitungseinstellungen von **Kontur**-Operationen verfügbar und
standardmäßig ausgeschaltet.

- **Schnittlänge**: die Strecke, die der Laser vor jeder Lücke schneidet.
- **Auslasslänge**: die Strecke, die zwischen zwei Schnitten mit ausgeschaltetem Laser gefahren
  wird.

Das Muster wird entlang jeder Kontur des Werkstücks gemessen und beginnt am Anfang jeder Kontur neu,
immer mit einem vollen Schnitt. Eine Kontur, die kürzer ist als ein Schnitt plus eine Lücke, wird
vollständig geschnitten.

## Tipps

- Für eine Falzlinie, die nicht durchschneiden soll, mit etwa gleich langer Schnitt- und
  Auslasslänge beginnen und die Leistung senken.
- Sehr kurze Längen (wenige Zehntelmillimeter) lassen den Laser schnell ein- und ausschalten, was
  die effektive Leistung verringert.
- Perforation funktioniert zusammen mit [Halte-Laschen](holding-tabs), [Lead-In/Out](lead-in-out)
  und [Mehrfachdurchgang](multi-pass).

## Verwandte Seiten

- [Konturschneiden](operations/contour) - Die Schneidoperation, die Perforation verwendet
- [Halte-Laschen](holding-tabs) - Von Hand gesetzte Lücken, die Teile festhalten
