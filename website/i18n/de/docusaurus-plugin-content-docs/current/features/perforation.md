---
description:
  "Gestrichelte Schnitte in Rayforge mit dem Perforation-Nachbearbeitungsschritt: Schnittlänge und
  Auslasslänge für Falzlinien, Abreißlinien und Nählöcher festlegen."
---

# Perforation

Nicht jeder Schnitt soll ein Teil aus dem Material lösen. Manche Linien sind da, damit sich das
Material entlang ihnen falten lässt, sauber abreißt oder den Weg der Nadel vorgezeichnet. Dafür gibt
es den Nachbearbeitungsschritt **Perforation**: Er macht aus einem durchgehenden Schnitt einen
gestrichelten. Statt den Laser von Anfang bis Ende eingeschaltet zu lassen, feuert er über die
**Schnittlänge**, fährt die **Auslasslänge** mit ausgeschaltetem Laser zurück und wiederholt diesen
Rhythmus entlang jeder Kontur des Werkstücks. Übrig bleibt eine Reihe kleiner Schnitte, die das
Material zusammenhalten — bis es nachgeben soll.

![Die Perforation-Einstellungen auf der Nachbearbeitung-Seite eines Kontur-Schritts](/screenshots/step-settings-contour-post-perforation.webp)

## Beispielhafte Anwendungsfälle

Überall dort, wo eine Linie das Material schwächen soll, ohne es zu durchtrennen, lohnt sich ein
perforierter Schnitt. Die klassischen Fälle:

- **Falzlinien** in Karton und Wellpappe brechen entlang einer gestrichelten Linie sauber — etwa bei
  Schachteln, Karten und Pop-up-Designs.
- **Abreißlinien** öffnen Tickets, Gutscheine und Verpackungen ganz ohne Schere.
- **Nählöcher** in Leder markieren den Weg der Nadel, bevor genäht wird.
- **Gestrichelte Konturen** sind darüber hinaus ein dekorativer Effekt — auf Holz, Papier und Acryl.

## Perforation aktivieren

Perforation findet sich auf der Seite **Nachbearbeitung** der Einstellungen eines
[Kontur](operations/contour)-Schritts, neben den übrigen Nachbearbeitungsschritten. Sie ist
standardmäßig ausgeschaltet — sobald der Schalter umgelegt wird, erscheinen zwei Werte:

- Die **Schnittlänge** gibt an, wie weit der Laser vor jeder Lücke eingeschaltet bleibt.
- Die **Auslasslänge** ist die Strecke, die zwischen zwei Schnitten mit ausgeschaltetem Laser
  zurückgelegt wird.

Das Muster beginnt am Anfang jeder Kontur mit einem vollen Schnitt und beginnt auf jeder Kontur neu,
sodass alle gestrichelten Linien des Werkstücks gleich ansetzen. Eine Kontur, zu klein für auch nur
einen Schnitt und eine Lücke, wird einfach am Stück geschnitten.

## Das Ergebnis

Auf der Arbeitsfläche ersetzt die gestrichelte Linie die durchgehende Schnittkontur — jede Lücke im
Werkzeugpfad ist auch eine Lücke im fertigen Teil.

![Ein Rechteck und ein Kreis werden als gestrichelte Linien in eine Birkenplywood-Platte geschnitten](/screenshots/operations-perforation.webp)

## Tipps

Für eine Falzlinie, die nicht durchtrennen soll, beginnt man am besten mit einer Auslasslänge von
etwa der Schnittlänge und senkt die Leistung — eine flache Rille faltet genauso gut wie ein tiefer
gestrichelter Schnitt, und die Unterseite bleibt sauber.

Sehr kurze Längen (wenige Zehntelmillimeter) lassen den Laser in schneller Folge ein- und
ausschalten, was die effektive Leistung verringert und ungleichmäßige Spuren hinterlassen kann.
Wirkt das Muster uneinheitlich, helfen längere Werte.

Perforation lässt sich gut mit dem Rest der Nachbearbeitung kombinieren:
[Halte-Laschen](holding-tabs), [Einlauf / Auslauf](lead-in-out) und [Mehrfach-Durchgang](multi-pass)
funktionieren wie gewohnt.

## Verwandte Seiten

- [Kontur-Schneiden](operations/contour) — die Schneidoperation, zu der dieser
  Nachbearbeitungsschritt gehört
- [Halte-Laschen](holding-tabs) — von Hand gesetzte Lücken, die Teile festhalten
