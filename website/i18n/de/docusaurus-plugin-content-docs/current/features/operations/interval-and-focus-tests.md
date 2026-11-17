---
description:
  "Finde mit dem Intervalltest und dem Fokustest den besten Linienabstand zum Gravieren und den
  Brennpunkt deines Lasers."
---

# Intervall- und Fokustests

Neben dem [Materialtest-Raster](material-test-grid.md) bietet das Menü Werkzeuge zwei weitere
Kalibrierungsjobs. Alle drei funktionieren gleich: **Werkzeuge → … erstellen** fügt den Test mit
Standardeinstellungen auf einer neuen Ebene hinzu und wählt ihn aus, und seine Einstellungen
erscheinen unter **Test Settings** im Eigenschaften-Panel rechts. Jede Änderung dort baut den Test
sofort neu auf, als ein Schritt, den du rückgängig machen kannst, sodass du das Ergebnis siehst,
während du ihn anpasst. Der Test ist normaler Inhalt: Du kannst ihn verschieben, umranden und in der
Vorschau ansehen wie alles andere. Jedes Feld und jede Linie hat eine eigene Operation, die du auch
von Hand feinjustieren kannst, aber eine spätere Änderung in den Test Settings baut diese
Operationen neu auf.

## Intervalltest

**Werkzeuge → Intervalltest erstellen** graviert eine Reihe gefüllter Quadrate. Jedes Quadrat hat
eine eigene **Gravieren**-Operation mit eigenem Linienabstand, gleichmäßig verteilt vom kleinsten
bis zum größten eingegebenen Abstand, während Leistung und Geschwindigkeit für alle gleich bleiben.
Die Beschriftungen unter jedem Quadrat zeigen den Abstand in Millimetern und die passende
Liniendichte in Linien pro Zoll (LPI).

![Intervalltest](/screenshots/interval-test.webp)

Wähle das Quadrat, das gleichmäßig ohne sichtbare Linien und ohne zu tiefes Einbrennen gefüllt ist,
und verwende seinen Abstand für Gravuren auf diesem Material. Die Beschriftungen werden vor den
Quadraten mit einer eigenen Operation mit geringer Leistung geschnitten.

## Fokustest

**Werkzeuge → Fokustest erstellen** findet die Höhe, in der der Strahl am schärfsten ist. Ein
positiver Versatz bedeutet mehr Abstand zwischen Kopf und Material. Die dünnste Linie markiert den
besten Fokus.

![Fokustest](/screenshots/focus-test.webp)

| Methode                             | Wie sich die Höhe ändert                                                                         |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Z-Achsen-Schritte**               | Der Kopf fährt mit relativen Z-Bewegungen zu jedem Versatz und am Ende zurück zur Ausgangshöhe   |
| **Manuell (Pause zwischen Linien)** | Der Job pausiert (`M0`) vor jeder Linie; du verschiebst den Kopf von Hand und drückst Fortsetzen |
| **Rampe (schräges Material)**       | Eine lange Linie mit Abstandsstrichen; du stützt ein Ende eines flachen Streifens ab             |

Z-Achsen-Schritte werden nur bei Maschinen mit Z-Achse angeboten, und sowohl Z-Achsen-Schritte als
auch die manuelle Methode benötigen einen G-Code-Controller, weil sie zwischen den Linien
[Befehls](command.md)-Operationen verwenden. Die Versätze sind auf ±10 mm von der Ausgangshöhe
begrenzt. Ein neuer Fokustest verwendet die erste Methode, die die Maschine unterstützt, in der
Reihenfolge der Tabelle oben.

Fokussiere den Laser bei der manuellen Methode zuerst wie gewohnt. Die Beschriftungen werden in
dieser Höhe graviert. Stelle den Kopf bei der ersten Pause auf den ersten Versatz und verschiebe ihn
bei jeder weiteren Pause um einen Schritt. Prüfe, ob dein Controller bei `M0` anhält und die
Fortsetzen-Schaltfläche den Job weiterlaufen lässt, bevor du dich darauf verlässt, zum Beispiel mit
einem Probelauf bei 0 % Leistung.

Bei der Rampe ergibt sich die Höhe unter jedem Punkt der Linie aus dem Anstieg des Streifens: Im
Abstand _d_ entlang einer Linie der Länge _L_ auf einem Streifen, der um _h_ ansteigt, liegt das
Material _h_ × _d_ / _L_ höher als am Anfang.
