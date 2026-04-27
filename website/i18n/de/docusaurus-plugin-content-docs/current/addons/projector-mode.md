# Projektormodus

Der Projektormodus zeigt deinen Schnittbereich in einem separaten Fenster an,
das für einen externen Projektor oder Zweitmonitor gedacht ist. So kannst du
sehen, wo der Laser schneidet, indem die Werkzeugpfade direkt auf dein
Material projiziert werden, was die Ausrichtung erleichtert.

Das Projektorfenster zeigt deine Werkstücke in leuchtendem Grün auf schwarzem
Hintergrund an. Es stellt den Achsenbereich-Rahmen der Maschine und den
Arbeitsursprung dar, sodass du den gesamten Schnittbereich und die Position
des Ursprungspunkts sehen kannst. Die Ansicht wird in Echtzeit aktualisiert,
wenn du Werkstücke auf der Hauptleinwand verschiebst oder bearbeitest.

## Projektorfenster öffnen

Öffne das Projektorfenster über **Ansicht - Projektor-Dialog anzeigen**. Das
Fenster öffnet sich als separates, unabhängiges Fenster, das du auf jeden
angeschlossenen Bildschirm ziehen kannst.

Ein Schalter steuert das Projektorfenster — derselbe Menüeintrag schließt es,
und das Drücken von Escape, während das Projektorfenster fokussiert ist,
schließt es ebenfalls.

## Vollbildmodus

Klicke auf die Schaltfläche **Vollbild** in der Titelleiste des
Projektorfensters, um in den Vollbildmodus zu wechseln. Dies blendet die
Fensterdekorationen aus und füllt den gesamten Bildschirm. Klicke auf
**Vollbild beenden** (dieselbe Schaltfläche), um in den Fenstermodus
zurückzukehren.

Vollbild ist der vorgesehene Modus bei der Projektion auf Material, da er
störenden Fenster-Schmuck entfernt und die gesamte Bildschirmfläche nutzt.

## Deckkraft

Die Deckkraft-Schaltfläche in der Titelleiste durchläuft vier Stufen: 100%,
80%, 60% und 40%. Eine niedrigere Deckkraft macht das Projektorfenster
halbtransparent, was auf einem Desktop-Monitor nützlich sein kann, um dahinter
liegende Fenster zu sehen. Jeder Klick wechselt zur nächsten Deckkraftstufe
und kehrt zum Anfang zurück.

![Projektormodus](/screenshots/addon-projector-mode.png)

## Was der Projektor anzeigt

Die Projektoranzeige rendert eine vereinfachte Ansicht deines Dokuments.
Werkstücke erscheinen als leuchtend grüne Umrisse, die die berechneten
Werkzeugpfade zeigen — dieselben Pfade, die an den Laser gesendet werden.
Die Basisbilder deiner Werkstücke werden nicht angezeigt, sodass die Anzeige
auf die Schnittwege konzentriert bleibt.

Der Maschinenbereich-Rahmen erscheint als Rand, der den gesamten Verfahrweg
der Maschinenachsen darstellt. Das Arbeitsursprung-Fadenkreuz zeigt, wo sich
der Ursprung des Koordinatensystems innerhalb dieses Bereichs befindet. Beide
aktualisieren sich automatisch, wenn du den Arbeitskoordinatensystem-Versatz
an deiner Maschine änderst.

## Verwandte Themen

- [Koordinatensysteme](../general-info/coordinate-systems) - Maschinenkoordinaten und Arbeitsversätze verstehen
- [Werkstückpositionierung](../features/workpiece-positioning) - Werkstücke auf der Leinwand positionieren
