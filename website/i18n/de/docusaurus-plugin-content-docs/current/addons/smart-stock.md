# Smart Stock

Smart Stock verwendet maschinelles Sehen, um Material auf deinem Laserbett zu
erkennen und passende Bestandselemente in deinem Dokument zu erstellen. Durch
den Vergleich eines Referenzbildes des leeren Betts mit der aktuellen
Kameraansicht identifiziert das Addon die Umrisse des physischen Materials und
erzeugt korrekt positionierte Bestandselemente mit der richtigen Form und Größe.

## Voraussetzungen

Du benötigst eine konfigurierte und kalibrierte Kamera, die an deine Maschine
angeschlossen ist. Die Kamera muss mit Perspektivkorrektur eingerichtet sein,
damit das aufgenommene Bild mit dem physischen Koordinatensystem der Maschine
übereinstimmt. Außerdem benötigst du eine konfigurierte Maschine, damit das
Addon die Abmessungen des Arbeitsbereichs kennt.

## Den Erkennungsdialog öffnen

Öffne den Dialog über **Werkzeuge - Bestand über Kamera erkennen**. Das Fenster
zeigt links eine Live-Kameravorschau und rechts die Erkennungseinstellungen.

## Ein Referenzbild aufnehmen

Bevor du Material erkennst, benötigst du ein Referenzbild des leeren Laserbetts.
Ohne Material auf dem Bett klicke auf die Schaltfläche **Aufnehmen** neben
**Referenz aufnehmen**. Das Addon speichert dieses Bild und vergleicht es mit
dem Live-Kamerabild, um neue Objekte zu finden.

Referenzbilder werden pro Kamera gespeichert. Wenn du den Dialog mit derselben
Kamera erneut öffnest, wird die zuvor aufgenommene Referenz automatisch geladen
und die Erkennung wird sofort ausgeführt, falls sich bereits Material auf dem
Bett befindet.

## Bestand erkennen

Lege dein Material auf das Laserbett und klicke dann auf **Bestand erkennen**
unten im Einstellungsbereich. Das Addon vergleicht das aktuelle Kamerabild mit
dem Referenzbild und verfolgt die Umrisse aller neuen Objekte. Erkannte Formen
erscheinen in der Vorschau als magentafarbene Umrisse mit grüner Füllung.

Die Statuszeile unten im Einstellungsbereich zeigt an, wie viele Elemente
gefunden wurden. Wenn kein Bestand erkannt wird, passe die Platzierung oder
Beleuchtung an und versuche es erneut.

## Erkennungseinstellungen

**Kamera** zeigt die aktuell ausgewählte Kamera an. Klicke auf **Ändern**, um zu
einer anderen konfigurierten Kamera zu wechseln.

**Empfindlichkeit** steuert, wie viel visuelle Änderung erforderlich ist, um als
Bestand erkannt zu werden. Bei höheren Werten werden kleinere oder subtilere
Unterschiede zwischen der Referenz und dem aktuellen Bild erkannt. Bei
niedrigeren Werten werden nur große Änderungen erfasst. Wenn das Addon
vorhandenes Material nicht erkennt, erhöhe die Empfindlichkeit. Wenn es
Schatten oder Reflexionen als Bestand erkennt, verringere sie.

**Glättung** steuert, wie glatt die erkannten Umrisse sind. Höhere Werte
erzeugen rundere, einfachere Konturen, indem kleine gezackte Kanten aus dem
Kamerabild herausgefiltert werden. Niedrigere Werte bewahren mehr Details der
tatsächlichen Materialform.

## Bestandselemente erstellen

Sobald die Vorschau die erkannten Umrisse passend zu deinem Material zeigt,
klicke auf **Bestandselemente erstellen** in der Titelleiste. Das Addon fügt
für jede erkannte Form ein Bestands-Asset und ein Bestandselement zu deinem
Dokument hinzu, positioniert an den korrekten physischen Koordinaten auf der
Leinwand. Der Dialog schließt sich, nachdem die Elemente erstellt wurden.

## Verwandte Themen

- [Kamera-Einrichtung](../machine/camera) - Kamera konfigurieren und kalibrieren
- [Material-Handhabung](../features/stock-handling) - Mit Bestandselementen im Dokument arbeiten
