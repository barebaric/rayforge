# Print & Cut

Richte Laserschnitte auf vorgedrucktem Material aus, indem du Referenzpunkte
auf deinem Design registrierst und sie deren physischen Positionen auf dem
Material zuordnest. Dies ist nützlich zum Schneiden von Aufklebern, Etiketten
oder allem, das mit einem vorhandenen Druck übereinstimmen muss.

## Voraussetzungen

Das Addon erfordert eine konfigurierte Maschine. Deine Maschine muss für den
Jogging-Schritt verbunden sein. Du benötigst außerdem ein auf der Leinwand
ausgewähltes Werkstück oder eine Gruppe.

## Den Assistenten öffnen

Wähle ein einzelnes Werkstück oder eine Gruppe auf der Leinwand aus und öffne
dann **Werkzeuge - Auf physische Position ausrichten**. Der Assistent öffnet
sich als dreischrittiger Dialog mit einer Vorschau deines Werkstücks links und
den Steuerungen rechts.

## Schritt 1: Designpunkte auswählen

![Designpunkte auswählen](/screenshots/addon-print-and-cut-pick.png)

Das linke Panel zeigt ein gerendertes Bild deines ausgewählten Werkstücks.
Klicke direkt auf das gerenderte Bild, um den ersten Ausrichtungspunkt zu
platzieren (grün markiert), dann klicke erneut, um den zweiten Punkt zu
platzieren (blau markiert). Eine gestrichelte Linie verbindet die beiden
Punkte.

Wähle zwei Punkte, die identifizierbaren Merkmalen auf deinem physischen
Material entsprechen — zum Beispiel gedruckte Registrierungsmarken oder
deutliche Ecken. Die Punkte müssen weit genug voneinander entfernt sein für
eine genaue Ausrichtung. Du kannst entweder Punkt nach dem Platzieren ziehen,
um die Position zu optimieren.

Verwende das Scrollrad, um in der Vorschau heranzuzoomen, und Mittelklick-Ziehen,
um sich zu bewegen. Die Schaltfläche **Zurücksetzen** unten löscht beide Punkte
und lässt dich von vorn beginnen.

Sobald beide Punkte platziert sind, klicke auf **Weiter**, um fortzufahren.

## Schritt 2: Physische Positionen aufzeichnen

![Physische Positionen aufzeichnen](/screenshots/addon-print-and-cut-jog.png)

Auf dieser Seite bewegst du den Laser zu den physischen Positionen, die den
beiden ausgewählten Designpunkten entsprechen. Das rechte Panel zeigt ein
Richtungs-Pad zum Joggen und eine Abstandssteuerung, die festlegt, wie weit
der Laser pro Schritt bewegt wird.

Bewege den Laser zur physischen Position, die deinem ersten Designpunkt
entspricht, und klicke dann auf **Aufzeichnen** neben Position 1. Die
aufgezeichneten Koordinaten erscheinen in der Zeile. Wiederhole den Vorgang
für Position 2. Du kannst eine aufgezeichnete Position jederzeit erneut
aufrufen, indem du auf die Schaltfläche **Gehe zu** daneben klickst.

Der Schalter **Fokuslimiere Laser** schaltet den Laser mit der konfigurierten
Fokusleistung ein, was einen sichtbaren Punkt auf dem Material erzeugt, um
Positionen präzise zu lokalisieren. Dieser Schalter erfordert einen
Fokusleistungswert größer Null in deinen Lasereinstellungen.

Die aktuelle Laserposition wird unten im Panel angezeigt. Wenn beide
Positionen aufgezeichnet sind, klicke auf **Weiter**, um fortzufahren.

## Schritt 3: Transformation überprüfen und anwenden

![Transformation überprüfen und anwenden](/screenshots/addon-print-and-cut-apply.png)

Die letzte Seite zeigt die berechnete Ausrichtung als Versatz und
Drehwinkel an. Diese Werte werden aus dem Unterschied zwischen deinen
Designpunkten und den aufgezeichneten physischen Positionen abgeleitet.

Standardmäßig ist die Skalierung auf 1,0 gesperrt. Wenn dein physisches
Material in der Größe vom Design abweicht — zum Beispiel durch
Druckerskalierung — aktiviere den Schalter **Skalierung zulassen**. Der
Skalierungsfaktor wird dann aus dem Verhältnis der physischen Distanz zur
Design-Distanz zwischen deinen beiden Punkten berechnet. Ein Hinweis
erscheint, wenn die Skalierung gesperrt ist, die Distanzen aber nicht
übereinstimmen, was darauf hinweist, dass der zweite Punkt möglicherweise
nicht genau ausgerichtet ist.

Klicke auf **Anwenden**, um das Werkstück auf der Leinwand zu verschieben
und zu drehen, sodass es den physischen Positionen entspricht. Die
Transformation wird als rücknehmbare Aktion angewendet.

## Verwandte Themen

- [Werkstückpositionierung](../features/workpiece-positioning) - Werkstücke manuell positionieren und transformieren
- [Lasereinstellungen](../machine/laser) - Fokusleistung für den Laser konfigurieren
