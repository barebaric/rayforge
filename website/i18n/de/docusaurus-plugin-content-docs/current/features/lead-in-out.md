# Einlauf / Auslauf

Einlauf- und Auslaufbewegungen erweitern jeden Konturpfad um kurze Segmente ohne Laserleistung, bevor der Schnitt beginnt und nachdem er endet. So hat der Laserkopf Zeit, vor dem eigentlichen Schnitt eine konstante Geschwindigkeit zu erreichen, und kann nach dem Ende des Schnitts allmählich abbremsen. Das führt zu saubereren Ergebnissen an den Start- und Endpunkten jedes Schnitts.

## So funktioniert es

Wenn Einlauf/Auslauf aktiviert ist, betrachtet Rayforge die Tangentenrichtung jedes Konturpfads an seinen Start- und Endpunkten. Dann wird eine kurze gerade Bewegung ohne Laserleistung entlang dieser Tangente vor dem ersten Schneidepunkt und eine weitere nach dem letzten Schneidepunkt eingefügt. Der Laser ist während dieser zusätzlichen Segmente ausgeschaltet, sodass kein Material außerhalb des vorgesehenen Pfads abgetragen wird.

## Einstellungen

### Einlauf/Auslauf aktivieren

Schaltet die Funktion für die Operation ein oder aus. Wenn deaktiviert, beginnt und endet der Schnitt genau an den Pfadendpunkten ohne zusätzliche Anfahrts- oder Abfahrtsbewegungen.

### Automatischer Abstand

Wenn diese Option aktiviert ist, berechnet Rayforge den Einlauf- und Auslaufabstand automatisch basierend auf der Schnittgeschwindigkeit und der Beschleunigungseinstellung der Maschine. Die Formel verwendet einen Sicherheitsfaktor von zwei, um sicherzustellen, dass der Laserkopf genug Raum hat, um die volle Geschwindigkeit zu erreichen. Sobald du die Schnittgeschwindigkeit änderst oder die Beschleunigung der Maschine aktualisiert wird, wird der Abstand neu berechnet.

### Einlaufabstand

Die Länge der Anfahrtbewegung ohne Laserleistung vor Schnittbeginn in Millimetern. Der Standardwert ist 2 mm. Dieses Feld ist nur bearbeitbar, wenn der automatische Abstand deaktiviert ist.

### Auslaufabstand

Die Länge der Abfahrtsbewegung ohne Laserleistung nach Schnittende in Millimetern. Der Standardwert ist 2 mm. Dieses Feld ist nur bearbeitbar, wenn der automatische Abstand deaktiviert ist.

## Wann man Einlauf/Auslauf verwenden sollte

Einlauf/Auslauf ist besonders hilfreich, wenn du Brandspuren, Überbrennungen oder ungleichmäßige Schnittqualität an den Start- und Endpunkten deiner Konturen bemerkst. Die Anfahrt ohne Laserleistung gibt der Maschine Zeit, auf Schnittgeschwindigkeit zu beschleunigen, sodass der Laser mit voller Geschwindigkeit auf das Material trifft. Der Auslauf ohne Laserleistung ermöglicht ein sanftes Abbremsen, anstatt bei voller Leistung auf dem letzten Punkt zu verweilen.

Es ist als Nachbearbeitungsoption für Kontur-, Rahmenumriss- und Shrink-Wrap-Operationen verfügbar.

---

## Verwandte Seiten

- [Kontur-Schneiden](operations/contour) - Primäre Schneideoperation
- [Rahmenumriss](operations/frame-outline) - Rechteckige Begrenzungsschneidung
- [Shrink Wrap](operations/shrink-wrap) - Effiziente Begrenzungsschneidung
- [Halte-Laschen](holding-tabs) - Teile während des Schneidens sichern
