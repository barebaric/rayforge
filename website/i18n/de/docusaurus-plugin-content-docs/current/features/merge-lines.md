# Linien zusammenführen

Wenn du ein Design importierst, das überlappende Pfade enthält, kann es passieren,
dass der Laser dieselbe Linie mehrmals schneidet. Das kostet Zeit, kann zu
übermäßiger Verkohlung führen und den Schnittspalt breiter machen als beabsichtigt.

Der **Linien zusammenführen**-Postprozessor erkennt überlappende und deckungsgleiche
Pfadsegmente und führt sie zu einem einzigen Durchgang zusammen. Der Laser folgt
jeder eindeutigen Linie nur einmal.

## Wann du es verwenden solltest

Dies kommt am häufigsten vor, wenn:

- Du ein SVG oder DXF importierst, bei dem Formen Kanten teilen (z. B. ein
  Rastermuster oder eine Parkettierung)
- Du mehrere Werkstücke kombinierst, deren Umrisse sich überlappen
- Deine Design-Software doppelte Pfade exportiert

## Wann du es nicht verwenden solltest

Wenn überlappende Schnitte beabsichtigt sind – zum Beispiel, wenn du mehrere
Durchgänge über dieselbe Linie machst, um dickeres Material zu durchschneiden –
lasse Linien zusammenführen deaktiviert. In diesem Fall möchtest du vielleicht die
[Mehrfach-Durchgang](multi-pass)-Funktion verwenden, die dir explizite Kontrolle
über die Anzahl der Durchgänge gibt.

## Verwandte Seiten

- [Pfadoptimierung](path-optimization) - Reduziert unnötige Leerfahrten
- [Mehrfach-Durchgang](multi-pass) - Beabsichtigte mehrfache Durchgänge über denselben Pfad
- [Konturschnitt](operations/contour) - Die Hauptschneideoperation
