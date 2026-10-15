# Erweiterte Einstellungen

Die Erweiterte Seite in den Maschineneinstellungen enthält zusätzliche Konfigurationsoptionen für
spezielle Anwendungsfälle.

![Erweiterte Einstellungen](/screenshots/machine-settings-advanced.webp)

## Verbindungsverhalten

Einstellungen, die steuern, wie Rayforge während der Verbindung mit deiner Maschine interagiert.

### Beim Verbinden referenzieren

Wenn aktiviert, sendet Rayforge automatisch einen Referenzierungsbefehl ($H) beim Verbinden mit der
Maschine.

- **Aktivieren, wenn**: deine Maschine zuverlässige Endschalter hat
- **Deaktivieren, wenn**: deine Maschine keine Endschalter hat oder Referenzierung unzuverlässig ist

### Alarme beim Verbinden löschen

Wenn aktiviert, löscht Rayforge automatisch jeden Alarmzustand beim Verbinden.

- **Aktivieren, wenn**: deine Maschine häufig im Alarmzustand startet
- **Deaktivieren, wenn**: du Alarme manuell untersuchen möchtest, bevor du sie löschst

### Einzelachsen-Referenzierung erlauben

Wenn aktiviert, kannst du einzelne Achsen unabhängig referenzieren (X, Y oder Z), anstatt alle
Achsen gleichzeitig zu referenzieren. Dies ist nützlich für Maschinen, bei denen eine Achse bereits
korrekt positioniert ist.

### Z-Referenzfahrt

Legt fest, wie die Z-Achse beim Referenzieren ihre Referenzposition findet. Diese Einstellung
erscheint nur, wenn der Treiber der Maschine mehr als eine Methode unterstützt.

- **Referenzschalter**: Z fährt wie die X- und Y-Achse zu ihrem Referenzschalter
- **Taster**: Z findet die Werkstückoberfläche mit dem Autofokus-Taster der Steuerung und fährt dann
  auf den eingestellten Fokusabstand

## Bogen- und Kurveneinstellungen

Einstellungen zur Steuerung, wie gekrümmte Pfade in G-Code-Bewegungen umgewandelt werden.

### Bögen unterstützen

Wenn aktiviert, generiert Rayforge Bogenbefehle (G2/G3) für gekrümmte Pfade, anstatt sie in viele
kleine lineare Bewegungen aufzubrechen. Dies erzeugt kompakteren G-Code und flüssigere Bewegungen
auf den meisten Controllern.

Wenn deaktiviert, werden alle Kurven in Liniensegmente (G1-Befehle) umgewandelt, was maximale
Kompatibilität mit Controllern bietet, die keine Bögen unterstützen.

### Bézier-Kurven unterstützen

Wenn aktiviert, generiert Rayforge native kubische Bézier-Befehle (wie den G5-Befehl unter LinuxCNC)
für gekrümmte Pfade. Dies erzeugt sehr flüssige Bewegungen und kompakten G-Code auf Controllern, die
dies unterstützen. Du solltest diese Einstellung deaktivieren, wenn die Firmware deiner Maschine
keine Bézier-Befehle versteht – in diesem Fall werden die Kurven stattdessen in Liniensegmente
zerlegt.

### Bogen- und Kurventoleranz

Diese Einstellung steuert die maximal erlaubte Abweichung beim Anpassen von Bögen und Kurven an
gekrümmte Pfade, angegeben in Millimetern. Ein kleinerer Wert erzeugt genauere Pfade, kann aber mehr
Befehle erfordern. Ein größerer Wert erlaubt mehr Abweichung, generiert aber weniger Befehle.

Typische Werte reichen von 0,01mm für Präzisionsarbeit bis 0,1mm für schnellere Verarbeitung.

## Bidirektionaler Scan-Offset nach Geschwindigkeit {#bidirectional-scan-offset-by-speed}

Die Tabelle für den bidirektionalen Scan-Offset nach Geschwindigkeit korrigiert die horizontale
Fehlausrichtung zwischen Links-nach-rechts- und Rechts-nach-links-Rasterdurchgängen. Auf vielen
Maschinen ist diese Versetzung nicht konstant: Je schneller sich der Laserkopf bewegt, desto weiter
driften die Rechts-nach-links-Durchgänge aus der Ausrichtung, was als Bänderung sichtbar wird, die
bei höheren Gravurgeschwindigkeiten schlimmer wird.

![Bidirektionaler Scan-Offset nach Geschwindigkeit](/screenshots/machine-settings-advanced-bidir-offset-table.webp)

Jede Zeile der Tabelle verknüpft eine Gravurgeschwindigkeit mit dem Offset in Millimetern, den
Rechts-nach-links-Durchgänge bei dieser Geschwindigkeit benötigen:

- Mit **Zeile hinzufügen** fügst du eine Zeile hinzu, mit dem Papierkorb-Symbol einer Zeile
  entfernst du sie
- Positive Werte verschieben Rechts-nach-links-Durchgänge nach rechts, negative nach links
- Zeilen werden nach Geschwindigkeit sortiert gehalten; haben zwei Zeilen dieselbe Geschwindigkeit,
  zählt die letzte

Läuft ein Gravurschritt, schlägt Rayforge die Geschwindigkeit des Schritts in dieser Tabelle nach
und interpoliert den Offset linear zwischen den benachbarten Zeilen. Geschwindigkeiten unterhalb der
ersten oder oberhalb der letzten Zeile verwenden den Offset der jeweils nächsten Zeile, und eine
leere Tabelle bedeutet, dass jede Geschwindigkeit einen Offset von 0 verwendet.

**Ohne Korrektur** – die Fehlausrichtung wächst mit der Gravurgeschwindigkeit:

![Fehlausgerichtete Rasterdurchgänge bei 1000, 4000 und 8000 mm/min](/images/bidir-scan-offset-uncorrected.webp)

**Mit angewandter Offset-Tabelle:**

![Ausgerichtete Rasterdurchgänge bei jeder Geschwindigkeit](/images/bidir-scan-offset-table-applied.webp)

Die Tabelle gilt nur für Gravurschritte, deren eigener **Bidirektionaler Scan-Offset** auf 0 steht;
ein nicht nuller Offset am Schritt hat immer Vorrang. Unter
[Bidirektionaler Scan-Offset](../features/operations/engrave.md#bidirectional-scan-offset) erfährst
du, wie die Fehlausrichtung aussieht und wie man sie kalibriert.

## Siehe auch

- [Hardware-Einstellungen](hardware) - Achsenursprung und Umkehreinstellungen
- [Geräteeinstellungen](device) - GRBL-spezifische Einstellungen
