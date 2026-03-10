# Erweiterte Einstellungen

Die Erweiterte Seite in den Maschineneinstellungen enthält zusätzliche Konfigurationsoptionen für spezielle Anwendungsfälle.

![Erweiterte Einstellungen](/screenshots/machine-advanced.png)

## Verbindungsverhalten

Einstellungen, die steuern, wie Rayforge während der Verbindung mit deiner Maschine interagiert.

### Beim Verbinden referenzieren

Wenn aktiviert, sendet Rayforge automatisch einen Referenzierungsbefehl ($H) beim Verbinden mit der Maschine.

- **Aktivieren, wenn**: deine Maschine zuverlässige Endschalter hat
- **Deaktivieren, wenn**: deine Maschine keine Endschalter hat oder Referenzierung unzuverlässig ist

### Alarme beim Verbinden löschen

Wenn aktiviert, löscht Rayforge automatisch jeden Alarmzustand beim Verbinden.

- **Aktivieren, wenn**: deine Maschine häufig im Alarmzustand startet
- **Deaktivieren, wenn**: du Alarme manuell untersuchen möchtest, bevor du sie löschst

### Einzelachsen-Referenzierung erlauben

Wenn aktiviert, kannst du einzelne Achsen unabhängig referenzieren (X, Y oder Z), anstatt alle Achsen gleichzeitig zu referenzieren. Dies ist nützlich für Maschinen, bei denen eine Achse bereits korrekt positioniert ist.

## Bogeneinstellungen

Einstellungen zur Steuerung, wie gekrümmte Pfade in G-Code-Bewegungen umgewandelt werden.

### Bögen unterstützen

Wenn aktiviert, generiert Rayforge Bogenbefehle (G2/G3) für gekrümmte Pfade, anstatt sie in viele kleine lineare Bewegungen aufzubrechen. Dies erzeugt kompakteren G-Code und flüssigere Bewegungen auf den meisten Controllern.

Wenn deaktiviert, werden alle Kurven in Liniensegmente (G1-Befehle) umgewandelt, was maximale Kompatibilität mit Controllern bietet, die keine Bögen unterstützen.

### Bogentoleranz

Diese Einstellung steuert die maximal erlaubte Abweichung beim Anpassen von Bögen an gekrümmte Pfade, angegeben in Millimetern. Ein kleinerer Wert erzeugt genauere Bögen, kann aber mehr Bogenbefehle erfordern. Ein größerer Wert erlaubt mehr Abweichung, generiert aber weniger Befehle.

Typische Werte reichen von 0,01mm für Präzisionsarbeit bis 0,1mm für schnellere Verarbeitung.

## Siehe auch

- [Hardware-Einstellungen](hardware) - Achsenursprung und Umkehreinstellungen
- [Geräteeinstellungen](device) - GRBL-spezifische Einstellungen
