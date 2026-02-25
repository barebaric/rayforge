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

## Achsen umkehren

Diese Einstellungen kehren die Richtung der Achsenbewegungen um.

### X-Achse umkehren

Kehrt die X-Achsenrichtung um. Wenn aktiviert, bewegt sich positives X nach links statt nach rechts.

### Y-Achse umkehren

Kehrt die Y-Achsenrichtung um. Wenn aktiviert, bewegt sich positives Y nach unten statt nach oben.

:::info
Achsen umkehren ist nützlich, wenn:
- Das Koordinatensystem deiner Maschine nicht dem erwarteten Verhalten entspricht
- Du deine Motoren falsch verdrahtet hast
- Du das Verhalten einer anderen Maschine nachbilden möchtest
:::

## Siehe auch

- [Hardware-Einstellungen](hardware) - Achsenursprung-Konfiguration
- [Geräteeinstellungen](device) - GRBL-Achsenrichtungseinstellungen
