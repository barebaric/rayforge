# Erweiterte Einstellungen

Die Erweiterte Seite in den Maschineneinstellungen enthält zusätzliche Konfigurationsoptionen für spezielle Anwendungsfälle.

![Erweiterte Einstellungen](/screenshots/machine-advanced.png)

## Verbindungsverhalten

Einstellungen, die steuern, wie Rayforge während der Verbindung mit Ihrer Maschine interagiert.

### Beim Verbinden referenzieren

Wenn aktiviert, sendet Rayforge automatisch einen Referenzierungsbefehl ($H) beim Verbinden mit der Maschine.

- **Aktivieren, wenn**: Ihre Maschine zuverlässige Endschalter hat
- **Deaktivieren, wenn**: Ihre Maschine keine Endschalter hat oder Referenzierung unzuverlässig ist

### Alarme beim Verbinden löschen

Wenn aktiviert, löscht Rayforge automatisch jeden Alarmzustand beim Verbinden.

- **Aktivieren, wenn**: Ihre Maschine häufig im Alarmzustand startet
- **Deaktivieren, wenn**: Sie Alarme manuell untersuchen möchten, bevor Sie sie löschen

## Achsen umkehren

Diese Einstellungen kehren die Richtung der Achsenbewegungen um.

### X-Achse umkehren

Kehrt die X-Achsenrichtung um. Wenn aktiviert, bewegt sich positives X nach links statt nach rechts.

### Y-Achse umkehren

Kehrt die Y-Achsenrichtung um. Wenn aktiviert, bewegt sich positives Y nach unten statt nach oben.

:::info
Achsen umkehren ist nützlich, wenn:
- Das Koordinatensystem Ihrer Maschine nicht dem erwarteten Verhalten entspricht
- Sie Ihre Motoren falsch verdrahtet haben
- Sie das Verhalten einer anderen Maschine nachbilden möchten
:::

## Siehe auch

- [Hardware-Einstellungen](hardware) - Achsenursprung-Konfiguration
- [Geräteeinstellungen](device) - GRBL-Achsenrichtungseinstellungen
