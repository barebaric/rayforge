# Hardware-Einstellungen

Die Hardware-Seite in den Maschineneinstellungen konfiguriert die physischen Abmessungen, das Koordinatensystem und die Bewegungsgrenzen deiner Maschine.

![Hardware-Einstellungen](/screenshots/machine-hardware.png)

## Achsen

Konfiguriere die Achsenbereiche und das Koordinatensystem für deine Maschine.

### X/Y-Bereich

Der vollständige Verfahrweg jeder Achse in Maschineneinheiten.

- Miss den tatsächlichen Schneidebereich, nicht das Maschinengehäuse
- Berücksichtige alle Hindernisse oder Grenzen
- Beispiel: 400 für einen typischen K40-Laser

### Koordinatenursprung

Wähle, wo sich der Koordinatenursprung (0,0) deiner Maschine befindet. Dies bestimmt, wie Koordinaten interpretiert werden.

- **Unten links**: Am häufigsten bei GRBL-Geräten. X nimmt nach rechts zu, Y nimmt nach oben zu.
- **Oben links**: Häufig bei einigen CNC-artigen Maschinen. X nimmt nach rechts zu, Y nimmt nach unten zu.
- **Oben rechts**: X nimmt nach links zu, Y nimmt nach unten zu.
- **Unten rechts**: X nimmt nach links zu, Y nimmt nach oben zu.

#### Deinen Ursprung finden

1. Referenziere deine Maschine mit der Home-Taste
2. Beobachte, wohin sich der Laserkopf bewegt
3. Diese Position ist dein (0,0)-Ursprung

:::info
Die Einstellung des Koordinatenursprungs beeinflusst, wie G-Code generiert wird. Stelle sicher, dass sie mit der Homing-Konfiguration deiner Firmware übereinstimmt.
:::

### Achsenrichtung

Kehre die Richtung einer beliebigen Achse um, falls erforderlich:

- **X-Achsenrichtung umkehren**: Macht X-Koordinatenwerte negativ
- **Y-Achsenrichtung umkehren**: Macht Y-Koordinatenwerte negativ
- **Z-Achsenrichtung umkehren**: Aktivieren, wenn ein positiver Z-Befehl (z.B. G0 Z10) den Kopf nach unten bewegt

## Arbeitsbereich

Ränder definieren den unbenutzbaren Raum um die Kanten deiner Achsenbereiche. Dies ist nützlich, wenn deine Maschine Bereiche hat, die der Laser nicht erreichen kann (z.B. aufgrund des Laserkopf-Aufbaus, Kabelketten oder anderer Hindernisse).

- **Linker/Oberer/Rechter/Unterer Rand**: Der unbenutzbare Raum von jeder Kante in Maschineneinheiten

Wenn Ränder festgelegt sind, wird der Arbeitsbereich (nutzbarer Raum) als die Achsenbereiche abzüglich der Ränder berechnet.

## Software-Limits

Konfigurierbare Sicherheitsgrenzen für das Verfahren des Maschinenkopfes. Wenn aktiviert, verhindern die Jog-Steuerungen Bewegungen außerhalb dieser Grenzen.

- **Benutzerdefinierte Software-Limits aktivieren**: Umschalten, um benutzerdefinierte Limits statt der Arbeitsflächen-Grenzen zu verwenden
- **X/Y Min**: Minimale Koordinate für jede Achse
- **X/Y Max**: Maximale Koordinate für jede Achse

Software-Limits werden automatisch eingeschränkt, um innerhalb der Achsenbereiche zu bleiben (0 bis Bereichswert).

:::tip
Verwende Software-Limits, um Bereiche deiner Arbeitsfläche zu schützen, die während des Joggings niemals erreicht werden sollten, wie Bereiche mit Vorrichtungen oder empfindlicher Ausrüstung.
:::

## Siehe auch

- [Allgemeine Einstellungen](general) - Maschinenname und Geschwindigkeitseinstellungen
- [Geräteeinstellungen](device) - GRBL-Homing und Achseneinstellungen
