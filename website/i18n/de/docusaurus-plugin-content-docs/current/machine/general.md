# Allgemeine Einstellungen

Die Allgemeine Seite in den Maschineneinstellungen enthält grundlegende Maschineninformationen und Geschwindigkeitseinstellungen.

![Allgemeine Einstellungen](/screenshots/machine-general.png)

## Maschinenname

Geben Sie Ihrer Maschine einen beschreibenden Namen. Dies hilft, die Maschine im Maschinenwähler-Dropdown zu identifizieren, wenn Sie mehrere Maschinen konfiguriert haben.

Beispiele:
- "Werkstatt K40"
- "Garagen-Diodenlaser"
- "Ortur LM2 Pro"

## Geschwindigkeiten & Beschleunigung

Diese Einstellungen steuern die maximalen Geschwindigkeiten und Beschleunigung für Bewegungsplanung und Zeitschätzung.

### Max. Verfahrgeschwindigkeit

Die maximale Geschwindigkeit für schnelle (nicht schneidende) Bewegungen. Dies wird verwendet, wenn der Laser aus ist und der Kopf sich an eine neue Position bewegt.

- **Typischer Bereich**: 2000-5000 mm/min
- **Zweck**: Bewegungsplanung und Zeitschätzung
- **Hinweis**: Tatsächliche Geschwindigkeit wird auch durch Ihre Firmware-Einstellungen begrenzt

### Max. Schnittgeschwindigkeit

Die maximal erlaubte Geschwindigkeit während Schneid- oder Gravur-Operationen.

- **Typischer Bereich**: 500-2000 mm/min
- **Zweck**: Begrenzt Operationsgeschwindigkeiten zur Sicherheit
- **Hinweis**: Einzelne Operationen können niedrigere Geschwindigkeiten verwenden

### Beschleunigung

Die Rate, mit der die Maschine beschleunigt und verzögert.

- **Typischer Bereich**: 500-2000 mm/s²
- **Zweck**: Zeitschätzung und Bewegungsplanung
- **Hinweis**: Muss mit den Firmware-Beschleunigungseinstellungen übereinstimmen oder niedriger sein

:::tip
Beginnen Sie mit konservativen Geschwindigkeitswerten und erhöhen Sie diese schrittweise. Beobachten Sie Ihre Maschine auf Riemenrutschen, Motorblockaden oder Verlust der Positionierungsgenauigkeit.
:::

## Siehe auch

- [Hardware-Einstellungen](hardware) - Maschinenabmessungen und Achsenkonfiguration
- [Geräteeinstellungen](device) - Verbindung und GRBL-Einstellungen
