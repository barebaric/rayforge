# Lasereinstellungen

Die Laser-Seite in den Maschineneinstellungen konfiguriert Ihre Laserkopf/Köpfe und deren Eigenschaften.

![Lasereinstellungen](/screenshots/machine-laser.png)

## Laserköpfe

Rayforge unterstützt Maschinen mit mehreren Laserköpfen. Jeder Laserkopf hat seine eigene Konfiguration.

### Einen Laserkopf hinzufügen

Klicken Sie auf die Schaltfläche **Laser hinzufügen**, um eine neue Laserkopf-Konfiguration zu erstellen.

### Laserkopf-Eigenschaften

Jeder Laserkopf hat die folgenden Einstellungen:

#### Name

Ein beschreibender Name für diesen Laserkopf.

Beispiele:
- "10W Diode"
- "CO2-Röhre"
- "Infrarotlaser"

#### Werkzeugnummer

Der Werkzeugindex für diesen Laserkopf. Im G-Code mit dem T-Befehl verwendet.

- Ein-Kopf-Maschinen: 0 verwenden
- Multi-Kopf-Maschinen: Eindeutige Nummern zuweisen (0, 1, 2, usw.)

#### Maximale Leistung

Der maximale Leistungswert für Ihren Laser.

- **GRBL typisch**: 1000 (S0-S1000 Bereich)
- **Einige Controller**: 255 (S0-S255 Bereich)
- **Prozentmodus**: 100 (S0-S100 Bereich)

Dieser Wert sollte mit der $30-Einstellung Ihrer Firmware übereinstimmen.

#### Rahmen-Leistung

Der Leistungswert, der für Rahmen-Operationen verwendet wird (Umreißen ohne Schneiden).

- Auf 0 setzen, um Rahmen zu deaktivieren
- Typische Werte: 5-20 (gerade sichtbar, markiert das Material nicht)
- Passen Sie basierend auf Ihrem Laser und Material an

#### Punktgröße

Die physische Größe Ihres fokussierten Laserstrahls in Millimetern.

- Geben Sie sowohl X- als auch Y-Abmessungen ein
- Die meisten Laser haben einen runden Punkt (z.B. 0.1 x 0.1)
- Beeinflusst Gravurqualitäts-Berechnungen

:::tip Punktgröße messen
Um Ihre Punktgröße zu messen:
1. Feuern Sie einen kurzen Impuls bei niedriger Leistung auf ein Testmaterial
2. Messen Sie die resultierende Markierung mit einer Schieblehre
3. Verwenden Sie den Durchschnitt mehrerer Messungen
:::

## Siehe auch

- [Geräteeinstellungen](device) - GRBL Lasermodus-Einstellungen
