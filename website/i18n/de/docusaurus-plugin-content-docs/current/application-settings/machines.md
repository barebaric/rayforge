# Maschinen

![Maschinen-Einstellungen](/screenshots/application-machines.png)

Die Maschinen-Seite in den Anwendungseinstellungen ermöglicht es Ihnen, Maschinenprofile zu verwalten. Jedes Profil enthält die gesamte Konfiguration für eine spezifische Lasermaschine.

## Maschinenprofile

Maschinenprofile speichern die vollständige Konfiguration für einen Laserschneider oder Gravierer, einschließlich:

- **Allgemeine Einstellungen**: Name, Geschwindigkeiten, Beschleunigung
- **Hardware-Einstellungen**: Arbeitsbereich-Abmessungen, Achsenkonfiguration
- **Lasereinstellungen**: Leistungsbereich, PWM-Frequenz
- **Geräteeinstellungen**: Serieller Port, Baudrate, Firmware-Typ
- **G-Code-Einstellungen**: Benutzerdefinierte G-Code-Dialekt-Optionen
- **Kameraeinstellungen**: Kamera-Kalibrierung und Ausrichtung

## Maschinen verwalten

### Eine neue Maschine hinzufügen

1. Klicken Sie auf die Schaltfläche **Neue Maschine hinzufügen**
2. Geben Sie einen beschreibenden Namen für Ihre Maschine ein
3. Konfigurieren Sie die Maschineneinstellungen (siehe [Maschinen-Setup](../machine/general) für Details)
4. Klicken Sie auf **Speichern**, um das Profil zu erstellen

### Zwischen Maschinen wechseln

Verwenden Sie das Maschinenwähler-Dropdown im Hauptfenster, um zwischen konfigurierten Maschinen zu wechseln. Alle Einstellungen, einschließlich der ausgewählten Maschine, werden zwischen Sitzungen gespeichert.

### Eine Maschine duplizieren

Um ein ähnliches Maschinenprofil zu erstellen:

1. Wählen Sie die zu duplizierende Maschine aus
2. Klicken Sie auf die Schaltfläche **Duplizieren**
3. Benennen Sie die neue Maschine um und passen Sie die Einstellungen nach Bedarf an

### Eine Maschine löschen

1. Wählen Sie die zu löschende Maschine aus
2. Klicken Sie auf die Schaltfläche **Löschen**
3. Bestätigen Sie das Löschen

:::warning
Das Löschen eines Maschinenprofils kann nicht rückgängig gemacht werden. Stellen Sie sicher, dass Sie sich wichtige Einstellungen notiert haben, bevor Sie löschen.
:::

## Verwandte Themen

- [Maschinen-Setup](../machine/general) - Detaillierte Maschinenkonfiguration
- [Ersteinrichtung](../getting-started/first-time-setup) - Anleitung zur Ersteinrichtung
