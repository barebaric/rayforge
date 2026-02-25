# Maschinen

![Maschinen-Einstellungen](/screenshots/application-machines.png)

Die Maschinen-Seite in den Anwendungseinstellungen ermöglicht es dir, Maschinenprofile zu verwalten. Jedes Profil enthält die gesamte Konfiguration für eine spezifische Lasermaschine.

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

1. Klicke auf die Schaltfläche **Neue Maschine hinzufügen**
2. Gib einen beschreibenden Namen für deine Maschine ein
3. Konfiguriere die Maschineneinstellungen (siehe [Maschinen-Setup](../machine/general) für Details)
4. Klicke auf **Speichern**, um das Profil zu erstellen

### Zwischen Maschinen wechseln

Verwende das Maschinenwähler-Dropdown im Hauptfenster, um zwischen konfigurierten Maschinen zu wechseln. Alle Einstellungen, einschließlich der ausgewählten Maschine, werden zwischen Sitzungen gespeichert.

### Eine Maschine duplizieren

Um ein ähnliches Maschinenprofil zu erstellen:

1. Wähle die zu duplizierende Maschine aus
2. Klicke auf die Schaltfläche **Duplizieren**
3. Benennen Sie die neue Maschine um und passen Sie die Einstellungen nach Bedarf an

### Eine Maschine löschen

1. Wähle die zu löschende Maschine aus
2. Klicke auf die Schaltfläche **Löschen**
3. Bestätigen Sie das Löschen

:::warning
Das Löschen eines Maschinenprofils kann nicht rückgängig gemacht werden. Stelle sicher, dass du sich wichtige Einstellungen notiert haben, bevor Sie löschen.
:::

## Verwandte Themen

- [Maschinen-Setup](../machine/general) - Detaillierte Maschinenkonfiguration
- [Ersteinrichtung](../getting-started/first-time-setup) - Anleitung zur Ersteinrichtung
