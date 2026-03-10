# Wartung

Die Wartungsseite in den Maschineneinstellungen hilft Ihnen, die Maschinennutzung zu verfolgen und Wartungsaufgaben zu planen.

![Wartungseinstellungen](/screenshots/machine-maintenance.png)

## Nutzungsverfolgung

Rayforge verfolgt, wie lange deine Maschine in Gebrauch ist. Diese Informationen helfen Ihnen, vorbeugende Wartung in geeigneten Intervallen zu planen.

### Gesamtstunden

Der Gesamtstundenzähler verfolgt die gesamte Zeit, die mit dem Ausführen von Jobs auf der Maschine verbracht wurde. Dieser kumulative Zähler kann nicht zurückgesetzt werden und bietet eine vollständige Historie der Maschinennutzung.

Verwende dies, um das Gesamtalter der Maschine zu verfolgen und größere Wartungsintervalle zu planen.

## Benutzerdefinierte Wartungszähler

Du kannst benutzerdefinierte Zähler erstellen, um spezifische Wartungsintervalle zu verfolgen. Jeder Zähler hat einen Namen, verfolgt Stunden und kann mit einem Benachrichtigungsschwellenwert konfiguriert werden.

### Einen Zähler erstellen

1. Klicke auf die Hinzufügen-Schaltfläche, um einen neuen Zähler zu erstellen
2. Gib einen beschreibenden Namen ein (z.B. "Laserröhre", "Riemenzug", "Spiegelreinigung")
3. Setze bei Bedarf einen Benachrichtigungsschwellenwert in Stunden

### Zähler-Funktionen

- **Benutzerdefinierte Namen**: Zähler für jede Wartungsaufgabe beschriften
- **Stundenverfolgung**: Akkumuliert automatisch Zeit während der Jobausführung
- **Benachrichtigungsschwellenwerte**: Erinnerung erhalten, wenn Wartung fällig ist
- **Zurücksetzungsmöglichkeit**: Zähler nach durchgeführter Wartung zurücksetzen

### Beispielzähler

**Laserröhre**: Verfolge CO2-Röhrenstunden, um den Austausch zu planen (typischerweise 1000-3000 Stunden). Setze eine Benachrichtigung bei 2500 Stunden, um vorauszuplanen.

**Riemenzug**: Verfolge Stunden seit dem letzten Riemenzug. Nach Durchführung der Wartung zurücksetzen.

**Spiegelreinigung**: Verfolge Nutzung seit der letzten Spiegelreinigung. Nach Reinigung zurücksetzen.

**Lager-Schmierung**: Verfolge Stunden für Lager-Wartungsintervalle.

## Zähler zurücksetzen

Nach Durchführung der Wartung kannst du den entsprechenden Zähler zurücksetzen:

1. Klicke auf die Zurücksetzen-Taste neben dem Zähler
2. Bestätige das Zurücksetzen im Dialog
3. Der Zähler kehrt zu Null zurück

:::tip Wartungsplan
Häufige Wartungsintervalle:
- **Täglich**: Linse reinigen, Spiegelausrichtung überprüfen
- **Wöchentlich**: Schienen reinigen, Riemenzug überprüfen
- **Monatlich**: Lager schmieren, elektrische Verbindungen überprüfen
- **Jährlich**: Vollständige Inspektion, verschlissene Teile ersetzen

Passe Intervalle basierend auf deinen Nutzungsmustern und Herstellerempfehlungen an.
:::

## Siehe auch

- [Lasereinstellungen](laser) - Laserkopf-Konfiguration
- [Hardware-Einstellungen](hardware) - Maschinenabmessungen
