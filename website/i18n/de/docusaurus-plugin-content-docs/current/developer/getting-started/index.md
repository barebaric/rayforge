# Den Code erhalten

Dieser Leitfaden behandelt, wie Sie den Rayforge-Quellcode für die Entwicklung erhalten.

## Das Repository forken

Forken Sie das [Rayforge-Repository](https://github.com/barebaric/rayforge) auf GitHub, um Ihre eigene Kopie zu erstellen, in der Sie Änderungen vornehmen können.

## Ihren Fork klonen

```bash
git clone https://github.com/IHR_BENUTZERNAME/rayforge.git
cd rayforge
```

## Upstream-Repository hinzufügen

Fügen Sie das Original-Repository als Upstream-Remote hinzu, um Änderungen zu verfolgen:

```bash
git remote add upstream https://github.com/barebaric/rayforge.git
```

## Repository verifizieren

Prüfen Sie, dass die Remotes korrekt konfiguriert sind:

```bash
git remote -v
```

Sie sollten sowohl Ihren Fork (origin) als auch das Upstream-Repository sehen.

## Nächste Schritte

Nachdem Sie den Code haben, fahren Sie mit [Einrichtung](setup) fort, um Ihre Entwicklungsumgebung zu konfigurieren.
