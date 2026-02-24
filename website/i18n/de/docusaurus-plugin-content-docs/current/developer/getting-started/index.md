# Den Code erhalten

Dieser Leitfaden behandelt, wie du den Rayforge-Quellcode für die Entwicklung erhältst.

## Das Repository forken

Forke das [Rayforge-Repository](https://github.com/barebaric/rayforge) auf GitHub, um deine eigene Kopie zu erstellen, in der du Änderungen vornehmen kannst.

## Deinen Fork klonen

```bash
git clone https://github.com/IHR_BENUTZERNAME/rayforge.git
cd rayforge
```

## Upstream-Repository hinzufügen

Füge das Original-Repository als Upstream-Remote hinzu, um Änderungen zu verfolgen:

```bash
git remote add upstream https://github.com/barebaric/rayforge.git
```

## Repository verifizieren

Prüfe, dass die Remotes korrekt konfiguriert sind:

```bash
git remote -v
```

Du solltest sowohl deinen Fork (origin) als auch das Upstream-Repository sehen.

## Nächste Schritte

Nachdem du den Code hast, fahre mit [Einrichtung](setup) fort, um deine Entwicklungsumgebung zu konfigurieren.
