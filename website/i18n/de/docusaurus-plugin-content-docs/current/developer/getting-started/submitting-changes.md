# Änderungen einreichen

Dieser Leitfaden behandelt den Prozess für das Einreichen von Code-Verbesserungen zu Rayforge.

## Einen Feature-Branch erstellen

Erstelle einen beschreibenden Branch für deine Änderungen:

```bash
git checkout -b feature/ihr-feature-name
# oder
git checkout -b fix/issue-nummer-beschreibung
```

## Änderungen vornehmen

- Folge dem bestehenden Code-Stil und Konventionen
- Schreibe saubere, fokussierte Commits mit klaren Nachrichten
- Füge Tests für neue Funktionalität hinzu
- Aktualisiere Dokumentation nach Bedarf

## Änderungen testen

Führe die vollständige Test-Suite aus, um sicherzustellen, dass nichts kaputt ist:

```bash
# Alle Tests und Linting ausführen
pixi run test
pixi run lint
```

## Mit Upstream synchronisieren

Bevor du einen Pull-Request erstellst, synchronisiere mit dem Upstream-Repository:

```bash
# Die neuesten Änderungen abrufen
git fetch upstream

# Deinen Branch auf dem neuesten Main rebasen
git rebase upstream/main
```

## Einen Pull-Request einreichen

1. Pushe deinen Branch zu deinem Fork:
   ```bash
   git push origin feature/ihr-feature-name
   ```

2. Erstelle einen Pull-Request auf GitHub mit:
   - Einer klaren Überschrift, die die Änderung beschreibt
   - Einer detaillierten Beschreibung, was du geändert hast und warum
   - Referenz auf verwandte Issues
   - Screenshots, wenn die Änderung die UI betrifft

## Code-Review-Prozess

- Alle Pull-Requests erfordern Review vor dem Mergen
- Gehe zeitnah auf Feedback ein und mache angeforderte Änderungen
- Halte die Diskussion fokussiert und konstruktiv

## Merge-Anforderungen

Pull-Requests werden gemergt, wenn sie:

- [ ] Alle automatisierten Tests bestehen
- [ ] Dem Code-Stil des Projekts folgen
- [ ] Appropriate Tests für neue Funktionalität enthalten
- [ ] Dokumentations-Updates haben, falls erforderlich
- [ ] Von mindestens einem Maintainer genehmigt sind

## Zusätzliche Richtlinien

### Commit-Nachrichten

Verwende klare, beschreibende Commit-Nachrichten:

- Beginne mit einem Großbuchstaben
- Halte die erste Zeile unter 50 Zeichen
- Verwende den Imperativ ("Feature hinzufügen" nicht "Feature hinzugefügt")
- Füge bei Bedarf mehr Details im Body hinzu

### Kleine, fokussierte Änderungen

Halte Pull-Requests auf ein einzelnes Feature oder einen Fix fokussiert. Große Änderungen sollten in kleinere, logische Teile zerlegt werden.

:::tip Erst besprechen
Für größere Änderungen öffne zuerst ein [Issue](https://github.com/barebaric/rayforge/issues), um deinen Ansatz zu besprechen, bevor du erhebliche Zeit investierst.
:::


:::note Hilfe benötigt?
Wenn du unsicher über einen Teil des Beitragsprozesses bist, zögere nicht, in einem Issue oder einer Diskussion um Hilfe zu fragen.
:::
