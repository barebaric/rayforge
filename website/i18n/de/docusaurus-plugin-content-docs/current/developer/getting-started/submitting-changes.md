# Änderungen einreichen

Dieser Leitfaden behandelt den Prozess für das Einreichen von Code-Verbesserungen zu Rayforge.

## Einen Feature-Branch erstellen

Erstellen Sie einen beschreibenden Branch für Ihre Änderungen:

```bash
git checkout -b feature/ihr-feature-name
# oder
git checkout -b fix/issue-nummer-beschreibung
```

## Änderungen vornehmen

- Folgen Sie dem bestehenden Code-Stil und Konventionen
- Schreiben Sie saubere, fokussierte Commits mit klaren Nachrichten
- Fügen Sie Tests für neue Funktionalität hinzu
- Aktualisieren Sie Dokumentation nach Bedarf

## Änderungen testen

Führen Sie die vollständige Test-Suite aus, um sicherzustellen, dass nichts kaputt ist:

```bash
# Alle Tests und Linting ausführen
pixi run test
pixi run lint
```

## Mit Upstream synchronisieren

Bevor Sie einen Pull-Request erstellen, synchronisieren Sie mit dem Upstream-Repository:

```bash
# Die neuesten Änderungen abrufen
git fetch upstream

# Ihren Branch auf dem neuesten Main rebasen
git rebase upstream/main
```

## Einen Pull-Request einreichen

1. Pushen Sie Ihren Branch zu Ihrem Fork:
   ```bash
   git push origin feature/ihr-feature-name
   ```

2. Erstellen Sie einen Pull-Request auf GitHub mit:
   - Einer klaren Überschrift, die die Änderung beschreibt
   - Einer detaillierten Beschreibung, was Sie geändert haben und warum
   - Referenz auf verwandte Issues
   - Screenshots, wenn die Änderung die UI betrifft

## Code-Review-Prozess

- Alle Pull-Requests erfordern Review vor dem Mergen
- Gehen Sie zeitnah auf Feedback ein und machen Sie angeforderte Änderungen
- Halten Sie die Diskussion fokussiert und konstruktiv

## Merge-Anforderungen

Pull-Requests werden gemergt, wenn sie:

- [ ] Alle automatisierten Tests bestehen
- [ ] Dem Code-Stil des Projekts folgen
- [ ] Appropriate Tests für neue Funktionalität enthalten
- [ ] Dokumentations-Updates haben, falls erforderlich
- [ ] Von mindestens einem Maintainer genehmigt sind

## Zusätzliche Richtlinien

### Commit-Nachrichten

Verwenden Sie klare, beschreibende Commit-Nachrichten:

- Beginnen Sie mit einem Großbuchstaben
- Halten Sie die erste Zeile unter 50 Zeichen
- Verwenden Sie den Imperativ ("Feature hinzufügen" nicht "Feature hinzugefügt")
- Fügen Sie bei Bedarf mehr Details im Body hinzu

### Kleine, fokussierte Änderungen

Halten Sie Pull-Requests auf ein einzelnes Feature oder einen Fix fokussiert. Große Änderungen sollten in kleinere, logische Teile zerlegt werden.

:::tip Erst besprechen
Für größere Änderungen öffnen Sie zuerst ein [Issue](https://github.com/barebaric/rayforge/issues), um Ihren Ansatz zu besprechen, bevor Sie erhebliche Zeit investieren.
:::


:::note Hilfe benötigt?
Wenn Sie unsicher über einen Teil des Beitragsprozesses sind, zögern Sie nicht, in einem Issue oder einer Diskussion um Hilfe zu fragen.
:::
