# Schnellstart-Anleitung

Nachdem Rayforge installiert und deine Maschine konfiguriert ist, lass uns deinen ersten Laserjob ausführen! Diese Anleitung führt dich durch den Import eines Designs, die Konfiguration von Operationen und das Senden von G-Code an deine Maschine.

## Schritt 1: Ein Design importieren

Rayforge unterstützt verschiedene Dateiformate, darunter SVG, DXF, PDF und Rasterbilder (JPEG, PNG, BMP).

1. **Klicke** auf **Datei → Öffnen** oder drücke <kbd>ctrl+o</kbd>
2. Navigiere zu deiner Designdatei und wähle sie aus
3. Das Design erscheint auf der Arbeitsfläche

![Arbeitsfläche mit importiertem Design](/screenshots/main-standard.png)

:::tip Noch kein Design?
Du kannst einfache Formen mit den Arbeitsflächen-Werkzeugen erstellen oder kostenlose SVG-Dateien von Seiten wie [Flaticon](https://www.flaticon.com/) oder [SVG Repo](https://www.svgrepo.com/) herunterladen.
:::


## Schritt 2: Dein Design positionieren

Verwende die Arbeitsflächen-Werkzeuge, um dein Design zu positionieren und anzupassen:

- **Verschieben**: Mittelklick und ziehen, oder halte <kbd>Leertaste</kbd> und ziehe
- **Zoom**: Mausrad, oder <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Bewegen**: Klicke dein Design an und ziehe es
- **Drehen**: Wähle das Design aus und verwende die Drehgriffe
- **Skalieren**: Wähle das Design aus und ziehe die Eckgriffe

## Schritt 3: Eine Operation zuweisen

Operationen definieren, wie Rayforge dein Design verarbeitet. Häufige Operationen sind:

- **Kontur**: Entlang der Umrisse von Formen schneiden
- **Rastergravur**: Formen mit Hin-und-Her-Linien füllen (für Gravuren)
- **Tiefengravur**: 3D-Tiefeneffekte aus Bildern erstellen

### Eine Operation hinzufügen

1. Wähle dein Design auf der Arbeitsfläche aus
2. Klicke auf **Operationen → Operation hinzufügen** oder drücke <kbd>ctrl+shift+a</kbd>
3. Wähle den Operationstyp (z.B. "Kontur" zum Schneiden)
4. Konfiguriere die Operationseinstellungen:
   - **Leistung**: Laserleistung in Prozent (niedrig beginnen und testen!)
   - **Geschwindigkeit**: Bewegungsgeschwindigkeit in mm/min
   - **Durchgänge**: Anzahl der Wiederholungen der Operation (nützlich zum Schneiden dicker Materialien)

![Operationseinstellungen](/screenshots/step-settings-contour-general.png)

:::warning Mit niedriger Leistung beginnen
Wenn du mit neuen Materialien arbeitest, beginne immer mit niedrigeren Leistungseinstellungen und führe Testschnitte durch. Erhöhe die Leistung schrittweise, bis du das gewünschte Ergebnis erzielst. Verwende die Funktion [Materialtest-Raster](../features/operations/material-test-grid), um systematisch optimale Einstellungen zu finden.
:::


## Schritt 4: Vorschau

Bevor du an deine Maschine sendest, zeige den Werkzeugweg in 3D an:

1. Klicke auf **Ansicht → 3D-Vorschau** oder drücke <kbd>ctrl+3</kbd>
2. Das 3D-Vorschau-Fenster zeigt den vollständigen Werkzeugweg
3. Verwende deine Maus, um die Vorschau zu drehen und zu zoomen
4. Überprüfe, ob der Pfad korrekt aussieht

![3D-Vorschau](/screenshots/main-3d.png)

:::tip Fehler frühzeitig erkennen
Die 3D-Vorschau hilft dir, Probleme zu erkennen wie:

- Fehlende Pfade
- Falsche Reihenfolge
- Auf falsche Objekte angewendete Operationen
- Pfade, die deinen Arbeitsbereich überschreiten
:::


## Schritt 5: An die Maschine senden

:::danger Sicherheit geht vor
- Stelle sicher, dass der Arbeitsbereich frei ist
- Verlasse die Maschine während des Betriebs niemals unbeaufsichtigt
- Halte Feuerlöschgeräte in der Nähe
- Trage geeigneten Augenschutz
:::


### Dein Material vorbereiten

1. Platziere dein Material auf dem Laserbett
2. Fokussiere den Laser entsprechend den Anweisungen deiner Maschine
3. Wenn du die Kamera verwendest, richte dein Design mit der [Kamera-Überlagerung](../machine/camera) aus

### Den Job starten

1. **Laser positionieren**: Verwende die Jog-Steuerung, um den Laser an die Startposition zu bewegen
   - Klicke auf **Ansicht → Steuerungsfeld** oder drücke <kbd>ctrl+l</kbd>
   - Verwende die Pfeiltasten oder Tastaturpfeile, um den Laser zu bewegen
   - Drücke <kbd>home</kbd>, um die Maschine zu referenzieren

2. **Design rahmen**: Führe die Rahm-Funktion aus, um die Platzierung zu überprüfen
   - Klicke auf **Maschine → Rahmen** oder drücke <kbd>ctrl+f</kbd>
   - Der Laser zeichnet den Begrenzungsrahmen deines Designs bei niedriger/keiner Leistung nach
   - Überprüfe, ob er in dein Material passt

3. **Job starten**: Klicke auf **Maschine → Job starten** oder drücke <kbd>ctrl+r</kbd>
4. Überwache den Fortschritt in der Statusleiste

### Während des Jobs

- Der rechte Abschnitt der Statusleiste zeigt den aktuellen Fortschritt und die geschätzte Gesamtlaufzeit
- Du kannst den Job mit <kbd>ctrl+p</kbd> pausieren oder auf die Pause-Taste klicken
- Drücke <kbd>esc</kbd> oder klicke auf Stopp, um den Job abzubrechen (Notstopp)

## Schritt 6: Abschluss

Nachdem der Job abgeschlossen ist:

1. Warte, bis der Abluftventilator alle Dämpfe abgesaugt hat
2. Entferne vorsichtig dein fertiges Teil
3. Reinige bei Bedarf das Laserbett

:::success Herzlichen Glückwunsch!
Du hast deinen ersten Rayforge-Job abgeschlossen! Jetzt kannst du erweiterte Funktionen erkunden.
:::


## Nächste Schritte

Nachdem du deinen ersten Job abgeschlossen hast, erkunde diese Funktionen:

- **[Mehrschicht-Operationen](../features/multi-layer)**: Unterschiedliche Operationen Ebenen zuweisen
- **[Halte-Laschen](../features/holding-tabs)**: Geschnittene Teile während des Schneidens an Ort und Stelle halten
- **[Kamera-Integration](../machine/camera)**: Eine Kamera zur präzisen Ausrichtung verwenden
- **[Hooks & Makros](../machine/hooks-macros)**: Wiederkehrende Aufgaben automatisieren

## Tipps für den Erfolg

1. **Speichere deine Arbeit**: Verwende <kbd>ctrl+s</kbd>, um dein Projekt häufig zu speichern
2. **Testschnitte**: Führe immer zuerst einen Testschnitt auf Abfallmaterial durch
3. **Materialdatenbank**: Notiere dir erfolgreiche Leistungs-/Geschwindigkeitseinstellungen für verschiedene Materialien
4. **Wartung**: Halte deine Laserlinse sauber und überprüfe den Riemenzug regelmäßig
5. **Luftunterstützung**: Wenn deine Maschine über Luftunterstützung verfügt, verwende sie, um Verrußung zu verhindern und die Schnittqualität zu verbessern

---

**Hilfe benötigt?** Schaue im Abschnitt [Fehlerbehebung](../troubleshooting/connection) nach oder besuche die [GitHub-Issues](https://github.com/barebaric/rayforge/issues)-Seite.
