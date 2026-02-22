# Schnellstart-Anleitung

Nachdem Rayforge installiert und Ihre Maschine konfiguriert ist, lassen Sie uns Ihren ersten Laserjob ausführen! Diese Anleitung führt Sie durch den Import eines Designs, die Konfiguration von Operationen und das Senden von G-Code an Ihre Maschine.

## Schritt 1: Ein Design importieren

Rayforge unterstützt verschiedene Dateiformate, darunter SVG, DXF, PDF und Rasterbilder (JPEG, PNG, BMP).

1. **Klicken Sie** auf **Datei → Öffnen** oder drücken Sie <kbd>ctrl+o</kbd>
2. Navigieren Sie zu Ihrer Designdatei und wählen Sie sie aus
3. Das Design erscheint auf der Arbeitsfläche

![Arbeitsfläche mit importiertem Design](/screenshots/main-standard.png)

:::tip Noch kein Design?
Sie können einfache Formen mit den Arbeitsflächen-Werkzeugen erstellen oder kostenlose SVG-Dateien von Seiten wie [Flaticon](https://www.flaticon.com/) oder [SVG Repo](https://www.svgrepo.com/) herunterladen.
:::


## Schritt 2: Ihr Design positionieren

Verwenden Sie die Arbeitsflächen-Werkzeuge, um Ihr Design zu positionieren und anzupassen:

- **Verschieben**: Mittelklick und ziehen, oder halten Sie <kbd>Leertaste</kbd> und ziehen
- **Zoom**: Mausrad, oder <kbd>ctrl+"+"</kbd> / <kbd>ctrl+"-"</kbd>
- **Bewegen**: Klicken Sie Ihr Design an und ziehen Sie es
- **Drehen**: Wählen Sie das Design aus und verwenden Sie die Drehgriffe
- **Skalieren**: Wählen Sie das Design aus und ziehen Sie die Eckgriffe

## Schritt 3: Eine Operation zuweisen

Operationen definieren, wie Rayforge Ihr Design verarbeitet. Häufige Operationen sind:

- **Kontur**: Entlang der Umrisse von Formen schneiden
- **Rastergravur**: Formen mit Hin-und-Her-Linien füllen (für Gravuren)
- **Tiefengravur**: 3D-Tiefeneffekte aus Bildern erstellen

### Eine Operation hinzufügen

1. Wählen Sie Ihr Design auf der Arbeitsfläche aus
2. Klicken Sie auf **Operationen → Operation hinzufügen** oder drücken Sie <kbd>ctrl+shift+a</kbd>
3. Wählen Sie den Operationstyp (z.B. "Kontur" zum Schneiden)
4. Konfigurieren Sie die Operationseinstellungen:
   - **Leistung**: Laserleistung in Prozent (niedrig beginnen und testen!)
   - **Geschwindigkeit**: Bewegungsgeschwindigkeit in mm/min
   - **Durchgänge**: Anzahl der Wiederholungen der Operation (nützlich zum Schneiden dicker Materialien)

![Operationseinstellungen](/screenshots/step-settings-contour-general.png)

:::warning Mit niedriger Leistung beginnen
Wenn Sie mit neuen Materialien arbeiten, beginnen Sie immer mit niedrigeren Leistungseinstellungen und führen Sie Test schnitte durch. Erhöhen Sie die Leistung schrittweise, bis Sie das gewünschte Ergebnis erzielen. Verwenden Sie die Funktion [Materialtest-Raster](../features/operations/material-test-grid), um systematisch optimale Einstellungen zu finden.
:::


## Schritt 4: Vorschau

Bevor Sie an Ihre Maschine senden, zeigen Sie den Werkzeugweg in 3D an:

1. Klicken Sie auf **Ansicht → 3D-Vorschau** oder drücken Sie <kbd>ctrl+3</kbd>
2. Das 3D-Vorschau-Fenster zeigt den vollständigen Werkzeugweg
3. Verwenden Sie Ihre Maus, um die Vorschau zu drehen und zu zoomen
4. Überprüfen Sie, ob der Pfad korrekt aussieht

![3D-Vorschau](/screenshots/main-3d.png)

:::tip Fehler frühzeitig erkennen
Die 3D-Vorschau hilft Ihnen, Probleme zu erkennen wie:

- Fehlende Pfade
- Falsche Reihenfolge
- Auf falsche Objekte angewendete Operationen
- Pfade, die Ihren Arbeitsbereich überschreiten
:::


## Schritt 5: An die Maschine senden

:::danger Sicherheit geht vor
- Stellen Sie sicher, dass der Arbeitsbereich frei ist
- Verlassen Sie die Maschine während des Betriebs niemals unbeaufsichtigt
- Halten Sie Feuerlöschgeräte in der Nähe
- Tragen Sie geeigneten Augenschutz
:::


### Ihr Material vorbereiten

1. Platzieren Sie Ihr Material auf dem Laserbett
2. Fokussieren Sie den Laser entsprechend den Anweisungen Ihrer Maschine
3. Wenn Sie die Kamera verwenden, richten Sie Ihr Design mit der [Kamera-Überlagerung](../machine/camera) aus

### Den Job starten

1. **Laser positionieren**: Verwenden Sie die Jog-Steuerung, um den Laser an die Startposition zu bewegen
   - Klicken Sie auf **Ansicht → Steuerungsfeld** oder drücken Sie <kbd>ctrl+l</kbd>
   - Verwenden Sie die Pfeiltasten oder Tastaturpfeile, um den Laser zu bewegen
   - Drücken Sie <kbd>home</kbd>, um die Maschine zu referenzieren

2. **Design rahmen**: Führen Sie die Rahm-Funktion aus, um die Platzierung zu überprüfen
   - Klicken Sie auf **Maschine → Rahmen** oder drücken Sie <kbd>ctrl+f</kbd>
   - Der Laser zeichnet den Begrenzungsrahmen Ihres Designs bei niedriger/keiner Leistung nach
   - Überprüfen Sie, ob er in Ihr Material passt

3. **Job starten**: Klicken Sie auf **Maschine → Job starten** oder drücken Sie <kbd>ctrl+r</kbd>
4. Überwachen Sie den Fortschritt in der Statusleiste

### Während des Jobs

- Der rechte Abschnitt der Statusleiste zeigt den aktuellen Fortschritt und die geschätzte Gesamtlaufzeit
- Sie können den Job mit <kbd>ctrl+p</kbd> pausieren oder auf die Pause-Taste klicken
- Drücken Sie <kbd>esc</kbd> oder klicken Sie auf Stopp, um den Job abzubrechen (Notstopp)

## Schritt 6: Abschluss

Nachdem der Job abgeschlossen ist:

1. Warten Sie, bis der Abluftventilator alle Dämpfe abgesaugt hat
2. Entfernen Sie vorsichtig Ihr fertiges Teil
3. Reinigen Sie bei Bedarf das Laserbett

:::success Herzlichen Glückwunsch!
Sie haben Ihren ersten Rayforge-Job abgeschlossen! Jetzt können Sie erweiterte Funktionen erkunden.
:::


## Nächste Schritte

Nachdem Sie Ihren ersten Job abgeschlossen haben, erkunden Sie diese Funktionen:

- **[Mehrschicht-Operationen](../features/multi-layer)**: Unterschiedliche Operationen Ebenen zuweisen
- **[Halte-Laschen](../features/holding-tabs)**: Geschnittene Teile während des Schneidens an Ort und Stelle halten
- **[Kamera-Integration](../machine/camera)**: Eine Kamera zur präzisen Ausrichtung verwenden
- **[Hooks & Makros](../machine/hooks-macros)**: Wiederkehrende Aufgaben automatisieren

## Tipps für den Erfolg

1. **Speichern Sie Ihre Arbeit**: Verwenden Sie <kbd>ctrl+s</kbd>, um Ihr Projekt häufig zu speichern
2. **Testschnitte**: Führen Sie immer zuerst einen Testschnitt auf Abfallmaterial durch
3. **Materialdatenbank**: Notieren Sie sich erfolgreiche Leistungs-/Geschwindigkeitseinstellungen für verschiedene Materialien
4. **Wartung**: Halten Sie Ihre Laserlinse sauber und überprüfen Sie den Riemenzug regelmäßig
5. **Luftunterstützung**: Wenn Ihre Maschine über Luftunterstützung verfügt, verwenden Sie sie, um Verrußung zu verhindern und die Schnittqualität zu verbessern

---

**Hilfe benötigt?** Schauen Sie im Abschnitt [Fehlerbehebung](../troubleshooting/connection) nach oder besuchen Sie die [GitHub-Issues](https://github.com/barebaric/rayforge/issues)-Seite.
