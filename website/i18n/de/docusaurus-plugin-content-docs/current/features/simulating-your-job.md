# Ihren Job simulieren

![Simulationsmodus-Screenshot](/screenshots/main-simulation.png)

Erfahren Sie, wie Sie den Simulationsmodus von Rayforge verwenden, um Ihren Laserjob vorauszuschauen, potenzielle Probleme zu identifizieren und die Fertigstellungszeit zu schätzen, bevor Sie auf echter Hardware ausführen.

## Übersicht

Der Simulationsmodus ermöglicht es Ihnen, Ihre Laserjob-Ausführung zu visualisieren, ohne die Maschine tatsächlich auszuführen. Dies hilft, Fehler zu erkennen, Einstellungen zu optimieren und Ihren Workflow zu planen.

## Vorteile der Simulation

- **Job-Ausführung vorschauen**: Genau sehen, wie sich der Laser bewegen wird
- **Zeit schätzen**: Genaue Job-Dauer-Schätzungen erhalten
- **Probleme identifizieren**: Überlappungen, Lücken oder unerwartetes Verhalten erkennen
- **Pfad-Reihenfolge optimieren**: Schneide-Sequenz visualisieren
- **G-Code lernen**: Verstehen, wie Operationen in Maschinenbefehle übersetzt werden

## Eine Simulation starten

1. **Laden oder erstellen Sie Ihr Design** in Rayforge
2. **Operationen konfigurieren** mit gewünschten Einstellungen
3. **Auf die Simulieren-Taste klicken** in der Symbolleiste (oder Tastenkombination verwenden)
4. **Die Simulation ansehen**, die Ihren Job abspielt

## Simulations-Steuerungen

### Wiedergabe-Steuerungen

- **Wiedergabe/Pause**: Simulation starten oder pausieren
- **Schritt Vorwärts/Rückwärts**: Durch den Job einen Befehl nach dem anderen bewegen
- **Geschwindigkeitssteuerung**: Wiedergabegeschwindigkeit anpassen (0.5x bis 10x)
- **Zu Position springen**: Zu bestimmtem Prozentsatz des Jobs springen
- **Neustart**: Simulation von Anfang an beginnen

### Visualisierungsoptionen

- **Werkzeugweg anzeigen**: Den Pfad anzeigen, dem der Laserkopf folgen wird
- **Verfahrbewegungen anzeigen**: Schnelle Positionierungsbewegungen visualisieren
- **Laserleistung anzeigen**: Pfade nach Leistungsstufe farbcodieren
- **Heatmap-Modus**: Verweildauer und Leistungsdichte visualisieren

### Informationsanzeige

Während der Simulation überwachen:

- **Aktuelle Position**: X, Y Koordinaten des Laserkopfs
- **Job-Fortschritt**: Prozentsatz abgeschlossen
- **Geschätzte verbleibende Zeit**: Basierend auf aktuellem Fortschritt
- **Aktuelle Operation**: Welche Operation wird ausgeführt
- **Leistung und Geschwindigkeit**: Aktuelle Laser-Parameter

## Simulationsergebnisse interpretieren

### Worauf achten

- **Pfad-Effizienz**: Gibt es unnötige Verfahrbewegungen?
- **Überlappende Schnitte**: Unbeabsichtigtes Doppelschneiden von Pfaden
- **Operations-Reihenfolge**: Macht die Sequenz Sinn?
- **Leistungsverteilung**: Wird Leistung konsistent angewendet?
- **Unerwartete Bewegungen**: Jegliche ruckartige oder seltsame Bewegungsmuster

### Heatmap-Visualisierung

Die Heatmap zeigt kumulative Laser-Exposition:

- **Kühle Farben (Blau/Grün)**: Niedrige Exposition
- **Warme Farben (Gelb/Orange)**: Mittlere Exposition
- **Heiße Farben (Rot)**: Hohe Exposition oder Verweildauer

Dies verwenden, um zu identifizieren:

- **Hotspots**: Bereiche, die überbrennen könnten
- **Lücken**: Bereiche, die unterbelichtet sein könnten
- **Überlappungsprobleme**: Unbeabsichtigte Doppel-Exposition

Siehe [Simulationsmodus](../features/simulation-mode) für detaillierte Informationen.

## Simulation zur Optimierung verwenden

### Schneide-Reihenfolge optimieren

Wenn die Simulation ineffiziente Pfad-Reihenfolge zeigt:

1. **Pfad-Optimierung aktivieren** in Operationseinstellungen
2. **Optimierungsmethode wählen** (Nächster Nachbar, TSP)
3. **Re-simulieren** zur Verifikation der Verbesserung

### Timing anpassen

Die Simulation liefert genaue Zeitschätzungen:

- **Lange Job-Zeiten**: Erwägen Sie, Pfade zu optimieren oder Geschwindigkeit zu erhöhen
- **Sehr kurze Zeiten**: Überprüfen Sie, ob Einstellungen für das Material korrekt sind
- **Unerwartete Dauer**: Auf versteckte Operationen oder Duplikate prüfen

### Mehrschicht-Jobs verifizieren

Für komplexe Mehrschicht-Projekte:

1. **Jede Ebene einzeln simulieren**
2. **Operations-Reihenfolge über Ebenen hinweg verifizieren**
3. **Auf Konflikte prüfen** zwischen Ebenen
4. **Gesamtzeit schätzen** für vollständigen Job

## Simulation vs. Echte Ausführung

### Zu beachtende Unterschiede

Die Simulation ist hochgenau, aber:

- **Berücksichtigt nicht**: Mechanische Unvollkommenheiten, Spiel, Vibration
- **Kann leicht abweichen**: Tatsächliche Beschleunigung/Verlangsamung vs. simuliert
- **Zeigt nicht**: Material-Interaktion, Rauch, Dämpfe
- **Zeitschätzungen**: Normalerweise innerhalb von 5-10% genau

### Wann Re-simulieren

- **Nach Änderung von Einstellungen**: Leistung, Geschwindigkeit oder Operationsparameter
- **Nach Design-Bearbeitung**: Jegliche Design-Änderungen
- **Vor teuren Materialien**: Doppelt prüfen, bevor Sie sich verpflichten
- **Bei Fehlerbehebung**: Überprüfen Sie Fixes für identifizierte Probleme

## Tipps für effektive Simulation

- **Immer simulieren** vor dem Ausführen wichtiger Jobs
- **Langsamere Wiedergabe verwenden**, um subtile Probleme zu erkennen
- **Heatmap aktivieren** für Gravur-Jobs
- **Mehrere Einstellungen vergleichen** durch Simulieren von Variationen
- **Ergebnisse dokumentieren**: Screenshot oder gefundene Probleme notieren

## Fehlerbehebung bei der Simulation

**Simulation startet nicht**: Überprüfen Sie, dass Operationen richtig konfiguriert sind

**Simulation läuft zu schnell**: Wiedergabegeschwindigkeit auf langsamere Einstellung anpassen

**Kann Details nicht sehen**: In spezifische Interessensbereiche hineinzoomen

**Zeitschätzung scheint falsch**: Verifizieren Sie, dass Maschinenprofil korrekte Max-Geschwindigkeiten hat

## Verwandte Themen

- [Simulationsmodus-Funktion](../features/simulation-mode)
- [Mehrschicht-Workflow](../features/multi-layer)
