# Simulationsmodus

![Simulationsmodus](/screenshots/main-simulation.png)

Der Simulationsmodus bietet Echtzeit-Visualisierung Ihrer Laserjob-Ausführung, bevor Sie ihn auf der tatsächlichen Maschine ausführen. Er zeigt Ausführungsreihenfolge, Geschwindigkeitsvariationen und Leistungsstufen durch ein interaktives Overlay in der 2D-Ansicht.

## Übersicht

Der Simulationsmodus hilft Ihnen:

- **Ausführungsreihenfolge visualisieren** - Die genaue Sequenz sehen, in der Operationen laufen werden
- **Geschwindigkeitsvariationen identifizieren** - Heatmap zeigt langsame (blau) bis schnelle (rot) Bewegungen
- **Leistungsstufen überprüfen** - Transparenz zeigt Leistung an (fahl=niedrig, fett=hoch)
- **Materialtests validieren** - Testraster-Ausführungsreihenfolge bestätigen
- **Fehler frühzeitig erkennen** - Probleme erkennen, bevor Material verschwendet wird
- **Timing verstehen** - Sehen, wie lange verschiedene Operationen dauern

## Simulationsmodus aktivieren

Es gibt drei Möglichkeiten, in den Simulationsmodus zu gelangen:

### Methode 1: Tastatur-Kurzbefehl
Drücken Sie <kbd>f7</kbd>, um den Simulationsmodus ein/aus zu schalten.

### Methode 2: Menü
- Navigieren Sie zu **Ansicht → Ausführung simulieren**
- Klicken Sie zum Ein-/Ausschalten

### Methode 3: Symbolleiste (falls verfügbar)
- Auf die Simulationsmodus-Taste in der Symbolleiste klicken

:::note Nur 2D-Ansicht
Der Simulationsmodus funktioniert in der 2D-Ansicht. Wenn Sie sich in der 3D-Ansicht befinden (<kbd>f6</kbd>), wechseln Sie zuerst zur 2D-Ansicht (<kbd>f5</kbd>).
:::

## Die Visualisierung verstehen

### Geschwindigkeits-Heatmap

Operationen werden basierend auf ihrer Geschwindigkeit gefärbt:

| Farbe  | Geschwindigkeit | Bedeutung |
|--------|-----------------|-----------|
| 🔵 **Blau** | Langsamste | Minimale Geschwindigkeit in Ihrem Job |
| 🔵 **Cyan** | Langsam | Unterhalb der durchschnittlichen Geschwindigkeit |
| 🟢 **Grün** | Mittel | Durchschnittliche Geschwindigkeit |
| 🟡 **Gelb** | Schnell | Oberhalb der durchschnittlichen Geschwindigkeit |
| 🔴 **Rot** | Schnellste | Maximale Geschwindigkeit in Ihrem Job |

Die Heatmap wird auf den tatsächlichen **Geschwindigkeitsbereich Ihres Jobs normalisiert**:
- Wenn Ihr Job mit 100-1000 mm/min läuft, ist blau=100, rot=1000
- Wenn Ihr Job mit 5000-10000 mm/min läuft, ist blau=5000, rot=10000

### Leistungs-Transparenz

Liniendeckkraft zeigt Laserleistung an:

- **Fahle Linien** (10% Deckkraft) = Niedrige Leistung (0%)
- **Durchscheinend** (50% Deckkraft) = Mittlere Leistung (50%)
- **Feste Linien** (100% Deckkraft) = Volle Leistung (100%)

Dies hilft zu identifizieren:
- Verfahrbewegungen (0% Leistung) - Sehr fahl
- Gravur-Operationen - Mittlere Deckkraft
- Schneide-Operationen - Feste, fette Linien

### Laserkopf-Indikator

Die Laserposition wird mit einem Fadenkreuz angezeigt:

- 🔴 Rotes Fadenkreuz (6mm Linien)
- Kreis-Umriss (3mm Radius)
- Mittelpunkt (0.5mm)

Der Indikator bewegt sich während der Wiedergabe und zeigt genau, wo sich der Laser in der Ausführungssequenz befindet.

## Wiedergabe-Steuerungen

Wenn der Simulationsmodus aktiv ist, erscheinen Wiedergabe-Steuerungen unten auf der Arbeitsfläche:

### Wiedergabe/Pause-Taste

- **▶️ Wiedergabe**: Startet automatische Wiedergabe
- **⏸️ Pause**: Stoppt an aktueller Position
- **Auto-Wiedergabe**: Wiedergabe startet automatisch, wenn Sie den Simulationsmodus aktivieren

### Fortschritts-Schieberegler

- **Ziehen**, um durch die Ausführung zu scrollen
- **Klicken**, um zu einem bestimmten Punkt zu springen
- Zeigt aktuellen Schritt / Gesamtschritte
- Unterstützt fraktionale Positionen für sanftes Scrollen

### Geschwindigkeitsbereich-Anzeige

Zeigt die minimale und maximale Geschwindigkeit in Ihrem Job:

```
Geschwindigkeitsbereich: 100 - 5000 mm/min
```

Dies hilft Ihnen, die Heatmap-Farben zu verstehen.

## Den Simulationsmodus verwenden

### Ausführungsreihenfolge validieren

Die Simulation zeigt die genaue Reihenfolge, in der Operationen ausgeführt werden:

1. Simulationsmodus aktivieren (<kbd>f7</kbd>)
2. Die Wiedergabe ansehen
3. Verifizieren, dass Operationen in der erwarteten Sequenz laufen
4. Überprüfen, dass Schnitte nach dem Gravieren erfolgen (falls zutreffend)

**Beispiel:** Materialtest-Raster
- Risikooptimierte Reihenfolge beobachten (schnellste Geschwindigkeiten zuerst)
- Bestätigen, dass niedrigleistungs-Zellen vor hochleistungs-Zellen ausgeführt werden
- Validieren, dass Tests in sicherer Sequenz läuft

### Geschwindigkeitsvariationen überprüfen

Die Heatmap verwenden, um Geschwindigkeitsänderungen zu identifizieren:

- **Konsistente Farbe** = Gleichmäßige Geschwindigkeit (gut zum Gravieren)
- **Farbänderungen** = Geschwindigkeitsvariationen (erwartet an Ecken)
- **Blaue Bereiche** = Langsame Bewegungen (prüfen, ob beabsichtigt)

### Job-Zeit schätzen

Die Wiedergabedauer wird auf 5 Sekunden für den vollständigen Job skaliert:

- Die Wiedergabegeschwindigkeit ansehen
- Tatsächliche Zeit schätzen: Wenn die Wiedergabe sich flüssig anfühlt, wird der Job schnell sein
- Wenn die Wiedergabe schnell springt, hat der Job viele kleine Segmente

:::tip Tatsächliche Zeit
Für die tatsächliche Job-Zeit während der Ausführung (nicht Simulation), überprüfen Sie den rechten Abschnitt der Statusleiste nach der G-Code-Generierung.
:::

### Material-Tests debuggen

Für Materialtest-Raster zeigt die Simulation:

1. **Ausführungsreihenfolge** - Verifizieren, dass Zellen schnellsten→langsamsten laufen
2. **Geschwindigkeits-Heatmap** - Jede Spalte sollte eine andere Farbe haben
3. **Leistungs-Transparenz** - Jede Zeile sollte unterschiedliche Deckkraft haben

Dies hilft zu bestätigen, dass der Test korrekt laufen wird, bevor Material verwendet wird.

## Während des Simulierens bearbeiten

Im Gegensatz zu vielen CAM-Tools ermöglicht es Rayforge Ihnen, **Werkstücke während der Simulation zu bearbeiten**:

- Objekte bewegen, skalieren, drehen ✅
- Operationseinstellungen ändern ✅
- Werkstücke hinzufügen/entfernen ✅
- Zoomen und schwenken ✅

**Auto-Update:** Die Simulation aktualisiert sich automatisch, wenn Sie Einstellungen ändern.

:::note Kein Kontextwechsel
Sie können im Simulationsmodus bleiben, während Sie bearbeiten - kein Hin- und Her-Schalten nötig.
:::

## Tipps & Best Practices

### Wann Simulation verwenden

✅ **Immer simulieren vor:**
- Ausführen teurer Materialien
- Lange Jobs (>30 Minuten)
- Materialtest-Raster
- Jobs mit komplexer Ausführungsreihenfolge

✅ **Simulation verwenden, um:**
- Operations-Reihenfolge zu verifizieren
- Auf unerwartete Verfahrbewegungen zu prüfen
- Geschwindigkeits-/Leistungseinstellungen zu validieren
- Neue Benutzer zu schulen

### Die Visualisierung lesen

✅ **Suchen nach:**
- Konsistenten Farben innerhalb von Operationen (gut)
- Sanften Übergängen zwischen Segmenten (gut)
- Unerwarteten blauen Bereichen (untersuchen - warum so langsam?)
- Fahlen Linien in Schneidebereichen (falsch - Leistungseinstellungen überprüfen)

⚠️ **Rote Flaggen:**
- Schneiden vor Gravieren (Werkstück kann sich bewegen)
- Sehr lange blaue (langsame) Abschnitte (ineffizient)
- Leistungsänderungen mitten in der Operation (Einstellungen überprüfen)

### Leistungstipps

- Die Simulation aktualisiert sich automatisch bei Änderungen
- Bei sehr komplexen Jobs (1000+ Operationen) kann die Simulation langsamer werden
- Simulation deaktivieren (<kbd>f7</kbd>), wenn nicht benötigt, für bessere Leistung

## Tastatur-Kurzbefehle

| Kurzbefehl | Aktion |
|------------|--------|
| <kbd>f7</kbd> | Simulationsmodus ein/aus schalten |
| <kbd>f5</kbd> | Zur 2D-Ansicht wechseln (erforderlich für Simulation) |
| <kbd>Leertaste</kbd> | Wiedergabe/Pause |
| <kbd>links</kbd> | Schritt zurück |
| <kbd>rechts</kbd> | Schritt vorwärts |
| <kbd>home</kbd> | Zum Anfang springen |
| <kbd>ende</kbd> | Zum Ende springen |

## Verwandte Themen

- **[3D-Vorschau](../ui/3d-preview)** - 3D-Werkzeugweg-Visualisierung
- **[Materialtest-Raster](operations/material-test-grid)** - Simulation verwenden, um Tests zu validieren
- **[ Ihren Job simulieren](simulating-your-job)** - Detaillierte Simulationsanleitung
