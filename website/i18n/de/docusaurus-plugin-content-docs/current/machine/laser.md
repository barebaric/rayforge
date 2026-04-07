# Lasereinstellungen

Die Laser-Seite in den Maschineneinstellungen konfiguriert deine Laserköpfe und
deren Eigenschaften.

![Lasereinstellungen](/screenshots/machine-laser.png)

## Laserköpfe

Rayforge unterstützt Maschinen mit mehreren Laserköpfen. Jeder Laserkopf hat seine eigene Konfiguration.

### Einen Laserkopf hinzufügen

Klicke auf die Schaltfläche **Laser hinzufügen**, um eine neue Laserkopf-Konfiguration zu erstellen.

### Laserkopf-Eigenschaften

Jeder Laserkopf hat die folgenden Einstellungen:

#### Name

Ein beschreibender Name für diesen Laserkopf.

Beispiele:
- "10W Diode"
- "CO2-Röhre"
- "Infrarotlaser"

#### Werkzeugnummer

Der Werkzeugindex für diesen Laserkopf. Im G-Code mit dem T-Befehl verwendet.

- Ein-Kopf-Maschinen: 0 verwenden
- Multi-Kopf-Maschinen: Eindeutige Nummern zuweisen (0, 1, 2, usw.)

#### Maximale Leistung

Der maximale Leistungswert für deinen Laser.

- **GRBL typisch**: 1000 (S0-S1000 Bereich)
- **Einige Controller**: 255 (S0-S255 Bereich)
- **Prozentmodus**: 100 (S0-S100 Bereich)

Dieser Wert sollte mit der $30-Einstellung deiner Firmware übereinstimmen.

#### Rahmen-Leistung

Der Leistungswert, der für Rahmen-Operationen verwendet wird (Umreißen ohne
Schneiden).

- Auf 0 setzen, um Rahmen zu deaktivieren
- Passe ihn basierend auf deinem Laser und Material an

#### Rahmen-Geschwindigkeit

Die Geschwindigkeit, mit der sich der Laserkopf während des Einrahmens bewegt.
Dies wird pro Laserkopf eingestellt, sodass du bei Maschinen mit mehreren Lasern
mit unterschiedlichen Eigenschaften eine angemessene Geschwindigkeit für jeden
wählen kannst. Langsamere Geschwindigkeiten machen den Rahmen-Pfad leichter
visuell verfolgbar.

#### Fokus-Leistung

Die Leistungsstufe, die verwendet wird, wenn der Fokusmodus aktiviert ist.
Der Fokusmodus schaltet den Laser mit niedriger Leistung ein, um als
"Laserzeiger" zur Positionierung zu dienen.

- Auf 0 setzen, um die Fokusmodus-Funktion zu deaktivieren
- Verwende für visuelle Ausrichtung und Positionierung

:::tip Fokusmodus verwenden
Klicke auf die Fokus-Taste (Laser-Symbol) in der Symbolleiste, um den
Fokusmodus umzuschalten. Der Laser wird bei dieser Leistungsstufe
eingeschaltet und hilft dir, genau zu sehen, wo der Laser positioniert ist.
Siehe [Werkstückpositionierung](../features/workpiece-positioning) für weitere
Informationen.
:::

#### Punktgröße

Die physische Größe deines fokussierten Laserstrahls in Millimetern.

- Gib sowohl X- als auch Y-Abmessungen ein
- Die meisten Laser haben einen runden Punkt (z.B. 0.1 x 0.1)
- Beeinflusst Gravurqualitäts-Berechnungen

:::tip Punktgröße messen
Um deine Punktgröße zu messen:
1. Feuere einen kurzen Impuls bei niedriger Leistung auf ein Testmaterial
2. Miss die resultierende Markierung mit einer Schieblehre
3. Verwende den Durchschnitt mehrerer Messungen
:::

#### Schnittfarbe

Die Farbe, die zum Anzeigen von Schnittoperationen für diesen Laser im Canvas und
in der 3D-Vorschau verwendet wird. Dies hilft dir, visuell zu unterscheiden,
welcher Laser welche Schnittoperation durchführen wird, wenn du mit mehreren
Laserköpfen arbeitest.

- Klicke auf die Farbauswahl, um einen Farbwähler zu öffnen
- Wähle eine Farbe, die gut mit deiner Materialvorschau kontrastiert
- Standardfarben werden automatisch zugewiesen

#### Rasterfarbe

Die Farbe, die zum Anzeigen von Raster-/Gravuroperationen für diesen Laser im
Canvas und in der 3D-Vorschau verwendet wird.

- Klicke auf die Farbauswahl, um einen Farbwähler zu öffnen
- Nützlich zur Unterscheidung von Rasteroperationen von Schnitten
- Jeder Laser kann seine eigene Rasterfarbe haben

:::tip Multi-Laser-Workflows
Bei der Verwendung mehrerer Laserköpfe erleichtert das Zuweisen unterschiedlicher
Farben zu jedem Laser es, zu erkennen, welche Operationen von welchem Laser
durchgeführt werden. Verwende beispielsweise Rot für deinen Hauptschneidelaser
und Blau für einen sekundären Gravurlaser.
:::

#### 3D-Modell

Jedem Laserkopf kann ein 3D-Modell zugewiesen werden. Dieses Modell wird in
der [3D-Ansicht](../ui/3d-preview) gerendert und folgt dem Werkzeugweg
während der Simulation.

Klicke auf die Modellauswahlzeile, um verfügbare Modelle zu durchsuchen.
Sobald ein Modell ausgewählt ist, kannst du dessen Skalierung, Rotation (X/Y/Z)
und Fokusabstand an deinen physischen Laserkopf anpassen.

## Siehe auch

- [Geräteeinstellungen](device) - GRBL Lasermodus-Einstellungen
- [Werkstückpositionierung](../features/workpiece-positioning) - Verwendung
  von Fokusmodus und anderen Positionierungsmethoden
