# Bedienfeld

Das Bedienfeld am unteren Rand des Rayforge-Fensters bietet manuelle Kontrolle über die Position Ihres Laserschneiders, Echtzeit-Maschinenstatus und eine Protokollansicht zur Überwachung von Operationen.

## Übersicht

Das Bedienfeld kombiniert mehrere Funktionen in einer praktischen Schnittstelle:

1. **Jog-Steuerung**: Manuelle Bewegung und Positionierung
2. **Maschinenstatus**: Echtzeit-Position und Verbindungszustand
3. **Konsole**: Interaktives G-Code-Terminal mit Syntaxhervorhebung
4. **Werkstückkoordinatensystem (WCS)**: Schnelle WCS-Auswahl

![Bedienfeld](/screenshots/control-panel.png)

## Zugriff auf das Bedienfeld

Das Bedienfeld ist immer am unteren Rand des Hauptfensters sichtbar. Es kann umgeschaltet werden über:

- **Menü**: Ansicht → Bedienfeld
- **Tastaturkürzel**: Strg+L

:::note Verbindung erforderlich
Die Jog-Steuerung ist nur verfügbar, wenn mit einer Maschine verbunden ist, die Jogging-Operationen unterstützt.
:::


## Jog-Steuerung

Die Jog-Steuerung bietet manuelle Kontrolle über die Position Ihres Laserschneiders, sodass Sie den Laserkopf präzise für Einrichtung, Ausrichtung und Testzwecke bewegen können.

### Referenzfahrt-Steuerung

Referenzieren Sie die Achsen Ihrer Maschine, um eine Referenzposition zu etablieren:

| Schaltfläche | Funktion        | Beschreibung                             |
| ------------ | --------------- | ---------------------------------------- |
| X referenzieren | Referenziert X-Achse | Bewegt X-Achse zur Referenzposition |
| Y referenzieren | Referenziert Y-Achse | Bewegt Y-Achse zur Referenzposition |
| Z referenzieren | Referenziert Z-Achse | Bewegt Z-Achse zur Referenzposition |
| Alle referenzieren | Referenziert alle Achsen | Referenziert alle Achsen gleichzeitig |

:::tip Referenzfahrt-Sequenz
Es wird empfohlen, alle Achsen zu referenzieren bevor ein Auftrag gestartet wird, um präzise Positionierung sicherzustellen.
:::


### Richtungssteuerung

Die Jog-Steuerung bietet Schaltflächen für Richtungssteuerung:

```
  ↖  ↑  ↗
  ←  •  →
  ↙  ↓  ↘
```

| Schaltfläche      | Bewegung                         | Tastaturkürzel |
| ----------------- | -------------------------------- | ---------------|
| ↑                 | Y+ (Y- wenn Maschine Y-gespiegelt ist) | Pfeil nach oben |
| ↓                 | Y- (Y+ wenn Maschine Y-gespiegelt ist) | Pfeil nach unten |
| ←                 | X- (links)                       | Pfeil nach links |
| →                 | X+ (rechts)                      | Pfeil nach rechts |
| ↖ (oben-links)    | X- Y+/- (diagonal)               | -               |
| ↗ (oben-rechts)   | X+ Y+/- (diagonal)               | -               |
| ↙ (unten-links)   | X- Y-/+ (diagonal)               | -               |
| ↘ (unten-rechts)  | X+ Y-/+ (diagonal)               | -               |
| Z+                | Z-Achse hoch                     | Bild ↑          |
| Z-                | Z-Achse runter                   | Bild ↓          |

:::note Fokus erforderlich
Tastaturkürzel funktionieren nur, wenn das Hauptfenster den Fokus hat.
:::


### Visuelles Feedback

Die Jog-Schaltflächen bieten visuelles Feedback:

- **Normal**: Schaltfläche ist aktiviert und sicher zu verwenden
- **Warnung (orange)**: Bewegung würde Software-Endschalter erreichen oder überschreiten
- **Deaktiviert**: Bewegung wird nicht unterstützt oder Maschine ist nicht verbunden

### Jog-Einstellungen

Konfigurieren Sie das Verhalten von Jog-Operationen:

**Jog-Geschwindigkeit:**
- **Bereich**: 1-10.000 mm/min
- **Standard**: 1.000 mm/min
- **Zweck**: Steuert, wie schnell sich der Laserkopf bewegt

:::tip Geschwindigkeitsauswahl
- Niedrigere Geschwindigkeiten (100-500 mm/min) für präzise Positionierung verwenden
- Höhere Geschwindigkeiten (1.000-3.000 mm/min) für größere Bewegungen verwenden
- Sehr hohe Geschwindigkeiten können bei einigen Maschinen zu verpassten Schritten führen
:::


**Jog-Distanz:**
- **Bereich**: 0,1-1.000 mm
- **Standard**: 10,0 mm
- **Zweck**: Steuert, wie weit sich der Laserkopf pro Schaltflächenklick bewegt

:::tip Distanzauswahl
- Kleine Distanzen (0,1-1,0 mm) zum Feinabstimmen verwenden
- Mittlere Distanzen (5-20 mm) für allgemeines Positionieren verwenden
- Große Distanzen (50-100 mm) für schnelles Umpositionieren verwenden
:::


## Maschinenstatus-Anzeige

Das Bedienfeld zeigt Echtzeitinformationen über Ihre Maschine:

### Aktuelle Position

Zeigt die Position des Laserkopfes im aktiven Koordinatensystem:

- Koordinaten sind relativ zum ausgewählten WCS-Ursprung
- Aktualisiert sich in Echtzeit beim Joggen oder Ausführen von Aufträgen
- Format: X-, Y-, Z-Werte in Millimetern

### Verbindungsstatus

- **Verbunden**: Grüner Indikator, Maschine antwortet
- **Getrennt**: Grauer Indikator, keine Maschinenverbindung
- **Fehler**: Roter Indikator, Verbindungs- oder Kommunikationsproblem

### Maschinenstatus

- **Bereit (Idle)**: Maschine ist bereit für Befehle
- **Ausführung (Run)**: Auftrag wird gerade ausgeführt
- **Pause (Hold)**: Auftrag ist pausiert
- **Alarm**: Maschine ist im Alarmzustand
- **Referenzfahrt (Home)**: Referenzfahrt läuft

## Werkstückkoordinatensystem (WCS)

Das Bedienfeld bietet schnellen Zugriff auf die Werkstückkoordinatensystem-Verwaltung.

### Aktives System auswählen

Wählen Sie, welches Koordinatensystem gerade aktiv ist:

| Option          | Typ   | Beschreibung                                     |
| --------------- | ----- | ------------------------------------------------ |
| G53 (Maschine)  | Fest  | Absolute Maschinenkoordinaten, können nicht geändert werden |
| G54 (Arbeit 1)  | Benutzer | Erstes Werkstückkoordinatensystem             |
| G55 (Arbeit 2)  | Benutzer | Zweites Werkstückkoordinatensystem            |
| G56 (Arbeit 3)  | Benutzer | Drittes Werkstückkoordinatensystem            |
| G57 (Arbeit 4)  | Benutzer | Viertes Werkstückkoordinatensystem            |
| G58 (Arbeit 5)  | Benutzer | Fünftes Werkstückkoordinatensystem            |
| G59 (Arbeit 6)  | Benutzer | Sechstes Werkstückkoordinatensystem           |

### Aktuelle Offsets

Zeigt die Offset-Werte für das aktive WCS:

- Angezeigt als (X, Y, Z) in Millimetern
- Stellt die Entfernung vom Maschinenursprung zum WCS-Ursprung dar
- Aktualisiert sich automatisch wenn sich WCS-Offsets ändern

### WCS-Null setzen

Definieren Sie, wo der Ursprung des aktiven WCS sein soll:

| Schaltfläche | Funktion | Beschreibung                                          |
| ------------ | -------- | ---------------------------------------------------- |
| X nullen     | X=0 setzen | Macht aktuelle X-Position zum X-Ursprung für aktives WCS |
| Y nullen     | Y=0 setzen | Macht aktuelle Y-Position zum Y-Ursprung für aktives WCS |
| Z nullen     | Z=0 setzen | Macht aktuelle Z-Position zum Z-Ursprung für aktives WCS |

:::note G53 kann nicht geändert werden
Null-Schaltflächen sind deaktiviert wenn G53 (Maschinenkoordinaten) ausgewählt ist, da Maschinenkoordinaten durch die Hardware festgelegt sind.
:::


:::tip WCS-Einrichtungs-Workflow
1. Mit Ihrer Maschine verbinden und alle Achsen referenzieren
2. Das WCS auswählen, das Sie konfigurieren möchten (z.B. G54)
3. Den Laserkopf zur gewünschten Ursprungsposition jogen
4. Auf X nullen und Y nullen klicken um diese Position als (0, 0) zu setzen
5. Der Offset wird im Controller Ihrer Maschine gespeichert
:::


## Konsole

Die Konsole bietet eine interaktive Terminal-ähnliche Schnittstelle zum Senden von G-Code-Befehlen und Überwachen der Maschinenkommunikation:

### Befehlseingabe

Das Befehlseingabefeld ermöglicht Ihnen, rohen G-Code direkt an die Maschine zu senden:

- **Mehrzeilen-Unterstützung**: Mehrere Befehle einfügen oder eingeben
- **Eingabetaste**: Sendet alle Befehle
- **Umschalt+Eingabe**: Fügt eine neue Zeile ein (zum Bearbeiten vor dem Senden)
- **Verlauf**: Pfeil nach oben/unten verwenden um durch zuvor gesendete Befehle zu navigieren

### Protokollanzeige

Das Protokoll zeigt die Kommunikation zwischen Rayforge und Ihrer Maschine mit Syntaxhervorhebung zur einfachen Lesbarkeit:

- **Benutzerbefehle** (blau): Befehle, die Sie eingegeben oder während Aufträgen gesendet haben
- **Zeitstempel** (grau): Uhrzeit jeder Nachricht
- **Fehler** (rot): Fehlermeldungen von der Maschine
- **Warnungen** (orange): Warnmeldungen
- **Statusabfragen** (gedimmt): Echtzeit-Position/Statusberichte wie `<Idle|WPos:0.000,0.000,0.000|...>`

### Ausführlicher Modus

Klicken Sie auf das Terminal-Symbol in der oberen rechten Ecke der Konsole um die ausführliche Ausgabe umzuschalten:

- **Aus** (Standard): Versteckt häufige Statusabfragen und "ok"-Antworten
- **Ein**: Zeigt gesamte Maschinenkommunikation

### Auto-Scroll-Verhalten

Die Konsole scrollt automatisch um neue Nachrichten anzuzeigen:

- Nach oben scrollen deaktiviert Auto-Scroll damit Sie den Verlauf durchsehen können
- Nach unten scrollen aktiviert Auto-Scroll wieder
- Neue Nachrichten erscheinen sofort wenn Auto-Scroll aktiv ist

### Die Konsole zur Fehlerbehebung verwenden

Die Konsole ist unschätzbar wertvoll für die Diagnose von Problemen:

- Verifizieren dass Befehle korrekt gesendet werden
- Auf Fehlermeldungen vom Controller prüfen
- Verbindungsstatus und -stabilität überwachen
- Ausführungsfortschritt von Aufträgen in Echtzeit überprüfen
- Diagnosebefehle senden (z.B. `$$` um GRBL-Einstellungen anzuzeigen)

## Maschinenkompatibilität

Das Bedienfeld passt sich an die Fähigkeiten Ihrer Maschine an:

### Achsen-Unterstützung

- **X/Y-Achse**: Von praktisch allen Laserschneidern unterstützt
- **Z-Achse**: Nur auf Maschinen mit Z-Achsen-Steuerung verfügbar
- **Diagonale Bewegung**: Erfordert Unterstützung für beide X- und Y-Achsen

### Maschinentypen

| Maschinentyp       | Jog-Unterstützung | Hinweise                      |
| ------------------ | ----------------- | ----------------------------- |
| GRBL (v1.1+)       | Vollständig       | Unterstützt alle Jog-Funktionen |
| Smoothieware       | Vollständig       | Unterstützt alle Jog-Funktionen |
| Custom-Controller  | Variabel          | Hängt von der Implementierung ab |

## Sicherheitsfunktionen

### Software-Endschalter

Wenn Software-Endschalter in Ihrem Maschinenprofil aktiviert sind:

- Schaltflächen zeigen orangefarbene Warnung wenn Grenzen erreicht werden
- Bewegung wird automatisch begrenzt um das Überschreiten von Grenzen zu verhindern
- Bietet visuelles Feedback um Abstürze zu verhindern

### Verbindungsstatus

- Alle Steuerungen sind deaktiviert wenn nicht mit einer Maschine verbunden
- Schaltflächen aktualisieren Empfindlichkeit basierend auf Maschinenstatus
- Verhindert versehentliche Bewegung während des Betriebs

---

**Verwandte Seiten:**

- [Werkstückkoordinatensysteme (WCS)](../general-info/work-coordinate-systems) - WCS verwalten
- [Maschineneinrichtung](../machine/general) - Ihre Maschine konfigurieren
- [Tastaturkürzel](../reference/shortcuts) - Vollständige Kürzelreferenz
- [Hauptfenster](main-window) - Hauptoberflächenübersicht
- [Allgemeine Einstellungen](../machine/general) - Gerätekonfiguration
