# Unteres Panel

Das untere Panel am unteren Rand des Rayforge-Fensters bietet manuelle
Kontrolle über die Position deines Laserschneiders, Echtzeit-Maschinenstatus,
eine Protokollansicht zur Überwachung von Operationen, einen G-Code-Viewer
und einen Asset-Browser.

## Übersicht

Das untere Panel kombiniert mehrere Funktionen in einer praktischen
Schnittstelle:

1. **Andockbare Tabs**: Zwischen Konsole, G-Code-Viewer und Assets wechseln
   über die Icon-Leiste auf der linken Seite
2. **Jog-Steuerung**: Manuelle Bewegung und Positionierung (immer sichtbar)
3. **Maschinenstatus**: Echtzeit-Position und Verbindungszustand
4. **Werkstückkoordinatensystem (WCS)**: Schnelle WCS-Auswahl (immer sichtbar)

Jeder Bereich des Panels hat eine Icon-Tab-Leiste auf der linken Seite, mit
der du zwischen der **Konsole**, dem **G-Code-Viewer** und dem **Assets**-Browser
wechseln kannst. Die Jog-Steuerung und WCS-Steuerung auf der rechten Seite
bleiben unabhängig davon sichtbar, welcher Tab aktiv ist. Tabs können durch
Ziehen innerhalb ihrer Leiste neu angeordnet werden, und du kannst Tabs
zwischen Panelbereichen oder auf Trennlinien ziehen, um das Layout in mehrere
Spalten umzuordnen. Leere Spalten werden automatisch entfernt.

![Unteres Panel](/screenshots/bottom-panel-console.png)

## Zugriff auf das untere Panel

Das untere Panel kann umgeschaltet werden über:

- **Menü**: Ansicht → Unteres Panel
- **Tastaturkürzel**: Strg+L

:::note Verbindung erforderlich
Die Jog-Steuerung ist nur verfügbar, wenn mit einer Maschine verbunden ist, die Jogging-Operationen unterstützt.
:::


## Jog-Steuerung

Die Jog-Steuerung bietet manuelle Kontrolle über die Position deines
Laserschneiders, sodass du den Laserkopf präzise für Einrichtung,
Ausrichtung und Testzwecke bewegen kannst.

### Referenzfahrt-Steuerung

Referenziere die Achsen deiner Maschine, um eine Referenzposition zu
etablieren:

| Schaltfläche | Funktion        | Beschreibung                             |
| ------------ | --------------- | ---------------------------------------- |
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
| ----------------- | -------------------------------- | -------------- |
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

Konfiguriere das Verhalten von Jog-Operationen:

**Jog-Geschwindigkeit:**
- **Bereich**: 1-60.000 mm/min
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

Das Steuerungs-Panel zeigt Echtzeitinformationen über deine Maschine:

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

Das Steuerungs-Panel bietet schnellen Zugriff auf die
Werkstückkoordinatensystem-Verwaltung.

### Aktives System auswählen

Wähle, welches Koordinatensystem gerade aktiv ist:

| Option          | Typ   | Beschreibung                                     |
| --------------- | ----- | ------------------------------------------------ |
| G53 (Maschine)  | Fest  | Absolute Maschinenkoordinaten, können nicht geändert werden |
| G54 (Arbeit 1)  | Benutzer | Erstes Werkstückkoordinatensystem             |
| G55 (Arbeit 2)  | Benutzer | Zweites Werkstückkoordinatensystem            |
| G56 (Arbeit 3)  | Benutzer | Drittes Werkstückkoordinatensystem            |
| G57 (Arbeit 4)  | Benutzer | Viertes Werkstückkoordinatensystem            |
| G58 (Arbeit 5)  | Benutzer | Fünftes Werkstückkoordinatensystem            |
| G59 (Arbeit 6)  | Benutzer | Sechstes Werkstückkoordinatensystem           |

### WCS-Null setzen

Definiere, wo der Ursprung des aktiven WCS sein soll:

| Schaltfläche     | Funktion      | Beschreibung                                                 |
| ---------------- | ------------- | ------------------------------------------------------------ |
| Klicken zum Nullen | X,Y=0 setzen  | Auf das Fadenkreuz-Symbol klicken, dann auf die Canvas klicken um Arbeitsnullpunkt zu setzen |
| Offsets bearbeiten | Bearbeiten    | WCS-Offset-Werte manuell bearbeiten                          |
| X nullen         | X=0 setzen    | Macht aktuelle X-Position zum X-Ursprung für aktives WCS    |
| Y nullen         | Y=0 setzen    | Macht aktuelle Y-Position zum Y-Ursprung für aktives WCS    |
| Z nullen         | Z=0 setzen    | Macht aktuelle Z-Position zum Z-Ursprung für aktives WCS    |

:::note G53 kann nicht geändert werden
Null-Schaltflächen sind deaktiviert wenn G53 (Maschinenkoordinaten) ausgewählt ist, da Maschinenkoordinaten durch die Hardware festgelegt sind.
:::


:::tip WCS-Einrichtungs-Workflow
1. Mit deiner Maschine verbinden und alle Achsen referenzieren
2. Das WCS auswählen, das du konfigurieren möchtest (z.B. G54)
3. Den Laserkopf zur gewünschten Ursprungsposition jogen
4. Auf X nullen und Y nullen klicken um diese Position als (0, 0) zu setzen
5. Der Offset wird im Controller deiner Maschine gespeichert
:::


## Konsole-Tab

Die Konsole bietet eine interaktive Terminal-ähnliche Schnittstelle zum Senden
von G-Code-Befehlen und Überwachen der Maschinenkommunikation. Klicke auf das
Konsolen-Symbol in der Tab-Leiste, um zu dieser Ansicht zu wechseln.

### Befehlseingabe

Das Befehlseingabefeld ermöglicht dir, rohen G-Code direkt an die Maschine zu
senden:

- **Mehrzeilen-Unterstützung**: Mehrere Befehle einfügen oder eingeben
- **Eingabetaste**: Sendet alle Befehle
- **Umschalt+Eingabe**: Fügt eine neue Zeile ein (zum Bearbeiten vor dem Senden)
- **Verlauf**: Pfeil nach oben/unten verwenden um durch zuvor gesendete Befehle zu navigieren

### Protokollanzeige

Das Protokoll zeigt die Kommunikation zwischen Rayforge und deiner Maschine
mit Syntaxhervorhebung zur einfachen Lesbarkeit:

- **Benutzerbefehle** (blau): Befehle, die du eingegeben oder während Aufträgen gesendet hast
- **Zeitstempel** (grau): Uhrzeit jeder Nachricht
- **Fehler** (rot): Fehlermeldungen von der Maschine
- **Warnungen** (orange): Warnmeldungen
- **Statusabfragen** (gedimmt): Echtzeit-Position/Statusberichte wie
  `&lt;Idle|WPos:0.000,0.000,0.000|...&gt;`

### Ausführlicher Modus

Klicke auf das Terminal-Symbol in der oberen rechten Ecke der Konsole um die
ausführliche Ausgabe umzuschalten:

- **Aus** (Standard): Versteckt häufige Statusabfragen und "ok"-Antworten
- **Ein**: Zeigt gesamte Maschinenkommunikation

### Auto-Scroll-Verhalten

Die Konsole scrollt automatisch um neue Nachrichten anzuzeigen:

- Nach oben scrollen deaktiviert Auto-Scroll, damit du den Verlauf durchsehen kannst
- Nach unten scrollen aktiviert Auto-Scroll wieder
- Neue Nachrichten erscheinen sofort wenn Auto-Scroll aktiv ist

### Die Konsole zur Fehlerbehebung verwenden

Die Konsole ist unschätzbar wertvoll für die Diagnose von Problemen:

- Verifizieren dass Befehle korrekt gesendet werden
- Auf Fehlermeldungen vom Controller prüfen
- Verbindungsstatus und -stabilität überwachen
- Ausführungsfortschritt von Aufträgen in Echtzeit überprüfen
- Diagnosebefehle senden (z.B. `$$` um GRBL-Einstellungen anzuzeigen)

## G-Code-Viewer-Tab

Der G-Code-Viewer zeigt den generierten G-Code für die aktuellen Operationen
an. Klicke auf das G-Code-Symbol in der Tab-Leiste, um zu dieser Ansicht zu
wechseln.

### Funktionen

- **Syntaxhervorhebung**: G-Code-Befehle sind farbcodiert für bessere Lesbarkeit
- **Zeilenmarkierung**: Die aktuell ausgeführte Zeile wird während der
  Auftragsausführung hervorgehoben
- **Automatische Aktualisierung**: Der G-Code-Inhalt wird automatisch
  aktualisiert wenn sich Operationen oder Dokumenteinstellungen ändern

## Assets-Tab

Der Assets-Tab zeigt alle Rohmaterialien und Skizzen in deinem Dokument an.
Klicke auf das Assets-Symbol in der Tab-Leiste, um zu dieser Ansicht zu
wechseln.

Wenn die Asset-Liste leer ist, werden Schaltflächen zum Hinzufügen von
Rohmaterial oder Erstellen einer neuen Skizze angezeigt. Du kannst Assets
aus dieser Liste auf die Canvas ziehen, um sie zu platzieren. Ein
Doppelklick auf ein Rohmaterial-Asset öffnet dessen Eigenschaften.

Ein Rechtsklick auf ein Asset öffnet ein Kontextmenü mit Optionen zum
Erstellen eines neuen Werkstücks aus dem Asset, Duplizieren, Kopieren,
Ausschneiden oder Löschen. Ein Rechtsklick auf eine leere Stelle in der
Asset-Liste bietet Optionen zum Erstellen einer neuen Skizze, Hinzufügen
von Rohmaterial, Importieren einer Datei oder Einfügen aus der
Zwischenablage.

Wenn der Konsole- oder G-Code-Viewer-Tab aktiv ist, kannst du auch
<kbd>Strg+F</kbd> drücken, um im Inhalt zu suchen.

## Maschinenkompatibilität

Das untere Panel passt sich an die Fähigkeiten deiner Maschine an:

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

Wenn Software-Endschalter in deinem Maschinenprofil aktiviert sind:

- Schaltflächen zeigen orangefarbene Warnung wenn Grenzen erreicht werden
- Bewegung wird automatisch begrenzt um das Überschreiten von Grenzen zu verhindern
- Bietet visuelles Feedback um Abstürze zu verhindern

### Verbindungsstatus

- Alle Steuerungen sind deaktiviert wenn nicht mit einer Maschine verbunden
- Schaltflächen aktualisieren Empfindlichkeit basierend auf Maschinenstatus
- Verhindert versehentliche Bewegung während des Betriebs

---

**Verwandte Seiten:**

- [Werkstückkoordinatensysteme (WCS)](../general-info/coordinate-systems) - WCS verwalten
- [Maschineneinrichtung](../machine/general) - Deine Maschine konfigurieren
- [Tastaturkürzel](../reference/shortcuts) - Vollständige Kürzelreferenz
- [Hauptfenster](main-window) - Hauptoberflächenübersicht
- [Allgemeine Einstellungen](../machine/general) - Gerätekonfiguration
