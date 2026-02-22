# Makros & Hooks

Rayforge bietet zwei leistungsstarke Automatisierungsfunktionen zur Anpassung Ihres Workflows: **Makros** und **Hooks**. Beide ermöglichen das Einfügen von benutzerdefiniertem G-Code in Ihre Jobs, dienen jedoch unterschiedlichen Zwecken.

![Hooks & Makros Einstellungen](/screenshots/machine-hooks-macros.png)

---

## Übersicht

| Funktion    | Zweck                       | Auslöser               | Anwendungsfall                                |
| ----------- | --------------------------- | ---------------------- | --------------------------------------------- |
| **Makros**  | Wiederverwendbare G-Code-Snippets | Manuelle Ausführung    | Schnelle Befehle, Testmuster, benutzerdefinierte Routinen |
| **Hooks**   | Automatische G-Code-Einschleusung | Job-Lebenszyklus-Ereignisse | Startsequenzen, Ebenenwechsel, Aufräumen     |

---

## Makros

Makros sind **benannte, wiederverwendbare G-Code-Skripte**, die Sie jederzeit manuell ausführen können.

### Wofür sind Makros?

Häufige Makro-Anwendungsfälle:

- **Maschine referenzieren** - Schnell `$H` senden
- **Arbeits-Offsets setzen** - G54/G55-Positionen speichern und abrufen
- **Luftunterstützung steuern** - Luftunterstützung ein/aus schalten
- **Fokustest** - Ein schnelles Fokustest-Muster ausführen
- **Benutzerdefinierte Werkzeugwechsel** - Für Multi-Laser-Setups
- **Notfall-Routinen** - Schnelles Herunterfahren oder Alarm löschen
- **Material-Tasten** - Autofokus oder Höhenmessung

### Ein Makro erstellen

1. **Maschineneinstellungen öffnen:**
   - Navigieren Sie zu **Einstellungen Maschine Makros**

2. **Ein neues Makro hinzufügen:**
   - Klicken Sie auf die **"+"**-Taste
   - Geben Sie einen beschreibenden Namen ein (z.B. "Maschine referenzieren", "Luftunterstützung aktivieren")

3. **Schreiben Sie Ihren G-Code:**
   - Jede Zeile ist ein separater G-Code-Befehl
   - Kommentare beginnen mit `;` oder `(`
   - Variablen können verwendet werden (siehe Variablensubstitution unten)

4. **Speichern Sie das Makro**

5. **Führen Sie das Makro aus:**
   - Klicken Sie in der Makroliste auf das Makro
   - Oder weisen Sie eine Tastenkombination zu (falls unterstützt)

### Beispiel-Makros

#### Einfach: Maschine referenzieren

**Name:** Maschine referenzieren
**Code:**

```gcode
$H
; Wartet auf Abschluss des Referenzierens
```

**Verwendung:** Die Maschine schnell referenzieren bevor Sie mit der Arbeit beginnen.

---

#### Mittel: Arbeits-Offset setzen

**Name:** G54 auf aktuelle Position setzen
**Code:**

```gcode
G10 L20 P1 X0 Y0
; Setzt G54 Arbeitskoordinatensystem-Ursprung auf aktuelle Position
```

**Verwendung:** Markieren Sie die aktuelle Laserposition als Job-Ursprung.

---

#### Erweitert: Fokus-Test-Raster

**Name:** 9-Punkt-Fokustest
**Code:**

```gcode
; 9-Punkt-Raster zum Finden des optimalen Fokus
G21  ; Millimeter
G90  ; Absolute Positionierung
G0 X10 Y10
M3 S1000
G4 P0.1
M5
G0 X20 Y10
M3 S1000
G4 P0.1
M5
; ... (Wiederholung für verbleibende Punkte)
```

**Verwendung:** Testen Sie schnell den Fokus an verschiedenen Positionen auf dem Bett.

---

---

### Makro-Beispiele

Hooks sind **automatische G-Code-Einschleusungen**, die durch bestimmte Ereignisse während der Jobausführung ausgelöst werden.

### Hook-Auslöser

Rayforge unterstützt diese Hook-Auslöser:

| Auslöser            | Wann es ausgeführt wird            | Häufige Verwendungen                          |
| ------------------- | ---------------------------------- | --------------------------------------------- |
| **Job-Start**       | Ganz am Anfang des Jobs            | Referenzieren, Arbeits-Offset, Luftunterstützung ein, Vorheizen |
| **Job-Ende**        | Ganz am Ende des Jobs              | Nach Hause zurückkehren, Luftunterstützung aus, Piepen, Abkühlen |
| **Ebenen-Start**    | Vor der Verarbeitung jeder Ebene   | Werkzeugwechsel, Leistung anpassen, Kommentare |
| **Ebenen-Ende**     | Nach der Verarbeitung jeder Ebene  | Fortschrittsbenachrichtigung, Pause           |
| **Werkstück-Start** | Vor der Verarbeitung jedes Werkstücks | Teilenummerierung, Ausrichtungsmarkierungen   |
| **Werkstück-Ende**  | Nach der Verarbeitung jedes Werkstücks | Abkühlen, Inspektionspause                    |

### Einen Hook erstellen

1. **Maschineneinstellungen öffnen:**
   - Navigieren Sie zu **Einstellungen Maschine Hooks**

2. **Einen Auslöser wählen:**
   - Wählen Sie das Ereignis, bei dem dieser Hook ausgeführt werden soll

3. **Schreiben Sie Ihren G-Code:**
   - Hook-Code wird am Auslöserpunkt eingeschoben
   - Verwenden Sie Variablen für dynamische Werte (siehe unten)

4. **Aktivieren/Deaktivieren:**
   - Schalten Sie Hooks ein/aus, ohne sie zu löschen

### Beispiel-Hooks

#### Job-Start: Maschine initialisieren

**Auslöser:** Job-Start
**Code:**

```gcode
G21         ; Millimeter
G90         ; Absolute Positionierung
$H          ; Maschine referenzieren
G0 X0 Y0    ; Zum Ursprung bewegen
M3 S0       ; Laser an aber Leistung 0 (manche Controller benötigen dies)
M8          ; Luftunterstützung EIN
```

**Zweck:** Stellt sicher, dass die Maschine vor jedem Job in einem bekannten Zustand ist.

---

#### Job-Ende: Nach Hause zurückkehren und piepen

**Auslöser:** Job-Ende
**Code:**

```gcode
M5          ; Laser AUS
M9          ; Luftunterstützung AUS
G0 X0 Y0    ; Zum Ursprung zurückkehren
M300 S800 P200  ; Piepen (falls unterstützt)
```

**Zweck:** Beendet den Job sicher und signalisiert den Abschluss.

---

#### Ebenen-Start: Kommentar hinzufügen

**Auslöser:** Ebenen-Start
**Code:**

```gcode
; Starte Ebene: {layer_name}
; Ebenen-Index: {layer_index}
```

**Zweck:** Macht G-Code besser lesbar zum Debuggen.

---

#### Werkstück-Start: Teilenummerierung

**Auslöser:** Werkstück-Start
**Code:**

```gcode
; Teil: {workpiece_name}
; Teil {workpiece_index} von {total_workpieces}
```

**Zweck:** Fortschritt in Multi-Teil-Jobs verfolgen.

---

### Hook-Ausführungsreihenfolge

Für einen Job mit 2 Ebenen, jede mit 2 Werkstücken:

```
[Job-Start Hook]
  [Ebenen-Start Hook] (Ebene 1)
    [Werkstück-Start Hook] (Werkstück 1)
      ... Werkstück 1 G-Code ...
    [Werkstück-Ende Hook] (Werkstück 1)
    [Werkstück-Start Hook] (Werkstück 2)
      ... Werkstück 2 G-Code ...
    [Werkstück-Ende Hook] (Werkstück 2)
  [Ebenen-Ende Hook] (Ebene 1)
  [Ebenen-Start Hook] (Ebene 2)
    [Werkstück-Start Hook] (Werkstück 3)
      ... Werkstück 3 G-Code ...
    [Werkstück-Ende Hook] (Werkstück 3)
    [Werkstück-Start Hook] (Werkstück 4)
      ... Werkstück 4 G-Code ...
    [Werkstück-Ende Hook] (Werkstück 4)
  [Ebenen-Ende Hook] (Ebene 2)
[Job-Ende Hook]
```

---

## Variablensubstitution

Sowohl Makros als auch Hooks unterstützen **Variablensubstitution**, um dynamische Werte einzufügen.

### Verfügbare Variablen

Variablen verwenden die `{variable_name}`-Syntax und werden während der G-Code-Generierung ersetzt.

**Job-Level-Variablen:**

| Variable     | Beschreibung                     | Beispielwert |
| ------------ | -------------------------------- | ------------ |
| `{job_name}` | Name des aktuellen Jobs/Dokuments | "test-job"   |
| `{date}`     | Aktuelles Datum                  | "2025-10-03" |
| `{time}`     | Aktuelle Uhrzeit                 | "14:30:25"   |

**Ebenen-Level-Variablen:**

| Variable         | Beschreibung                        | Beispielwert |
| ---------------- | ----------------------------------- | ------------ |
| `{layer_name}`   | Name der aktuellen Ebene            | "Schnitt-Ebene" |
| `{layer_index}`  | Null-basierter Index der aktuellen Ebene | 0, 1, 2...   |
| `{total_layers}` | Gesamtzahl der Ebenen im Job        | 3            |

**Werkstück-Level-Variablen:**

| Variable             | Beschreibung                            | Beispielwert |
| -------------------- | --------------------------------------- | ------------ |
| `{workpiece_name}`   | Name des Werkstücks                     | "Kreis 1"    |
| `{workpiece_index}`  | Null-basierter Index des aktuellen Werkstücks | 0, 1, 2...   |
| `{total_workpieces}` | Gesamtzahl der Werkstücke               | 5            |

**Maschinen-Variablen:**

| Variable         | Beschreibung                     | Beispielwert |
| ---------------- | -------------------------------- | ------------ |
| `{machine_name}` | Name des Maschinenprofils        | "Mein K40"   |
| `{max_speed}`    | Maximale Schnittgeschwindigkeit (mm/min) | 1000         |
| `{work_width}`   | Arbeitsbereich Breite (mm)       | 300          |
| `{work_height}`  | Arbeitsbereich Höhe (mm)         | 200          |

### Beispiel: Fortschrittsbenachrichtigung

**Hook:** Ebenen-Start
**Code:**

```gcode
; ========================================
; Ebene {layer_index} von {total_layers}: {layer_name}
; Job: {job_name}
; Uhrzeit: {time}
; ========================================
```

**Ergebnis im G-Code:**

```gcode
; ========================================
; Ebene 0 von 3: Schnitt-Ebene
; Job: test-projekt
; Uhrzeit: 14:30:25
; ========================================
```

---

## Erweiterte Anwendungsfälle

### Multi-Werkzeug-Setup

Für Maschinen mit mehreren Lasern oder Werkzeugen:

**Hook:** Werkstück-Start
**Code:**

```gcode
; Werkzeug für Werkstück {workpiece_name} wählen
T{tool_number}  ; Werkzeugwechsel-Befehl (falls unterstützt)
G4 P1           ; Auf Werkzeugwechsel warten
```

### Bedingte Pausen

Optionale Pausen für Inspektion hinzufügen:

**Hook:** Ebenen-Ende
**Code:**

```gcode
; M0  ; Auskommentieren, um nach jeder Ebene zur Inspektion zu pausieren
```

### Luftunterstützung pro Ebene

Luftunterstützung auf Ebenen-Basis steuern:

**Hook:** Ebenen-Start (für Schneide-Ebenen)
**Code:**

```gcode
M8  ; Luftunterstützung EIN
```

**Hook:** Ebenen-Start (für Gravur-Ebenen)
**Code:**

```gcode
M9  ; Luftunterstützung AUS (verhindert Staubspreizung beim Gravieren)
```

:::note Ebenen-spezifische Hooks
Rayforge unterstützt derzeit keine pro-Ebenen-Hook-Anpassung. Um dies zu erreichen, verwenden Sie bedingten G-Code oder separate Maschinenprofile.
:::

---

## Sicherheitsüberlegungen

:::danger Vor Produktion testen
Testen Sie Makros und Hooks immer im **Simulationsmodus** oder mit **deaktiviertem Laser**, bevor Sie echte Jobs ausführen. Falsch konfigurierter G-Code kann:

- Die Maschine gegen Grenzen krachen lassen
- Den Laser unerwartet feuern
- Material oder Ausrüstung beschädigen
  :::

**Sicherheits-Checkliste:**

- [ ] Makros enthalten Vorschub-Grenzen (`F`-Parameter)
- [ ] Makros überprüfen Positionsgrenzen
- [ ] Job-Start-Hooks enthalten `M5` oder Laser-aus-Befehl
- [ ] Job-Ende-Hooks schalten Laser (`M5`) und Luftunterstützung (`M9`) aus
- [ ] Keine destruktiven Befehle ohne Bestätigung
- [ ] In Simulation oder mit deaktiviertem Laser getestet

---

## Verwandte Seiten

- [Geräteeinstellungen](device) - GRBL-Befehlsreferenz
- [G-Code-Dialekte](../reference/gcode-dialects) - G-Code-Kompatibilität
- [Allgemeine Einstellungen](general) - Maschinenkonfiguration
- [Mehrschicht-Workflow](../features/multi-layer) - Hooks mit Ebenen verwenden
