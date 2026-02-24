# Geräteeinstellungen

Die Geräteseite in den Maschineneinstellungen ermöglicht es dir, Einstellungen direkt auf deinem verbundenen Gerät (Controller) zu lesen und anzuwenden. Diese sind auch als "Dollar-Einstellungen" oder `$$`-Einstellungen in GRBL bekannt.

![Geräteeinstellungen](/screenshots/machine-device.png)

:::warning Vorsicht beim Ändern von Einstellungen
Falsche Firmware-Einstellungen können dazu führen, dass deine Maschine sich unvorhersehbar verhält, die Position verliert oder sogar Hardware beschädigt. Notiere dir immer die ursprünglichen Werte, bevor du Änderungen vornimmst, und ändere eine Einstellung nach der anderen.
:::

## Übersicht

Die Geräteseite bietet direkten Zugriff auf die Firmware-Einstellungen deines Controllers. Hier kannst du:

- Aktuelle Einstellungen vom Gerät lesen
- Einzelne Einstellungen ändern
- Änderungen auf das Gerät anwenden

Firmware-Einstellungen steuern:

- **Bewegungsparameter**: Geschwindigkeitsgrenzen, Beschleunigung, Kalibrierung
- **Endschalter**: Homing-Verhalten, Software-/Hardware-Grenzen
- **Lasersteuerung**: Leistungsbereich, Lasermodus aktivieren
- **Elektrische Konfiguration**: Pin-Inversionen, Pullups
- **Berichterstattung**: Statusnachrichten-Format und -Häufigkeit

Diese Einstellungen werden auf deinem Controller gespeichert (nicht in Rayforge) und bleiben über Stromzyklen hinweg erhalten.

## Einstellungen lesen

Klicke auf die Schaltfläche **Vom Gerät lesen**, um die aktuellen Einstellungen von deinem verbundenen Controller abzurufen. Dies erfordert:

- Dass die Maschine verbunden ist
- Dass der Treiber das Lesen von Geräteeinstellungen unterstützt

## Einstellungen anwenden

Nach dem Ändern von Einstellungen werden Änderungen auf das Gerät angewendet. Das Gerät kann:

- Vorübergehend neu starten
- Trennen und wieder verbinden
- Einen Stromzyklus für einige Änderungen erfordern

## Konsolenzugriff

Du kannst Einstellungen auch über die G-Code-Konsole anzeigen/ändern:

**Alle Einstellungen anzeigen:**
```
$$
```

**Einzelne Einstellung anzeigen:**
```
$100
```

**Einstellung ändern:**
```
$100=80.0
```

**Standardeinstellungen wiederherstellen:**
```
$RST=$
```

:::danger Standardeinstellungen wiederherstellen löscht alle Einstellungen
Der Befehl `$RST=$` setzt alle GRBL-Einstellungen auf Werkseinstellungen zurück. Du verlierst alle Kalibrierungen und Einstellungen. Sichere deine Einstellungen zuerst!
:::

---

## Kritische Einstellungen für Laser

Diese Einstellungen sind am wichtigsten für den Laserbetrieb:

### $32 - Lasermodus

**Wert:** 0 = Deaktiviert, 1 = Aktiviert

**Zweck:** Aktiviert laserspezifische Funktionen in GRBL

**Wenn aktiviert (1):**
- Laser schaltet automatisch während G0 (Eilgang)-Bewegungen aus
- Leistung passt sich dynamisch während Beschleunigung/Verlangsamung an
- Verhindert versehentliche Verbrennungen während des Positionierens

**Wenn deaktiviert (0):**
- Laser verhält sich wie ein Spindel (CNC-Modus)
- Schaltet während Eilgängen nicht aus
- **Gefährlich für den Laserbetrieb!**

:::warning Immer Lasermodus aktivieren
$32 sollte **immer** auf 1 für Laserschneider gesetzt werden. Deaktivierter Lasermodus kann unbeabsichtigte Verbrennungen und Feuergefahren verursachen.
:::

### $30 & $31 - Laserleistungsbereich

**$30 - Maximale Laserleistung (RPM)**
**$31 - Minimale Laserleistung (RPM)**

**Zweck:** Definiert den Leistungsbereich für S-Befehle

**Typische Werte:**
- $30=1000, $31=0 (S0-S1000 Bereich, am häufigsten)
- $30=255, $31=0 (S0-S255 Bereich, einige Controller)

:::tip Rayforge-Konfiguration anpassen
Die "Max. Leistung"-Einstellung in deinen [Lasereinstellungen](laser) sollte deinem $30-Wert entsprechen. Wenn $30=1000, stelle die maximale Leistung in Rayforge auf 1000 ein.
:::

### $130 & $131 - Maximaler Verfahrweg

**$130 - X Maximaler Verfahrweg (mm)**
**$131 - Y Maximaler Verfahrweg (mm)**

**Zweck:** Definiert den Arbeitsbereich deiner Maschine

**Warum es wichtig ist:**
- Software-Limits ($20) verwenden diese Werte, um Abstürze zu verhindern
- Definiert die Grenzen des Koordinatensystems
- Muss mit deiner physischen Maschinengröße übereinstimmen

---

## Einstellungsreferenz

### Schrittmotor-Konfiguration ($0-$6)

Steuert die elektrischen Signale und das Timing der Schrittmotoren.

| Einstellung | Beschreibung | Typischer Wert |
|-------------|--------------|----------------|
| $0 | Schrittimpulszeit (μs) | 10 |
| $1 | Schritt-Leerlaufverzögerung (ms) | 25 |
| $2 | Schrittimpuls-Invertierung (Maske) | 0 |
| $3 | Schrittrichtung-Invertierung (Maske) | 0 |
| $4 | Schritt-Aktivierungs-Pin invertieren | 0 |
| $5 | Limit-Pins invertieren | 0 |
| $6 | Tast-Pin invertieren | 0 |

### Limits & Homing ($20-$27)

Steuert Endschalter und Homing-Verhalten.

| Einstellung | Beschreibung | Typischer Wert |
|-------------|--------------|----------------|
| $20 | Software-Limits aktivieren | 0 oder 1 |
| $21 | Hardware-Limits aktivieren | 0 |
| $22 | Homing-Zyklus aktivieren | 0 oder 1 |
| $23 | Homing-Richtung invertieren | 0 |
| $24 | Homing-Such-Vorschubrate (mm/min) | 25 |
| $25 | Homing-Suche-Suchrate (mm/min) | 500 |
| $26 | Homing-Entprell-Verzögerung (ms) | 250 |
| $27 | Homing-Abzug-Distanz (mm) | 1.0 |

### Spindel & Laser ($30-$32)

| Einstellung | Beschreibung | Laser-Wert |
|-------------|--------------|------------|
| $30 | Maximale Spindeldrehzahl | 1000.0 |
| $31 | Minimale Spindeldrehzahl | 0.0 |
| $32 | Lasermodus aktivieren | 1 |

### Achsen-Kalibrierung ($100-$102)

Definiert, wie viele Schrittmotor-Schritte einem Millimeter Bewegung entsprechen.

| Einstellung | Beschreibung | Hinweise |
|-------------|--------------|----------|
| $100 | X Schritte/mm | Abhängig von Riemenscheiben/Riemen-Verhältnis |
| $101 | Y Schritte/mm | Normalerweise gleich wie X |
| $102 | Z Schritte/mm | Nicht bei den meisten Lasern verwendet |

**Berechnung der Schritte/mm:**
```
Schritte/mm = (motor_steps_per_rev × microstepping) / (pulley_teeth × belt_pitch)
```

**Beispiel:** 200 Schritte/Rev, 16 Microstepping, 20 Zähne Riemenscheibe, GT2-Riemen:
```
Schritte/mm = (200 × 16) / (20 × 2) = 3200 / 40 = 80
```

### Achsen-Geschwindigkeit & Beschleunigung ($110-$122)

| Einstellung | Beschreibung | Typischer Wert |
|-------------|--------------|----------------|
| $110 | X max. Rate (mm/min) | 5000.0 |
| $111 | Y max. Rate (mm/min) | 5000.0 |
| $112 | Z max. Rate (mm/min) | 500.0 |
| $120 | X Beschleunigung (mm/s²) | 500.0 |
| $121 | Y Beschleunigung (mm/s²) | 500.0 |
| $122 | Z Beschleunigung (mm/s²) | 100.0 |

### Achsen-Verfahrweg ($130-$132)

| Einstellung | Beschreibung | Hinweise |
|-------------|--------------|----------|
| $130 | X max. Verfahrweg (mm) | Arbeitsbereich-Breite |
| $131 | Y max. Verfahrweg (mm) | Arbeitsbereich-Tiefe |
| $132 | Z max. Verfahrweg (mm) | Z-Verfahrweg (falls zutreffend) |

---

## Beispiel für häufige Konfiguration

### Typischer Diodenlaser (300×400mm)

```gcode
$0=10          ; Schrittimpuls 10μs
$1=255         ; Schritt-Leerlaufverzögerung 255ms
$2=0           ; Keine Schritt-Invertierung
$3=0           ; Keine Richtungs-Invertierung
$4=0           ; Keine Aktivierungs-Invertierung
$5=0           ; Keine Limit-Invertierung
$10=1          ; WPos melden
$11=0.010      ; Knoten-Abweichung 0.01mm
$12=0.002      ; Bogen-Toleranz 0.002mm
$13=0          ; mm melden
$20=1          ; Software-Limits aktiviert
$21=0          ; Hardware-Limits deaktiviert
$22=1          ; Homing aktiviert
$23=0          ; Home zu min
$24=50.0       ; Homing-Vorschub 50mm/min
$25=1000.0     ; Homing-Suche 1000mm/min
$26=250        ; Homing-Entprellung 250ms
$27=2.0        ; Homing-Abzug 2mm
$30=1000.0     ; Max. Leistung S1000
$31=0.0        ; Min. Leistung S0
$32=1          ; Lasermodus EIN
$100=80.0      ; X Schritte/mm
$101=80.0      ; Y Schritte/mm
$102=80.0      ; Z Schritte/mm
$110=5000.0    ; X max. Rate
$111=5000.0    ; Y max. Rate
$112=500.0     ; Z max. Rate
$120=500.0     ; X Beschleunigung
$121=500.0     ; Y Beschleunigung
$122=100.0     ; Z Beschleunigung
$130=400.0     ; X max. Verfahrweg
$131=300.0     ; Y max. Verfahrweg
$132=0.0       ; Z max. Verfahrweg
```

---

## Einstellungen sichern

### Sicherungsverfahren

1. **Über Rayforge:**
   - Geräteeinstellungen-Panel öffnen
   - "Einstellungen exportieren" klicken
   - Datei als `grbl-backup-YYYY-MM-DD.txt` speichern

2. **Über Konsole:**
   - Befehl `$$` senden
   - Gesamte Ausgabe in Textdatei kopieren
   - Mit Datum speichern

### Wiederherstellungsverfahren

1. Sicherungsdatei öffnen
2. Jede Zeile (`$100=80.0`, etc.) über Konsole senden
3. Mit Befehl `$$` verifizieren

:::tip Regelmäßige Sicherungen
Sichere deine Einstellungen nach jeder Kalibrierung oder Abstimmung. Speichere Sicherungen an einem sicheren Ort.
:::

---

## Siehe auch

- [Allgemeine Einstellungen](general) - Maschinenname und Geschwindigkeitseinstellungen
- [Lasereinstellungen](laser) - Laserkopf-Konfiguration
- [Verbindungsfehlerbehebung](../troubleshooting/connection) - Verbindungsprobleme beheben

## Externe Ressourcen

- [GRBL v1.1 Konfiguration](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Configuration)
- [GRBL v1.1 Befehle](https://github.com/gnea/grbl/wiki/Grbl-v1.1-Commands)
- [Grbl_ESP32 Dokumentation](https://github.com/bdring/Grbl_Esp32/wiki)
