# Firmware-Kompatibilität

Diese Seite dokumentiert die Firmware-Kompatibilität für Laser-Controller, die mit Rayforge verwendet werden.

## Übersicht

Rayforge ist primär für **GRBL-basierte Controller** konzipiert, bietet aber experimentelle Unterstützung für andere Firmware-Typen.

### Kompatibilitätsmatrix

| Firmware         | Version | Status           | Treiber                 | Hinweise                    |
| ---------------- | ------- | ---------------- | ----------------------- | --------------------------- |
| **GRBL**         | 1.1+    | Vollständig unterstützt | GRBL Serial       | Empfohlen                   |
| **grblHAL**      | 2023+   | Kompatibel       | GRBL Serial             | Moderner GRBL-Fork          |
| **GRBL**         | 0.9     | Eingeschränkt    | GRBL Serial             | Älter, kann Probleme haben  |
| **Smoothieware** | Alle    | Experimentell    | Keiner (GRBL-Treiber verwenden) | Ungetestet        |
| **Marlin**       | 2.0+    | Experimentell    | Keiner (GRBL-Treiber verwenden) | Laser-Modus erforderlich |
| **Andere**       | -       | Nicht unterstützt| -                       | Support anfordern           |

---

## GRBL-Firmware

**Status:** Vollständig unterstützt
**Versionen:** 1.1+
**Treiber:** GRBL Serial

### GRBL 1.1 (Empfohlen)

**Was ist GRBL 1.1?**

GRBL 1.1 ist die häufigste Firmware für Hobby-CNC- und Lasermaschinen. Veröffentlicht 2017, ist sie stabil, gut dokumentiert und weit verbreitet.

**Von Rayforge unterstützte Funktionen:**

- Serielle Kommunikation (USB)
- Echtzeit-Statusberichte
- Laser-Modus (M4 konstante Leistung)
- Einstellungen lesen/schreiben ($$, $X=Wert)
- Referenzfahrten ($H)
- Werkstückkoordinatensysteme (G54)
- Jogging-Befehle ($J=)
- Vorschub-Override
- Software-Endschalter
- Hardware-Endschalter (Endschalter)

**Bekannte Einschränkungen:**

- Leistungsbereich: 0-1000 (S-Parameter)
- Keine Netzwerkverbindung (nur USB)
- Begrenzter Onboard-Speicher (kleiner G-Code-Puffer)

### GRBL-Version überprüfen

**Version abfragen:**

Verbinde sich mit deinem Controller und sende:

```
$I
```

**Beispielantworten:**

```
[VER:1.1h.20190825:]
[OPT:V,15,128]
```

- `1.1h` = GRBL-Version 1.1h
- Datum zeigt den Build an

### GRBL 0.9 (Älter)

**Status:** Eingeschränkte Unterstützung

GRBL 0.9 ist eine ältere Version mit einigen Kompatibilitätsproblemen:

**Unterschiede:**

- Anderes Statusbericht-Format
- Kein Laser-Modus (M4) – verwendet nur M3
- Weniger Einstellungen
- Andere Jogging-Syntax

**Wenn du GRBL 0.9 haben:**

1. **Upgrade auf GRBL 1.1** wenn möglich (empfohlen)
2. **M3 statt M4 verwenden** (weniger vorhersagbare Leistung)
3. **Gründlich testen** – einige Funktionen funktionieren möglicherweise nicht

**Upgrade-Anleitung:** Siehe [GRBL Wiki](https://github.com/gnea/grbl/wiki)

---

## grblHAL

**Status:** Kompatibel
**Versionen:** 2023+
**Treiber:** GRBL Serial

### Was ist grblHAL?

grblHAL ist ein moderner Fork von GRBL mit erweiterten Funktionen:

- Unterstützung für mehrere Controller-Hardware (STM32, ESP32 usw.)
- Ethernet/WiFi-Netzwerk
- SD-Karten-Unterstützung
- Mehr I/O-Pins
- Erweiterter Laser-Support

**Kompatibilität mit Rayforge:**

- **Vollständig kompatibel** – grblHAL behält das GRBL 1.1-Protokoll bei
- Alle GRBL-Funktionen funktionieren
- Zusätzliche Funktionen (Netzwerk, SD) noch nicht von Rayforge unterstützt
- Statusberichte identisch zu GRBL

**Verwendung von grblHAL:**

1. Wähle "GRBL Serial"-Treiber in Rayforge
2. Verbinde über USB-Seriell (genau wie GRBL)
3. Alle Funktionen funktionieren wie für GRBL dokumentiert

**Zukunft:** Rayforge könnte Unterstützung für grblHAL-spezifische Funktionen hinzufügen (Netzwerk usw.)

---

## Smoothieware

**Versionen:** Alle
**Treiber:** GRBL Serial (Kompatibilitätsmodus)

### Kompatibilitätshinweise

Smoothieware verwendet eine andere G-Code-Syntax:

**Wichtige Unterschiede:**

| Funktion         | GRBL           | Smoothieware     |
| --------------- | -------------- | ---------------- |
| **Laser ein**   | `M4 S<Wert>`   | `M3 S<Wert>`     |
| **Leistungsbereich** | 0-1000    | 0.0-1.0 (Float)  |
| **Status**      | `<...>`-Format | Anderes Format   |

**Smoothieware mit Rayforge verwenden:**

1. **Smoothieware-Dialekt auswählen** in Maschineneinstellungen > G-Code > Dialekt
2. **Mit geringer Leistung testen** zuerst
3. **Leistungsbereich überprüfen** entspricht deiner Konfiguration
4. **Kein Echtzeit-Status** – eingeschränktes Feedback

**Einschränkungen:**

- Statusberichte nicht vollständig kompatibel
 Leistungsskalierung kann abweichen
- Einstellungen ($$-Befehle) nicht unterstützt
- Ungetestet auf echter Hardware

**Empfehlung:** Verwende wenn möglich GRBL-kompatible Firmware.

---

## Marlin

**Versionen:** 2.0+ mit Laser-Unterstützung
**Treiber:** GRBL Serial

### Marlin für Laser

Marlin 2.0+ kann Laser steuern, wenn richtig konfiguriert.

**Anforderungen:**

1. **Marlin 2.0 oder später** Firmware
2. **Laser-Funktionen aktiviert:**
   ```cpp
   #define LASER_FEATURE
   #define LASER_POWER_INLINE
   ```
3. **Korrekter Leistungsbereich** konfiguriert:
   ```cpp
   #define SPEED_POWER_MAX 1000
   ```

**Kompatibilität:**

- M4 Laser-Modus unterstützt
- Basis-G-Code (G0, G1, G2, G3)
- Statusberichte unterscheiden sich
- Einstellungsbefehle anders
- Luftunterstützung (M8/M9) funktioniert möglicherweise nicht

**Marlin mit Rayforge verwenden:**

1. **Marlin-Dialekt auswählen** in Maschineneinstellungen > G-Code > Dialekt
2. **Marlin für Laserbetrieb konfigurieren**
3. **Leistungsbereich testen** entspricht (0-1000 oder 0-255)
4. **Eingeschränkte Tests** – mit Vorsicht verwenden

**Bessere Alternative:** Verwende GRBL-Firmware auf Lasermaschinen.

---

## Firmware-Upgrade-Leitfaden

### Upgrade auf GRBL 1.1

**Warum upgraden?**

- Laser-Modus (M4) für konstante Leistung
- Bessere Statusberichte
- Zuverlässiger
- Bessere Rayforge-Unterstützung

**Wie man upgradet:**

1. **Identifiziere dein Controller-Board:**
   - Arduino Nano/Uno (ATmega328P)
   - Arduino Mega (ATmega2560)
   - Custom-Board

2. **GRBL 1.1 herunterladen:**
   - [GRBL Releases](https://github.com/gnea/grbl/releases)
   - Neueste 1.1-Version (1.1h empfohlen)

3. **Firmware flashen:**

   **Mit Arduino IDE:**

   ```
   1. Arduino IDE installieren
   2. GRBL-Sketch öffnen (grbl.ino)
   3. Korrektes Board und Port auswählen
   4. Hochladen
   ```

   **Mit avrdude:**

   ```bash
   avrdude -c arduino -p m328p -P /dev/ttyUSB0 \
           -U flash:w:grbl.hex:i
   ```

4. **GRBL konfigurieren:**
   - Über Seriell verbinden
   - `$$` senden um Einstellungen anzuzeigen
   - Für deine Maschine konfigurieren

### Backup vor Upgrade

**Einstellungen speichern:**

1. Mit Controller verbinden
2. `$$`-Befehl senden
3. Gesamte Einstellungsausgabe kopieren
4. In Datei speichern

**Nach dem Upgrade:**

- Einstellungen einzeln wiederherstellen: `$0=10`, `$1=25` usw.
- Oder Standards verwenden und neu konfigurieren

---

## Controller-Hardware

### Gängige Controller

| Board                  | Typische Firmware | Rayforge-Support |
| ---------------------- | ----------------- | ---------------- |
| **Arduino CNC Shield** | GRBL 1.1          | Ausgezeichnet    |
| **MKS DLC32**          | grblHAL           | Ausgezeichnet    |
| **Cohesion3D**         | Smoothieware      | Eingeschränkt    |
| **SKR-Boards**         | Marlin/grblHAL    | Variiert         |
| **Ruida**              | Proprietär        | Nicht unterstützt|
| **Trocen**             | Proprietär        | Nicht unterstützt|
| **TopWisdom**          | Proprietär        | Nicht unterstützt|

### Empfohlene Controller

Für beste Rayforge-Kompatibilität:

1. **Arduino Nano + CNC Shield** (GRBL 1.1)
   - Günstig (~$10-20)
   - Einfach zu flashen
   - Gut dokumentiert

2. **MKS DLC32** (grblHAL)
   - Modern (ESP32-basiert)
   - WiFi-fähig
   - Aktive Entwicklung

3. **Custom GRBL-Boards**
   - Viele auf Marktplätzen verfügbar
   - Auf GRBL 1.1+-Support prüfen

---

## Firmware-Konfiguration

### GRBL-Einstellungen für Laser

**Wesentliche Einstellungen:**

```
$30=1000    ; Max. Spindel/Laser-Leistung (1000 = 100%)
$31=0       ; Min. Spindel/Laser-Leistung
$32=1       ; Laser-Modus aktiviert (1 = ein)
```

**Maschineneinstellungen:**

```
$100=80     ; X Schritte/mm (für deine Maschine kalibrieren)
$101=80     ; Y Schritte/mm
$110=3000   ; X max. Rate (mm/min)
$111=3000   ; Y max. Rate
$120=100    ; X Beschleunigung (mm/sek)
$121=100    ; Y Beschleunigung
$130=300    ; X max. Verfahrweg (mm)
$131=200    ; Y max. Verfahrweg (mm)
```

**Sicherheitseinstellungen:**

```
$20=1       ; Software-Endschalter aktiviert
$21=1       ; Hardware-Endschalter aktiviert (wenn du Endschalter haben)
$22=1       ; Referenzfahrt aktiviert
```

### Firmware testen

**Grundlegende Testsequenz:**

1. **Verbindungstest:**

   ```
   Senden: ?
   Erwartet: <Idle|...>
   ```

2. **Versionsprüfung:**

   ```
   Senden: $I
   Erwartet: [VER:1.1...]
   ```

3. **Einstellungsprüfung:**

   ```
   Senden: $$
   Erwartet: $0=..., $1=..., usw.
   ```

4. **Bewegungstest:**

   ```
   Senden: G91 G0 X10
   Erwartet: Maschine bewegt sich 10mm in X
   ```

5. **Lasertest (sehr geringe Leistung):**
   ```
   Senden: M4 S10
   Erwartet: Laser schaltet ein (dim)
   Senden: M5
   Erwartet: Laser schaltet aus
   ```

---

## Fehlerbehebung bei Firmware-Problemen

### Firmware antwortet nicht

**Symptome:**

- Keine Antwort auf Befehle
- Verbindung fehlgeschlagen
- Status wird nicht gemeldet

**Diagnose:**

1. **Baudrate prüfen:**
   - GRBL 1.1 Standard: 115200
   - GRBL 0.9: 9600
   - Beide versuchen

2. **USB-Kabel prüfen:**
   - Datenkabel, nicht nur Ladekabel
   - Mit bekannt funktionierendem Kabel ersetzen

3. **Port prüfen:**
   - Linux: `/dev/ttyUSB0` oder `/dev/ttyACM0`
   - Windows: COM3, COM4 usw.
   - Korrekter Port in Rayforge ausgewählt

4. **Mit Terminal testen:**
   - screen, minicom oder PuTTY verwenden
   - `?` senden und schauen, ob Sie eine Antwort erhalten

### Firmware-Abstürze

**Symptome:**

- Controller friert während des Auftrags ein
- Zufällige Verbindungsabbrüche
- Inkonsistentes Verhalten

**Mögliche Ursachen:**

1. **Pufferüberlauf** – G-Code-Datei zu komplex
2. **Elektrische Störungen** – Schlechte Erdung oder EMI
3. **Firmware-Bug** – Auf neueste Version upgraden
4. **Hardware-Problem** – Defekter Controller

**Lösungen:**

- Firmware auf neueste stabile Version upgraden
- G-Code vereinfachen (Präzision reduzieren, weniger Segmente)
- Ferritkerne am USB-Kabel anbringen
- Erdung und Kabelverlegung verbessern

### Falsche Firmware

**Symptome:**

- Befehle abgelehnt
- Unerwartetes Verhalten
- Fehlermeldungen

**Lösung:**

1. Firmware-Version abfragen: `$I`
2. Mit Rayforge-Erwartungen vergleichen
3. Upgrade durchführen oder korrekten Dialekt auswählen

---

## Zukünftige Firmware-Unterstützung

### Angeforderte Funktionen

Benutzer haben Unterstützung angefordert für:

- **Ruida-Controller** – Chinesische Laser-Controller
- **Trocen/AWC** – Kommerzielle Laser-Controller
- **ESP32 WiFi** – Netzwerkverbindung für grblHAL
- **Laser-API** – Direkte Maschinen-API (kein G-Code)

**Status:** Derzeit nicht unterstützt. Funktionsanfragen auf GitHub willkommen.

### Mitwirken

Um Firmware-Support hinzuzufügen:

1. Treiber implementieren in `rayforge/machine/driver/`
2. G-Code-Dialekt definieren in `rayforge/machine/models/dialect.py`
3. Gründlich auf echter Hardware testen
4. Pull-Request mit Dokumentation einreichen

---

## Verwandte Seiten

- [G-Code-Dialekte](gcode-dialects) - Dialekt-Details
- [Geräteeinstellungen](../machine/device) - GRBL-Konfiguration
- [Verbindungsprobleme](../troubleshooting/connection) - Verbindungs-Fehlerbehebung
- [Allgemeine Einstellungen](../machine/general) - Maschineneinrichtung
