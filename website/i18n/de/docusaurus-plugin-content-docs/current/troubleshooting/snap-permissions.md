# Snap-Berechtigungen (Linux)

Diese Seite erklärt, wie Sie Berechtigungen für Rayforge konfigurieren, wenn es als Snap-Paket unter Linux installiert wurde.

## Was sind Snap-Berechtigungen?

Snaps sind containerisierte Anwendungen, die in einer Sandbox zur Sicherheit ausgeführt werden. Standardmäßig haben sie begrenzten Zugriff auf Systemressourcen. Um bestimmte Funktionen zu nutzen (wie serielle Ports für Laser-Controller), müssen Sie explizit Berechtigungen erteilen.

## Erforderliche Berechtigungen

Rayforge benötigt diese Snap-Schnittstellen für volle Funktionalität:

| Schnittstelle | Zweck | Erforderlich? |
|---------------|-------|---------------|
| `serial-port` | Zugriff auf USB-Seriell-Geräte (Laser-Controller) | **Ja** (für Maschinensteuerung) |
| `home` | Dateien im Home-Verzeichnis lesen/schreiben | Automatisch verbunden |
| `removable-media` | Zugriff auf externe Laufwerke und USB-Speicher | Optional |
| `network` | Netzwerkverbindung (für Updates usw.) | Automatisch verbunden |

---

## Seriellen Port-Zugriff gewähren

**Dies ist die wichtigste Berechtigung für Rayforge.**

### Aktuelle Berechtigungen prüfen

```bash
# Alle Verbindungen für Rayforge anzeigen
snap connections rayforge
```

Suchen Sie nach der `serial-port`-Schnittstelle. Wenn sie "disconnected" oder "-" anzeigt, müssen Sie sie verbinden.

### Serielle Port-Schnittstelle verbinden

```bash
# Seriellen Port-Zugriff gewähren
sudo snap connect rayforge:serial-port
```

**Sie müssen dies nur einmal tun.** Die Berechtigung bleibt über App-Updates und Neustarts hinweg bestehen.

### Verbindung verifizieren

```bash
# Prüfen, ob serial-port jetzt verbunden ist
snap connections rayforge | grep serial-port
```

Erwartete Ausgabe:
```
serial-port     rayforge:serial-port     :serial-port     -
```

Wenn Sie einen Plug/Slot-Indikator sehen, ist die Verbindung aktiv.

---

## Wechseldatenträger-Zugriff gewähren

Wenn Sie Dateien von USB-Laufwerken oder externem Speicher importieren/exportieren möchten:

```bash
# Zugriff auf Wechseldatenträger gewähren
sudo snap connect rayforge:removable-media
```

Jetzt können Sie auf Dateien in `/media` und `/mnt` zugreifen.

---

## Fehlerbehebung bei Snap-Berechtigungen

### Serieller Port funktioniert immer noch nicht

**Nach dem Verbinden der Schnittstelle:**

1. **USB-Gerät neu anschließen:**
   - Ihren Laser-Controller abziehen
   - 5 Sekunden warten
   - Wieder einstecken

2. **Rayforge neu starten:**
   - Rayforge vollständig schließen
   - Über das Anwendungsmenü neu starten oder:
     ```bash
     snap run rayforge
     ```

3. **Prüfen, dass der Port erscheint:**
   - Rayforge öffnen → Einstellungen → Maschine
   - Nach seriellen Ports im Dropdown suchen
   - Sollte `/dev/ttyUSB0`, `/dev/ttyACM0` oder ähnliches sehen

4. **Verifizieren, dass das Gerät existiert:**
   ```bash
   # USB-Seriell-Geräte auflisten
   ls -l /dev/ttyUSB* /dev/ttyACM*
   ```

### "Permission Denied" trotz verbundener Schnittstelle

Dies ist selten, kann aber passieren wenn:

1. **Die Snap-Installation beschädigt ist:**
   ```bash
   # Snap neu installieren
   sudo snap refresh rayforge --devmode
   # Oder falls das fehlschlägt:
   sudo snap remove rayforge
   sudo snap install rayforge
   # Schnittstellen neu verbinden
   sudo snap connect rayforge:serial-port
   ```

2. **Konfliktierende udev-Regeln:**
   - Prüfen Sie `/etc/udev/rules.d/` auf benutzerdefinierte serielle Port-Regeln
   - Diese könnten mit Snap's Gerätezugriff kollidieren

3. **AppArmor-Ablehnungen:**
   ```bash
   # Auf AppArmor-Ablehnungen prüfen
   sudo journalctl -xe | grep DENIED | grep rayforge
   ```

   Wenn Sie Ablehnungen für serielle Ports sehen, kann es einen AppArmor-Profil-Konflikt geben.

### Kein Zugriff auf Dateien außerhalb des Home-Verzeichnisses

**Standardmäßig** können Snaps nicht auf Dateien außerhalb Ihres Home-Verzeichnisses zugreifen, es sei denn, Sie gewähren `removable-media`.

**Behelfslösungen:**

1. **Dateien in Ihr Home-Verzeichnis verschieben:**
   ```bash
   # SVG-Dateien nach ~/Dokumente kopieren
   cp /irgendein/anderer/ort/*.svg ~/Dokumente/
   ```

2. **Removable-media-Zugriff gewähren:**
   ```bash
   sudo snap connect rayforge:removable-media
   ```

3. **Snap's Dateiauswahl verwenden:**
   - Der integrierte Dateiwähler hat breiteren Zugriff
   - Dateien über Datei → Öffnen öffnen statt über Kommandozeilen-Argumente

---

## Manuelle Schnittstellen-Verwaltung

### Alle verfügbaren Schnittstellen auflisten

```bash
# Alle Snap-Schnittstellen auf Ihrem System sehen
snap interface
```

### Eine Schnittstelle trennen

```bash
# Serial-port trennen (falls erforderlich)
sudo snap disconnect rayforge:serial-port
```

### Nach Trennung wieder verbinden

```bash
sudo snap connect rayforge:serial-port
```

---

## Alternative: Aus dem Quellcode installieren

Wenn Snap-Berechtigungen zu restriktiv für Ihren Workflow sind:

**Option 1: Aus dem Quellcode bauen**

```bash
# Repository klonen
git clone https://github.com/kylemartin57/rayforge.git
cd rayforge

# Abhängigkeiten mit pixi installieren
pixi install

# Rayforge ausführen
pixi run rayforge
```

**Vorteile:**
- Keine Berechtigungseinschränkungen
- Voller Systemzugriff
- Einfacheres Debugging
- Neueste Entwicklungsversion

**Nachteile:**
- Manuelle Updates (git pull)
- Mehr Abhängigkeiten zu verwalten
- Keine automatischen Updates

**Option 2: Flatpak verwenden (falls verfügbar)**

Flatpak hat ähnliches Sandboxing, aber manchmal mit anderen Berechtigungsmodellen. Prüfen Sie, ob Rayforge ein Flatpak-Paket anbietet.

---

## Snap-Berechtigungs-Best-Practices

### Nur verbinden, was Sie benötigen

Verbinden Sie keine Schnittstellen, die Sie nicht verwenden:

- ✓ Verbinden Sie `serial-port` wenn Sie einen Laser-Controller verwenden
- ✓ Verbinden Sie `removable-media` wenn Sie von USB-Laufwerken importieren
- ✗ Verbinden Sie nicht alles "für alle Fälle" - widerspricht dem Sicherheitszweck

### Snap-Quelle verifizieren

Installieren Sie immer aus dem offiziellen Snap Store:

```bash
# Herausgeber prüfen
snap info rayforge
```

Suchen Sie nach:
- Verifiziertem Herausgeber
- Offizieller Repository-Quelle
- Regelmäßigen Updates

---

## Snap-Sandbox verstehen

### Was können Snaps standardmäßig zugreifen?

**Erlaubt:**
- Dateien in Ihrem Home-Verzeichnis
- Netzwerkverbindungen
- Anzeige/Audio

**Nicht erlaubt ohne explizite Berechtigung:**
- Serielle Ports (USB-Geräte)
- Wechseldatenträger
- Systemdateien
- Home-Verzeichnisse anderer Benutzer

### Warum dies für Rayforge wichtig ist

Rayforge benötigt:

1. **Home-Verzeichnis-Zugriff** (automatisch gewährt)
   - Um Projektdateien zu speichern
   - Um importierte SVG/DXF-Dateien zu lesen
   - Um Einstellungen zu speichern

2. **Seriellen Port-Zugriff** (muss gewährt werden)
   - Um mit Laser-Controllern zu kommunizieren
   - **Dies ist die kritische Berechtigung**

3. **Wechseldatenträger** (optional)
   - Um Dateien von USB-Laufwerken zu importieren
   - Um G-Code auf externen Speicher zu exportieren

---

## Snap-Probleme debuggen

### Ausführliche Snap-Protokollierung aktivieren

```bash
# Snap mit Debug-Ausgabe ausführen
snap run --shell rayforge
# Innerhalb der Snap-Shell:
export RAYFORGE_LOG_LEVEL=DEBUG
exec rayforge
```

### Snap-Protokolle prüfen

```bash
# Rayforge-Protokolle anzeigen
snap logs rayforge

# Protokolle in Echtzeit verfolgen
snap logs -f rayforge
```

### System-Journal auf Ablehnungen prüfen

```bash
# Nach AppArmor-Ablehnungen suchen
sudo journalctl -xe | grep DENIED | grep rayforge

# Nach USB-Geräte-Ereignissen suchen
sudo journalctl -f -u snapd
# Dann Ihren Laser-Controller einstecken
```

---

## Hilfe erhalten

Wenn Sie immer noch Snap-bezogene Probleme haben:

1. **Zuerst Berechtigungen prüfen:**
   ```bash
   snap connections rayforge
   ```

2. **Einen seriellen Port-Test versuchen:**
   ```bash
   # Falls Sie screen oder minicom installiert haben
   sudo snap connect rayforge:serial-port
   # Dann in Rayforge testen
   ```

3. **Das Problem melden mit:**
   - Ausgabe von `snap connections rayforge`
   - Ausgabe von `snap version`
   - Ausgabe von `snap info rayforge`
   - Ihrer Ubuntu/Linux-Distributionsversion
   - Genauen Fehlermeldungen

4. **Alternativen in Betracht ziehen:**
   - Aus dem Quellcode installieren (siehe oben)
   - Ein anderes Paketformat verwenden (AppImage, Flatpak)

---

## Schnellreferenz-Befehle

```bash
# Seriellen Port-Zugriff gewähren (am wichtigsten)
sudo snap connect rayforge:serial-port

# Wechseldatenträger-Zugriff gewähren
sudo snap connect rayforge:removable-media

# Aktuelle Verbindungen prüfen
snap connections rayforge

# Rayforge-Protokolle anzeigen
snap logs rayforge

# Rayforge aktualisieren
sudo snap refresh rayforge

# Entfernen und neu installieren (letztes Mittel)
sudo snap remove rayforge
sudo snap install rayforge
sudo snap connect rayforge:serial-port
```

---

## Verwandte Seiten

- [Verbindungsprobleme](connection) - Serielle Verbindungs-Fehlerbehebung
- [Debug-Modus](debug) - Diagnose-Protokollierung aktivieren
- [Installation](../getting-started/installation) - Installationsanleitung
- [Allgemeine Einstellungen](../machine/general) - Maschineneinrichtung
