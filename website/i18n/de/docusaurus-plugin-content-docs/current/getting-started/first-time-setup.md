# Ersteinrichtung

Nach der Installation von Rayforge musst du deinen Laserschneider oder Gravierer konfigurieren. Diese Anleitung führt dich durch die Erstellung deiner ersten Maschine und den Aufbau einer Verbindung.

## Schritt 1: Rayforge starten

Starte Rayforge aus deinem Anwendungsmenü oder indem du `rayforge` in einem Terminal ausführst. Du solltest die Hauptoberfläche mit einer leeren Arbeitsfläche sehen.

## Schritt 2: Eine Maschine erstellen

Navigiere zu **Einstellungen → Maschinen** oder drücke <kbd>ctrl+comma</kbd>, um den Einstellungsdialog zu öffnen, und wähle dann die Seite **Maschinen**.

Klicke auf **Maschine hinzufügen**, um eine neue Maschine zu erstellen. Du kannst entweder:

1. **Ein integriertes Profil wählen** - Aus vordefinierten Maschinenvorlagen auswählen
2. **"Benutzerdefiniert" wählen** - Mit einer leeren Konfiguration beginnen

Nach der Auswahl öffnet sich der Maschineneinstellungs-Dialog für deine neue Maschine.

![Maschineneinstellungen](/screenshots/application-machines.png)

## Schritt 3: Allgemeine Einstellungen konfigurieren

Die Seite **Allgemein** enthält grundlegende Maschineninformationen, Treiberauswahl und Verbindungseinstellungen.

![Allgemeine Einstellungen](/screenshots/machine-general.png)

### Maschineninformationen

1. **Maschinenname**: Gib deiner Maschine einen beschreibenden Namen (z.B. "K40 Laser", "Ortur LM2")

### Treiberauswahl

Wähle den entsprechenden Treiber für dein Gerät aus dem Dropdown-Menü:

- **GRBL Seriell** - Für GRBL-Geräte, die über USB/Seriell-Port verbunden sind
- **GRBL Netzwerk** - Für GRBL-Geräte mit WiFi/Ethernet-Konnektivität
- **Smoothie** - Für Smoothieware-basierte Geräte

### Treibereinstellungen

Je nach ausgewähltem Treiber konfigurierst du die Verbindungsparameter:

#### GRBL Seriell (USB)

1. **Port**: Wähle dein Gerät aus dem Dropdown-Menü (z.B. `/dev/ttyUSB0` unter Linux, `COM3` unter Windows)
2. **Baudrate**: Wähle `115200` (Standard für die meisten GRBL-Geräte)

:::info
Wenn dein Gerät nicht in der Liste erscheint, überprüfe, ob es angeschlossen ist und dass du über die notwendigen Berechtigungen verfügst. Unter Linux musst du möglicherweise deinen Benutzer zur `dialout`-Gruppe hinzufügen.
:::

#### GRBL Netzwerk / Smoothie (WiFi/Ethernet)

1. **Host**: Gib die IP-Adresse deines Geräts ein (z.B. `192.168.1.100`)
2. **Port**: Gib die Portnummer ein (typischerweise `23` oder `8080`)

### Geschwindigkeiten & Beschleunigung

Diese Einstellungen werden für Job-Zeitschätzung und Pfadoptimierung verwendet:

- **Max. Verfahrgeschwindigkeit**: Maximale schnelle Bewegungsgeschwindigkeit
- **Max. Schnittgeschwindigkeit**: Maximale Schnittgeschwindigkeit
- **Beschleunigung**: Für Zeitschätzungen und Overscan-Berechnungen verwendet

## Schritt 4: Hardware-Einstellungen konfigurieren

Wechsle zum Reiter **Hardware**, um die physischen Abmessungen deiner Maschine einzurichten.

![Hardware-Einstellungen](/screenshots/machine-hardware.png)

### Abmessungen

- **Breite**: Gib die maximale Breite deines Arbeitsbereichs in Millimetern ein
- **Höhe**: Gib die maximale Höhe deines Arbeitsbereichs in Millimetern ein

### Achsen

- **Koordinatenursprung (0,0)**: Wähle, wo sich der Ursprung deiner Maschine befindet:
  - Unten links (am häufigsten bei GRBL)
  - Oben links
  - Oben rechts
  - Unten rechts

### Achsen-Offsets (Optional)

Konfiguriere X- und Y-Offsets, wenn deine Maschine diese für präzises Positionieren benötigt.

## Schritt 5: Automatische Verbindung

Rayforge verbindet sich automatisch mit deiner Maschine, wenn die Anwendung startet (wenn die Maschine eingeschaltet und verbunden ist). Du musst nicht manuell auf eine Verbindungstaste klicken.

Der Verbindungsstatus wird in der unteren linken Ecke des Hauptfensters mit einem Statussymbol und einer Beschriftung angezeigt, die den aktuellen Zustand zeigt (Verbunden, Verbinden, Getrennt, Fehler usw.).

:::success Verbunden!
Wenn deine Maschine den Status "Verbunden" anzeigt, bist du bereit, Rayforge zu verwenden!
:::

## Optional: Erweiterte Konfiguration

### Mehrere Laser

Wenn deine Maschine über mehrere Lasermodule verfügt (z.B. Diode und CO2), kannst du diese auf der Seite **Laser** konfigurieren.

![Lasereinstellungen](/screenshots/machine-laser.png)

Siehe [Laser-Konfiguration](../machine/laser) für Details.

### Kamera-Setup

Wenn du eine USB-Kamera für Ausrichtung und Positionierung hast, konfiguriere sie auf der Seite **Kamera**.

![Kameraeinstellungen](/screenshots/machine-camera.png)

Siehe [Kamera-Integration](../machine/camera) für Details.

### Geräte-Einstellungen

Die Seite **Gerät** ermöglicht es dir, Firmware-Einstellungen direkt auf deinem verbundenen Gerät zu lesen und zu ändern (wie GRBL-Parameter). Dies ist eine erweiterte Funktion und sollte mit Vorsicht verwendet werden.

:::warning
Das Bearbeiten von Geräteeinstellungen kann gefährlich sein und kann deine Maschine unbrauchbar machen, wenn falsche Werte angewendet werden!
:::

---

## Fehlerbehebung bei Verbindungsproblemen

### Gerät nicht gefunden

- **Linux (Seriell)**: Füge deinen Benutzer zur `dialout`-Gruppe hinzu. Dies
  ist für **Snap- und Nicht-Snap-Installationen** auf Debian-basierten
  Distributionen erforderlich, um AppArmor DENIED-Meldungen zu vermeiden:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  Melde dich ab und wieder an, damit die Änderungen wirksam werden.

- **Snap-Paket**: Zusätzlich zur `dialout`-Gruppe oben, stelle sicher, dass
  du serielle Port-Berechtigungen erteilt hast:
  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows**: Überprüfe den Geräte-Manager, um zu bestätigen, dass das Gerät
  erkannt wird, und notiere dir die COM-Port-Nummer.

### Verbindung verweigert

- Überprüfe, ob IP-Adresse und Portnummer korrekt sind
- Stelle sicher, dass deine Maschine eingeschaltet und mit dem Netzwerk verbunden ist
- Überprüfe die Firewall-Einstellungen bei Netzwerkverbindung

### Maschine reagiert nicht

- Versuche eine andere Baudrate (einige Geräte verwenden `9600` oder `57600`)
- Überprüfe auf lockere Kabel oder schlechte Verbindungen
- Schalte deinen Laserschneider aus und wieder ein und versuche es erneut

Weitere Hilfe findest du unter [Verbindungsprobleme](../troubleshooting/connection).

---

**Weiter:** [Schnellstart-Anleitung →](quick-start)
