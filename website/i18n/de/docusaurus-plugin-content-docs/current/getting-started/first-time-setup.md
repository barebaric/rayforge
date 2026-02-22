# Ersteinrichtung

Nach der Installation von Rayforge müssen Sie Ihren Laserschneider oder Gravierer konfigurieren. Diese Anleitung führt Sie durch die Erstellung Ihrer ersten Maschine und den Aufbau einer Verbindung.

## Schritt 1: Rayforge starten

Starten Sie Rayforge aus Ihrem Anwendungsmenü oder indem Sie `rayforge` in einem Terminal ausführen. Sie sollten die Hauptoberfläche mit einer leeren Arbeitsfläche sehen.

## Schritt 2: Eine Maschine erstellen

Navigieren Sie zu **Einstellungen → Maschinen** oder drücken Sie <kbd>ctrl+comma</kbd>, um den Einstellungsdialog zu öffnen, und wählen Sie dann die Seite **Maschinen**.

Klicken Sie auf **Maschine hinzufügen**, um eine neue Maschine zu erstellen. Sie können entweder:

1. **Ein integriertes Profil wählen** - Aus vordefinierten Maschinenvorlagen auswählen
2. **"Benutzerdefiniert" wählen** - Mit einer leeren Konfiguration beginnen

Nach der Auswahl öffnet sich der Maschineneinstellungs-Dialog für Ihre neue Maschine.

![Maschineneinstellungen](/screenshots/application-machines.png)

## Schritt 3: Allgemeine Einstellungen konfigurieren

Die Seite **Allgemein** enthält grundlegende Maschineninformationen, Treiberauswahl und Verbindungseinstellungen.

![Allgemeine Einstellungen](/screenshots/machine-general.png)

### Maschineninformationen

1. **Maschinenname**: Geben Sie Ihrer Maschine einen beschreibenden Namen (z.B. "K40 Laser", "Ortur LM2")

### Treiberauswahl

Wählen Sie den entsprechenden Treiber für Ihr Gerät aus dem Dropdown-Menü:

- **GRBL Seriell** - Für GRBL-Geräte, die über USB/Seriell-Port verbunden sind
- **GRBL Netzwerk** - Für GRBL-Geräte mit WiFi/Ethernet-Konnektivität
- **Smoothie** - Für Smoothieware-basierte Geräte

### Treibereinstellungen

Je nach ausgewähltem Treiber konfigurieren Sie die Verbindungsparameter:

#### GRBL Seriell (USB)

1. **Port**: Wählen Sie Ihr Gerät aus dem Dropdown-Menü (z.B. `/dev/ttyUSB0` unter Linux, `COM3` unter Windows)
2. **Baudrate**: Wählen Sie `115200` (Standard für die meisten GRBL-Geräte)

:::info
Wenn Ihr Gerät nicht in der Liste erscheint, überprüfen Sie, ob es angeschlossen ist und dass Sie über die notwendigen Berechtigungen verfügen. Unter Linux müssen Sie möglicherweise Ihren Benutzer zur `dialout`-Gruppe hinzufügen.
:::

#### GRBL Netzwerk / Smoothie (WiFi/Ethernet)

1. **Host**: Geben Sie die IP-Adresse Ihres Geräts ein (z.B. `192.168.1.100`)
2. **Port**: Geben Sie die Portnummer ein (typischerweise `23` oder `8080`)

### Geschwindigkeiten & Beschleunigung

Diese Einstellungen werden für Job-Zeitschätzung und Pfadoptimierung verwendet:

- **Max. Verfahrgeschwindigkeit**: Maximale schnelle Bewegungsgeschwindigkeit
- **Max. Schnittgeschwindigkeit**: Maximale Schnittgeschwindigkeit
- **Beschleunigung**: Für Zeitschätzungen und Overscan-Berechnungen verwendet

## Schritt 4: Hardware-Einstellungen konfigurieren

Wechseln Sie zum Reiter **Hardware**, um die physischen Abmessungen Ihrer Maschine einzurichten.

![Hardware-Einstellungen](/screenshots/machine-hardware.png)

### Abmessungen

- **Breite**: Geben Sie die maximale Breite Ihres Arbeitsbereichs in Millimetern ein
- **Höhe**: Geben Sie die maximale Höhe Ihres Arbeitsbereichs in Millimetern ein

### Achsen

- **Koordinatenursprung (0,0)**: Wählen Sie, wo sich der Ursprung Ihrer Maschine befindet:
  - Unten links (am häufigsten bei GRBL)
  - Oben links
  - Oben rechts
  - Unten rechts

### Achsen-Offsets (Optional)

Konfigurieren Sie X- und Y-Offsets, wenn Ihre Maschine diese für präzises Positionieren benötigt.

## Schritt 5: Automatische Verbindung

Rayforge verbindet sich automatisch mit Ihrer Maschine, wenn die Anwendung startet (wenn die Maschine eingeschaltet und verbunden ist). Sie müssen nicht manuell auf eine Verbindungstaste klicken.

Der Verbindungsstatus wird in der unteren linken Ecke des Hauptfensters mit einem Statussymbol und einer Beschriftung angezeigt, die den aktuellen Zustand zeigt (Verbunden, Verbinden, Getrennt, Fehler usw.).

:::success Verbunden!
Wenn Ihre Maschine den Status "Verbunden" anzeigt, sind Sie bereit, Rayforge zu verwenden!
:::

## Optional: Erweiterte Konfiguration

### Mehrere Laser

Wenn Ihre Maschine über mehrere Lasermodule verfügt (z.B. Diode und CO2), können Sie diese auf der Seite **Laser** konfigurieren.

![Lasereinstellungen](/screenshots/machine-laser.png)

Siehe [Laser-Konfiguration](../machine/laser) für Details.

### Kamera-Setup

Wenn Sie eine USB-Kamera für Ausrichtung und Positionierung haben, konfigurieren Sie diese auf der Seite **Kamera**.

![Kameraeinstellungen](/screenshots/machine-camera.png)

Siehe [Kamera-Integration](../machine/camera) für Details.

### Geräte-Einstellungen

Die Seite **Gerät** ermöglicht es Ihnen, Firmware-Einstellungen direkt auf Ihrem verbundenen Gerät zu lesen und zu ändern (wie GRBL-Parameter). Dies ist eine erweiterte Funktion und sollte mit Vorsicht verwendet werden.

:::warning
Das Bearbeiten von Geräteeinstellungen kann gefährlich sein und kann Ihre Maschine unbrauchbar machen, wenn falsche Werte angewendet werden!
:::

---

## Fehlerbehebung bei Verbindungsproblemen

### Gerät nicht gefunden

- **Linux (Seriell)**: Fügen Sie Ihren Benutzer zur `dialout`-Gruppe hinzu:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
  Melden Sie sich ab und wieder an, damit die Änderungen wirksam werden.

- **Snap-Paket**: Stellen Sie sicher, dass Sie serielle Port-Berechtigungen erteilt haben:
  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows**: Überprüfen Sie den Geräte-Manager, um zu bestätigen, dass das Gerät erkannt wird, und notieren Sie sich die COM-Port-Nummer.

### Verbindung verweigert

- Überprüfen Sie, ob IP-Adresse und Portnummer korrekt sind
- Stellen Sie sicher, dass Ihre Maschine eingeschaltet und mit dem Netzwerk verbunden ist
- Überprüfen Sie die Firewall-Einstellungen bei Netzwerkverbindung

### Maschine reagiert nicht

- Versuchen Sie eine andere Baudrate (einige Geräte verwenden `9600` oder `57600`)
- Überprüfen Sie auf lockere Kabel oder schlechte Verbindungen
- Schalten Sie Ihren Laserschneider aus und wieder ein und versuchen Sie es erneut

Weitere Hilfe finden Sie unter [Verbindungsprobleme](../troubleshooting/connection).

---

**Weiter:** [Schnellstart-Anleitung →](quick-start)
