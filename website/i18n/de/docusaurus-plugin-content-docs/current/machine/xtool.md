---
description:
  "Verbinde xTool D1- und D1-Pro-Laserschneider sowie den xTool M2 über WLAN mit den experimentellen
  xTool-Treibern von Rayforge."
---

# xTool über WLAN

Rayforge enthält experimentelle Treiber für xTool-Maschinen, die über WLAN verbunden sind. Sie
sprechen die proprietäre Netzwerk-API von xTool statt der GRBL-Telnet- oder FluidNC/ESP3D-
Schnittstellen an, daher muss die Maschine mit ihrem eigenen xTool-Treiber eingerichtet werden.

## Unterstützte Maschinen

| Profil           | Treiber         | Funktionen                                                                                                  |
| ---------------- | --------------- | ----------------------------------------------------------------------------------------------------------- |
| **xTool D1 Pro** | xTool (WLAN)    | Jobs hochladen und starten, Referenzfahrt, Joggen, Pause, Fortsetzen und Abbrechen                          |
| **xTool M2**     | xTool M2 (WLAN) | Maschine überwachen, Referenzfahrt, Joggen und einen vorbereiteten Job pausieren, fortsetzen oder abbrechen |

Beide Treiber sind als **experimentell** gekennzeichnet. Teste sie mit einem Reststück, bevor du sie
für die Produktion einsetzt, und melde Probleme mit angehängtem Debug-Log.

## Eine xTool-Maschine einrichten

1. Verbinde die Maschine mit demselben Netzwerk wie deinen Computer und finde ihre IP-Adresse (zum
   Beispiel in der xTool-App oder in der Geräteliste deines Routers).
2. Füge eine Maschine hinzu und wähle das Geräteprofil **xTool D1 Pro** oder **xTool M2**. Das
   Profil konfiguriert den Treiber und die Maschinenabmessungen bereits.
3. Trage **Hostname oder IP-Adresse** der Maschine ein.
4. Lass die Ports auf ihren Standardwerten, sofern dein Netzwerk nichts anderes verlangt:
   - D1 Pro: HTTP-Port `8080`, WebSocket-Port `8081`
   - M2: Port `28900`
5. Stelle die Verbindung her.

Wenn Rayforge meldet, dass keine xTool-Maschine geantwortet hat, prüfe die IP-Adresse, ob die
Maschine eingeschaltet ist, und ob Computer und Maschine im selben Netzwerk sind.

## D1 und D1 Pro

Der Treiber **xTool (WLAN)** lädt den Job auf die Maschine hoch, startet ihn und verfolgt seinen
Fortschritt über die WebSocket-Job-Ereignisse. Referenzfahrt, Joggen und absolute Bewegungen, Pause,
Fortsetzen und Abbrechen funktionieren wie gewohnt.

Die Netzwerk-API der Maschine meldet ihre Position nicht, daher leitet Rayforge die Positionsanzeige
aus dem Job-Status ab. Die Referenzfahrt ist nur für alle Achsen gleichzeitig verfügbar; die
Referenzfahrt einzelner Achsen ist deaktiviert. Das Schreiben von Firmware-Einstellungen,
Werkstückkoordinaten und Antasten werden von diesem Treiber nicht unterstützt.

## xTool M2

Der Treiber **xTool M2 (WLAN)** verbindet sich mit dem TLS-Dienst des M2 auf Port `28900`. Er
überwacht den Maschinenzustand und unterstützt Referenzfahrt, Joggen sowie Pause, Fortsetzen und
Abbrechen eines bereits auf der Maschine vorbereiteten Jobs.

Die Netzwerk-API des M2 kann noch keine G-Code-Daten empfangen, daher kann Rayforge keine eigenen
Jobs dorthin senden. Bereite den Job in xTool Studio vor, starte ihn dort und nutze Rayforge zur
Überwachung und Steuerung. Z-Bewegungen, Werkzeugwechsel, Fernsteuerung der Laserleistung,
Firmware-Einstellungen und Werkstückkoordinaten werden nicht unterstützt.

## TLS-Zertifikate

Der M2-Treiber prüft das TLS-Zertifikat der Maschine. Wenn die Maschine ein selbstsigniertes
Zertifikat verwendet, aktiviere **Ungültige TLS-Zertifikate erlauben** in den Treibereinstellungen.
Aktiviere dies nur in einem Netzwerk, dem du vertraust.

## Siehe auch

- [Allgemeine Einstellungen](general) – Treiberauswahl, Hostname- und Portfelder
- [Firmware-Referenz](../reference/firmware.md) – Unterstützte Controller und Treiber
- [Remote-Geräte](remote) – Verbindung zu einer Maschine über einen raydriver-Daemon
