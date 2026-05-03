---
description: "Verwende den Konfigurations-Assistenten, um ein angeschlossenes Gerät automatisch zu erkennen und zu konfigurieren, indem die Firmware-Einstellungen ausgelesen werden."
---

# Konfigurations-Assistent

Der Konfigurations-Assistent erkennt dein Gerät automatisch, indem er sich
damit verbindet und die Firmware-Einstellungen ausliest. Er erstellt in
Sekunden ein vollständig konfiguriertes Maschinenprofil und macht die
manuelle Einrichtung überflüssig.

## Assistenten starten

1. Öffne **Einstellungen → Maschinen** und klicke auf **Add Machine**
2. Klicke im Profil-Auswahldialog auf **Other Device…** unten

Dadurch wird der Assistent geöffnet. Er benötigt **kein** vorhandenes
Geräteprofil — der Assistent erstellt eines von Grund auf, indem er das
angeschlossene Gerät ausliest.

## Verbinden

Die erste Seite fordert dich auf, einen Treiber auszuwählen und
Verbindungsparameter anzugeben.

![Assistent Verbinden](/screenshots/app-settings-machines-wizard-connect.png)

### Treiberauswahl

Wähle den Treiber aus, der zum Controller deines Geräts passt. Es werden
nur Treiber angezeigt, die das Auslesen unterstützen. Typischerweise:

- **GRBL (Seriell)** — USB-verbundene GRBL-Geräte
- **GRBL (Netzwerk)** — WiFi/Ethernet GRBL-Geräte

### Verbindungsparameter

Nach der Auswahl eines Treibers gibst du die Verbindungsdetails ein
(serieller Port, Baudrate, Host usw.). Dies sind dieselben Parameter wie
in den [Allgemeinen Einstellungen](general).

Klicke auf **Next**, um das Auslesen zu starten.

## Erkennen

Der Assistent verbindet sich mit dem Gerät und fragt die Firmware nach
Konfigurationsdaten ab. Dazu gehören:

- Firmware-Version und Build-Info (`$I`)
- Alle Firmware-Einstellungen (`$$`)
- Achs-Verfahrwege, Geschwindigkeiten, Beschleunigung und Laser-Leistungsbereich

Dieser Schritt dauert normalerweise nur wenige Sekunden.

## Überprüfen

Nach erfolgreichem Auslesen zeigt die Überprüfungsseite alle erkannten
Einstellungen. Alles ist vorausgefüllt, kann aber vor dem Erstellen der
Maschine angepasst werden.

![Assistent Überprüfen](/screenshots/app-settings-machines-wizard-review.png)

### Geräteinformationen

Schreibgeschützte Informationen aus der Firmware:

- **Gerätename** — extrahiert aus den Firmware-Build-Informationen
- **Firmware-Version** — z.B. `1.1h`
- **RX-Puffergröße** — serieller Empfangspuffer
- **Bogen-Toleranz** — Toleranz der Bogen-Interpolation der Firmware

### Arbeitsfläche

- **X-Verfahrweg** / **Y-Verfahrweg** — maximale Achs-Verfahrwege in
  Maschineneinheiten, aus den Firmware-Einstellungen `$130` und `$131`

### Geschwindigkeit

- **Max. Verfahrgeschwindigkeit** — aus dem kleineren Wert von `$110` und `$111`
- **Max. Schnittgeschwindigkeit** — standardmäßig gleich der
  Verfahrgeschwindigkeit; bei Bedarf anpassen

### Beschleunigung

- **Beschleunigung** — aus dem kleineren Wert von `$120` und `$121`,
  in Maschineneinheiten pro Sekunde zum Quadrat

### Laser

- **Max. Leistung (S-Wert)** — aus der Firmware-Einstellung `$30`
  (Spindel-Maximum)

### Verhalten

- **Home beim Start** — aktiviert, wenn Firmware-Homing (`$22`) eingeschaltet ist
- **Einachsen-Homing** — aktiviert, wenn die Firmware mit dem `H`-Flag
  kompiliert wurde

### Warnungen

Der Assistent kann Warnungen zu potenziellen Problemen anzeigen, wie:

- Lasermodus nicht aktiviert (`$32=0`)
- Gerät meldet in Zoll (`$13=1`)

## Maschine erstellen

Klicke auf **Create Machine**, um den Vorgang abzuschließen. Der Assistent
gibt das konfigurierte Profil weiter und der normale
Maschinenerstellungsprozess wird fortgesetzt — der
[Maschineneinstellungen-Dialog](general) öffnet sich für weitere Anpassungen.

## Einschränkungen

- Der Assistent funktioniert nur mit Treibern, die das Auslesen unterstützen.
  Wenn dein Treiber nicht aufgeführt ist, verwende stattdessen ein
  vorgefertigtes Profil aus dem Auswahldialog.
- Das Auslesen erfordert, dass das Gerät eingeschaltet und verbunden ist.
- Einige Firmware-Einstellungen sind möglicherweise nicht auf allen Geräten
  lesbar.

## Siehe auch

- [Allgemeine Einstellungen](general) — manuelle Maschinenkonfiguration
- [Geräte-Einstellungen](device) — Firmware-Parameter auf einer bereits
  konfigurierten Maschine lesen und schreiben
- [Maschine hinzufügen](../application-settings/machines) — Übersicht über
  den Maschinenerstellungsprozess
