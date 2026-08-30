---
description:
  "Konfiguriere deinen Laserschneider oder Gravierer zum ersten Mal. Verwende den
  Konfigurations-Assistenten, um deine Maschine zu erstellen, verbinde sie dann und mache dich
  bereit zum Schneiden mit Rayforge."
---

# Ersteinrichtung

Nach der Installation von Rayforge musst du deinen Laserschneider oder Gravierer konfigurieren.
Diese Anleitung führt dich durch die Erstellung deiner ersten Maschine mit dem
Konfigurations-Assistenten und den Aufbau einer Verbindung.

## Schritt 1: Rayforge starten

Starte Rayforge aus deinem Anwendungsmenü oder indem du `rayforge` in einem Terminal ausführst. Beim
ersten Start — wenn noch keine echte Maschine konfiguriert wurde — öffnet sich der
Konfigurations-Assistent automatisch, damit du deine Maschine einrichten kannst, ohne durch Menüs zu
suchen. (Du kannst ihn jederzeit später über **Einstellungen → Maschinen → Add Machine** öffnen.)

## Schritt 2: Eine Maschine mit dem Assistenten erstellen

Navigiere zu **Einstellungen → Maschinen** oder drücke <kbd>ctrl+comma</kbd>, um den
Einstellungsdialog zu öffnen, und wähle dann die Seite **Maschinen**.

![Maschineneinstellungen](/screenshots/app-settings-machines.webp)

Klicke auf **Add Machine**, um die Maschinenauswahl zu öffnen.

![Maschine hinzufügen Dialog](/screenshots/app-settings-machines-add.webp)

### Berechtigungsprüfung

Bevor die Erkennung beginnt, prüft der Assistent, ob Rayforge die seriellen Ports und Kameras
tatsächlich öffnen kann. Wenn ein Gerät vorhanden ist, aber der Zugriff fehlt, erscheint zuerst eine
**Berechtigungsseite**, die erklärt, wie man das auf deiner Plattform behebt:

- **Snap-Installationen**: Erteile die `serial-port`-Schnittstelle (und bei Bedarf die
  Kamera-Schnittstelle) — die genauen Befehle werden mit einem Ein-Klick-Kopierbutton angezeigt.
- **Linux ohne Snap**: Füge deinen Benutzer zur `dialout`-Gruppe hinzu, damit der serielle
  Geräteknoten zugänglich ist.

Sobald der Zugriff gewährleistet ist, fährt der Assistent automatisch fort.

![Assistent — Berechtigungsprüfung](/screenshots/config-wizard-permissions.webp)

### Geräte automatisch entdecken

Der Assistent kann Geräte für dich entdecken, anstatt dass du einen Ausgangspunkt wählen und alles
von Hand ausfüllen musst:

- **USB-Seriellgeräte** werden aufgelistet, sobald sie erscheinen.
- **Netzwerkgeräte** werden über mDNS entdeckt: OctoPrint-Server und ESP3D-Boards erscheinen neben
  den USB-Seriellgeräten.
- Erkannte Geräte werden **mit eingebauten Profilen abgeglichen**, wenn ein zuverlässiger Treffer
  gefunden wird, sodass du oft nur die erkannten Einstellungen bestätigen musst, anstatt sie
  einzugeben.
- GRBL wählt automatisch den korrekten G-Code-Dialekt aus den Compile-Flags der Firmware, und
  OctoPrint/Smoothieware werden über das Netzwerk abgefragt.
- Geräte, die du bereits konfiguriert hast, werden als **nur lesen** angezeigt, damit du
  versehentlich keine Duplizate erstellst.

Klicke auf ein erkanntes Gerät, um den Assistenten vorzufüllen, oder wähle manuell einen
Ausgangspunkt wie unten beschrieben.

Der Konfigurations-Assistent passt die angezeigten Schritte an deine Auswahl an:

- Die Auswahl eines **eingebauten Profils** füllt Controller, Arbeitsbereich und Kopf vor — der
  Assistent springt direkt zu den Schritten Rotationsmodul, Kameras und Überprüfung
- Das **Importieren eines Profils** behält die Hardware- und Kopf-Schritte bei, damit du korrigieren
  kannst, was der Import falsch gemacht hat
- **Device Not Listed** führt dich durch jeden Schritt, einschließlich der Schritte Controller und
  KI-Spezifikationsabfrage

### Ausgangspunkt wählen

Wähle ein eingebautes Geräteprofil, um Controller-, Arbeitsbereichs- und Kopfeinstellungen
vorzufüllen, oder klicke auf **Device Not Listed**, um alles manuell zu konfigurieren. Du kannst
auch ein zuvor exportiertes Profil oder ein LightBurn-Geräteprofil (.lbdev) mit Kamerakalibrierung
und Lasereinstellungen über **Import from File…** importieren.

![Assistent — Ausgangspunkt wählen](/screenshots/config-wizard-profile.webp)

### Einen Controller wählen

Wähle die Firmware- oder Protokollfamilie, die zur Controller-Platine deiner Maschine passt (GRBL,
Marlin, Smoothie, Ruida, OctoPrint, …). Wähle **None — G-code export only**, wenn du G-Code nur in
Dateien exportieren und niemals eine physische Maschine ansteuern möchtest. Dieser Schritt wird
übersprungen, wenn du von einem eingebauten Profil oder einem Import startest.

![Assistent — Einen Controller wählen](/screenshots/config-wizard-controller.webp)

### Verbindung

Gib die Verbindungsparameter ein, die deine Maschine benötigt. Die genauen Felder hängen von dem von
dir gewählten Controller ab:

- **Serielle Treiber** — USB-Gerätepfad (z.B. `/dev/ttyUSB0` unter Linux, `COM3` unter Windows) und
  Baudrate
- **Netzwerktreiber** — Hostadresse und Port (z.B. `192.168.1.100`)
- **OctoPrint** — Server-URL und API-Schlüssel

![Assistent — Verbindung](/screenshots/config-wizard-connect.webp)

### Gerät entdecken

Wenn dein Controller es unterstützt, bietet der Assistent an, sich mit dem Gerät zu verbinden und
seine Konfiguration automatisch auszulesen — Arbeitsbereich, Geschwindigkeiten, Beschleunigung und
Firmware-Fähigkeiten. Dies funktioniert über USB-Seriell **und über das Netzwerk** (mDNS-Erkennung
für OctoPrint und ESP3D). Klicke auf **Probe Now**, um diese Werte automatisch zu erkennen, oder
verwende **Next**, um sie in den folgenden Schritten von Hand einzugeben.

![Assistent — Gerät entdecken](/screenshots/config-wizard-probe.webp)

### KI-Anbieter

Wird nur angezeigt, wenn noch kein KI-Anbieter konfiguriert ist. Gib einen OpenAI-kompatiblen
Endpunkt ein (Basis-URL und API-Schlüssel), damit der nächste Schritt Spezifikationen für bekannte
kommerzielle Maschinen nachschlagen kann. Überspringe diesen Schritt, um die Werte von Hand
einzugeben.

![Assistent — KI-Anbieter](/screenshots/config-wizard-ai-provider.webp)

### KI-Spezifikationsabfrage

Wenn deine Maschine ein bekanntes kommerzielles Modell ist, kann die KI ihre Spezifikationen aus der
Dokumentation des Herstellers vorfüllen. Gib Hersteller und Modell ein und klicke dann auf **Look Up
Specs**. Vorgeschlagene Werte erscheinen als Schalter-Zeilen und sind zunächst aktiviert —
deaktiviere alles, was du nicht angewendet haben möchtest.

![Assistent — KI-Spezifikationsabfrage](/screenshots/config-wizard-ai-lookup.webp)

### Hardware

Konfiguriere das physische Setup der Maschine:

- **Achsen** — X/Y-Achsenbereiche und die Ecke des Koordinatenursprungs (0,0)
- **Achsenrichtung** — kehre eine Achse um, wenn Koordinaten negativ ausfallen
- **Z-Achse** — ob die Maschine eine Z-Achse hat (Fokusmotor, verfahrbarer Tisch); wenn nicht
  vorhanden, werden keine Z-Bewegungen generiert und die 3D-Ansicht schichtet den Inhalt auf der
  Gravurebene
- **Panel-Ausrichtung** — drehe die flache Arbeitsfläche, wie sie auf dem Bildschirm dargestellt
  wird (Native, Nach links drehen, Nach rechts drehen); Rotationsschichten erfordern Native
- **Arbeitsbereich** — Ränder um den unbenutzbaren Raum der Arbeitsfläche
- **Software-Limits** — optionale Sicherheitsgrenzen für das Verfahren
- **Geschwindigkeiten** — maximale Eilganggeschwindigkeit, maximale Schnittgeschwindigkeit und
  Beschleunigung
- **Verhalten** — Referenzfahrt beim Start und Einzelachsen-Homing

![Assistent — Hardware](/screenshots/config-wizard-hardware.webp)

### Kopf

Lege fest, was am Portal befestigt ist — ein Laser- oder ein Spindelkopf — und stelle seine
Parameter ein. Für einen Laser: maximale Leistung (S-Wert), Punktgröße, PWM-Frequenz und
Fokusabstand. Für eine Spindel: max. und min. Drehzahl.

![Assistent — Kopf](/screenshots/config-wizard-head.webp)

### Rotationsmodul

Richte optional einen Rotationsvorsatz ein: Typ (Futter oder Rollen), Achse (A/B/C), Modus (echte 4.
Achse vs. Achsenersetzung), Geometrie und das Flag für die umgekehrte Richtung. Überspringe diesen
Schritt, um ein Rotationsmodul später aus den Maschineneinstellungen hinzuzufügen.

![Assistent — Rotationsmodul](/screenshots/config-wizard-rotary.webp)

### Kameras

Aktiviere optional alle Kameras, die du für Vorschau und Ausrichtung verwenden möchtest. Wenn du
eine Kamera aktivierst und fortfährst, öffnet sich der
[Kamera-Assistent](../machine/camera.md#schritt-2-kamera-assistent), der dich durch
Bildeinstellungen, Linsenkalibrierung und Bildausrichtung führt. Du kannst dies überspringen und
Kameras später über die Kamera-Einstellungen der Maschine einrichten.

![Assistent — Kameras](/screenshots/config-wizard-camera.webp)

### Überprüfen & Benennen

Gib der Maschine einen Namen und überprüfe eine Zusammenfassung von allem, was du konfiguriert hast
— Treiber, Verbindung, Arbeitsbereich, Geschwindigkeiten, Köpfe, Rotationsmodule und Kameras. Der
Assistent zeigt auch Warnungen an, z.B. einen fehlenden Treiber oder einen nicht gesetzten
Arbeitsbereich.

![Assistent — Überprüfen & Benennen](/screenshots/config-wizard-review.webp)

Klicke auf **Create Machine**, um abzuschließen. Der Maschineneinstellungen-Dialog öffnet sich für
deine neue Maschine, wo du alle Einstellungen anpassen kannst, die der Assistent vorausgefüllt hat.
Details findest du auf den Seiten [Maschineneinrichtung](../machine/general.md).

## Schritt 3: Automatische Verbindung

Rayforge verbindet sich automatisch mit deiner Maschine, wenn die Anwendung startet (wenn die
Maschine eingeschaltet und verbunden ist). Du musst nicht manuell auf eine Verbindungstaste klicken.

Der Verbindungsstatus wird in der unteren linken Ecke des Hauptfensters mit einem Statussymbol und
einer Beschriftung angezeigt, die den aktuellen Zustand zeigt (Verbunden, Verbinden, Getrennt,
Fehler usw.).

<!-- prettier-ignore-start -->
:::success[Verbunden!]
Wenn deine Maschine den Status "Verbunden" anzeigt, bist du bereit, Rayforge
zu verwenden!
:::
<!-- prettier-ignore-end -->

---

## Fehlerbehebung bei Verbindungsproblemen

### Gerät nicht gefunden

- **Linux (Seriell)**: Füge deinen Benutzer zur `dialout`-Gruppe hinzu. Dies ist für **Snap- und
  Nicht-Snap-Installationen** auf Debian-basierten Distributionen erforderlich, um AppArmor
  DENIED-Meldungen zu vermeiden:

  ```bash
  sudo usermod -a -G dialout $USER
  ```

  Melde dich ab und wieder an, damit die Änderungen wirksam werden.

- **Snap-Paket**: Zusätzlich zur `dialout`-Gruppe oben, stelle sicher, dass du serielle
  Port-Berechtigungen erteilt hast:

  ```bash
  sudo snap connect rayforge:serial-port
  ```

- **Windows**: Überprüfe den Geräte-Manager, um zu bestätigen, dass das Gerät erkannt wird, und
  notiere dir die COM-Port-Nummer.

### Verbindung verweigert

- Überprüfe, ob IP-Adresse und Portnummer korrekt sind
- Stelle sicher, dass deine Maschine eingeschaltet und mit dem Netzwerk verbunden ist
- Überprüfe die Firewall-Einstellungen bei Netzwerkverbindung

### Maschine reagiert nicht

- Versuche eine andere Baudrate (einige Geräte verwenden `9600` oder `57600`)
- Überprüfe auf lockere Kabel oder schlechte Verbindungen
- Schalte deinen Laserschneider aus und wieder ein und versuche es erneut

Weitere Hilfe findest du unter [Verbindungsprobleme](../troubleshooting/connection.md).

---

**Weiter:** [Schnellstart-Anleitung →](quick-start)
