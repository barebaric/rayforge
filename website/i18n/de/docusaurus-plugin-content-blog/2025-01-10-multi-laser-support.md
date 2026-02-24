---
slug: multi-laser-support
title: Multi-Laser-Unterstützung - Wähle verschiedene Laser für jede Operation
authors: rayforge_team
tags: [multi-laser, operations, workflow]
---

![Camera Overlay](/images/camera-overlay.png)

Eine der leistungsstärksten Funktionen in Rayforge ist die Möglichkeit, verschiedene
Laser unterschiedlichen Operationen innerhalb eines einzelnen Auftrags zuzuweisen.
Dies eröffnet spannende Möglichkeiten für Multi-Werkzeug-Setups und spezialisierte
Workflows.

<!-- truncate -->

## Was ist Multi-Laser-Unterstützung?

Wenn deine Maschine mit mehreren Lasermodulen ausgestattet ist – sagen wir, ein
Diodenlaser zum Gravieren und ein CO2-Laser zum Schneiden, oder unterschiedliche
Leistungs-Diodenlaser, optimiert für verschiedene Materialien – ermöglicht dir
Rayforge, dieses Setup voll auszuschöpfen.

Mit Multi-Laser-Unterstützung kannst du:

- **Verschiedene Laser verschiedenen Operationen zuweisen** in deinem Auftrag
- **Zwischen Lasermodulen wechseln** automatisch während der Auftragsausführung
- **Für Material und Aufgabe optimieren** durch Verwendung des richtigen Werkzeugs
  für jede Operation

## Anwendungsfälle

### Hybrid-Gravur und -Schnitt

Stell dir vor, du arbeitest an einem Holzzeichen-Projekt:

1. **Operation 1**: Verwende einen Niederleistungs-Diodenlaser, um feinen Text
   und detaillierte Grafiken zu gravieren
2. **Operation 2**: Wechsle zu einem leistungsstärkeren CO2-Laser, um die
   Zeichenform auszuschneiden

Mit Rayforge weist du einfach jede Operation dem entsprechenden Laser in deinem
Maschinenprofil zu, und die Software erledigt den Rest.

### Materialspezifische Optimierung

Verschiedene Lasertypen eignen sich für verschiedene Materialien:

- **Diodenlaser**: Großartig für Holzgravur, Leder und einige Kunststoffe
- **CO2-Laser**: Ausgezeichnet zum Schneiden von Acryl, Holz und anderen organischen
  Materialien
- **Faserlaser**: Perfekt zum Markieren von Metallen

Wenn du mehrere Lasertypen auf einem Portal-System hast, ermöglicht dir Rayforges
Multi-Laser-Unterstützung, das optimale Werkzeug für jeden Teil deines Projekts zu
verwenden.

## So richtest du es ein

### 1. Konfiguriere mehrere Laser in deinem Maschinenprofil

Gehe zu **Maschinen-Setup → Mehrere Laser** und definiere jedes Lasermodul
in deinem System. Du kannst angeben:

- Lasertyp und Leistungsbereich
- Offset-Positionen (wenn Laser an verschiedenen Positionen montiert sind)
- Materialkompatibilität

Siehe unseren [Laser-Konfigurationsleitfaden](/docs/machine/laser)
für detaillierte Anweisungen.

### 2. Weise Laser Operationen zu

Beim Erstellen von Operationen in deinem Projekt:

1. Wähle die Operation (Kontur, Raster, etc.)
2. In den Betriebseinstellungen wähle im Dropdown-Menü, welcher Laser
   verwendet werden soll
3. Konfiguriere die für diesen Laser spezifischen Operationsparameter

### 3. Vorschau und Ausführung

Verwende die 3D-Vorschau, um deine Werkzeugwege zu überprüfen, und sende
dann den Auftrag an deine Maschine. Rayforge generiert automatisch die entsprechenden
G-Code-Befehle, um bei Bedarf zwischen Lasern zu wechseln.

## Technische Details

Unter der Haube verwendet Rayforge G-Code-Befehle, um zwischen Lasermodulen zu
wechseln. Die genaue Implementierung hängt von deiner Firmware- und Hardware-
Konfiguration ab, aber gängige Ansätze umfassen:

- **M3/M4 mit Werkzeug-Offsets**: Wechseln zwischen Lasern mit Werkzeugwechsel-
  Befehlen
- **GPIO-Steuerung**: Verwende von der Firmware unterstützte GPIO-Pins, um
  verschiedene Lasermodule zu aktivieren/deaktivieren
- **Eigene Makros**: Definiere Vor- und Nach-Operations-Makros für Laser-
  Wechsel

## Erste Schritte

Multi-Laser-Unterstützung ist in Rayforge 0.15 und neuer verfügbar. Um loszulegen:

1. Aktualisiere auf die neueste Version
2. Konfiguriere dein Maschinenprofil mit mehreren Lasern
3. Probiere es an einem Testprojekt aus!

Schau dir die [Maschinenprofile-Dokumentation](/docs/machine/general)
für weitere Details an.

---

*Hast du ein Multi-Laser-Setup? Wir würden gerne von deiner Erfahrung hören!
Teile deine Projekte und dein Feedback auf
[GitHub](https://github.com/barebaric/rayforge).*
