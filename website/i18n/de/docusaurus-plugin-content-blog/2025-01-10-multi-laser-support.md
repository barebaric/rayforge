---
slug: multi-laser-support
title: Multi-Laser-Unterstützung - Wählen Sie verschiedene Laser für jede Operation
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

Wenn Ihre Maschine mit mehreren Lasermodulen ausgestattet ist – sagen wir, ein
Diodenlaser zum Gravieren und ein CO2-Laser zum Schneiden, oder unterschiedliche
Leistungs-Diodenlaser, optimiert für verschiedene Materialien – ermöglicht Ihnen
Rayforge, dieses Setup voll auszuschöpfen.

Mit Multi-Laser-Unterstützung können Sie:

- **Verschiedene Laser verschiedenen Operationen zuweisen** in Ihrem Auftrag
- **Zwischen Lasermodulen wechseln** automatisch während der Auftragsausführung
- **Für Material und Aufgabe optimieren** durch Verwendung des richtigen Werkzeugs
  für jede Operation

## Anwendungsfälle

### Hybrid-Gravur und -Schnitt

Stellen Sie sich vor, Sie arbeiten an einem Holzzeichen-Projekt:

1. **Operation 1**: Verwenden Sie einen Niederleistungs-Diodenlaser, um feinen Text
   und detaillierte Grafiken zu gravieren
2. **Operation 2**: Wechseln Sie zu einem leistungsstärkeren CO2-Laser, um die
   Zeichenform auszuschneiden

Mit Rayforge weisen Sie einfach jede Operation dem entsprechenden Laser in Ihrem
Maschinenprofil zu, und die Software erledigt den Rest.

### Materialspezifische Optimierung

Verschiedene Lasertypen eignen sich für verschiedene Materialien:

- **Diodenlaser**: Großartig für Holzgravur, Leder und einige Kunststoffe
- **CO2-Laser**: Ausgezeichnet zum Schneiden von Acryl, Holz und anderen organischen
  Materialien
- **Faserlaser**: Perfekt zum Markieren von Metallen

Wenn Sie mehrere Lasertypen auf einem Portal-System haben, ermöglicht Ihnen Rayforges
Multi-Laser-Unterstützung, das optimale Werkzeug für jeden Teil Ihres Projekts zu
verwenden.

## So richten Sie es ein

### 1. Konfigurieren Sie mehrere Laser in Ihrem Maschinenprofil

Gehen Sie zu **Maschinen-Setup → Mehrere Laser** und definieren Sie jedes Lasermodul
in Ihrem System. Sie können angeben:

- Lasertyp und Leistungsbereich
- Offset-Positionen (wenn Laser an verschiedenen Positionen montiert sind)
- Materialkompatibilität

Siehe unseren [Laser-Konfigurationsleitfaden](/docs/machine/laser)
für detaillierte Anweisungen.

### 2. Weisen Sie Laser Operationen zu

Beim Erstellen von Operationen in Ihrem Projekt:

1. Wählen Sie die Operation (Kontur, Raster, etc.)
2. In den Betriebseinstellungen wählen Sie im Dropdown-Menü, welcher Laser
   verwendet werden soll
3. Konfigurieren Sie die für diesen Laser spezifischen Operationsparameter

### 3. Vorschau und Ausführung

Verwenden Sie die 3D-Vorschau, um Ihre Werkzeugwege zu überprüfen, und senden Sie
dann den Auftrag an Ihre Maschine. Rayforge generiert automatisch die entsprechenden
G-Code-Befehle, um bei Bedarf zwischen Lasern zu wechseln.

## Technische Details

Unter der Haube verwendet Rayforge G-Code-Befehle, um zwischen Lasermodulen zu
wechseln. Die genaue Implementierung hängt von Ihrer Firmware- und Hardware-
Konfiguration ab, aber gängige Ansätze umfassen:

- **M3/M4 mit Werkzeug-Offsets**: Wechseln zwischen Lasern mit Werkzeugwechsel-
  Befehlen
- **GPIO-Steuerung**: Verwenden Sie von der Firmware unterstützte GPIO-Pins, um
  verschiedene Lasermodule zu aktivieren/deaktivieren
- **Eigene Makros**: Definieren Sie Vor- und Nach-Operations-Makros für Laser-
  Wechsel

## Erste Schritte

Multi-Laser-Unterstützung ist in Rayforge 0.15 und neuer verfügbar. Um loszulegen:

1. Aktualisieren Sie auf die neueste Version
2. Konfigurieren Sie Ihr Maschinenprofil mit mehreren Lasern
3. Probieren Sie es an einem Testprojekt aus!

Schauen Sie sich die [Maschinenprofile-Dokumentation](/docs/machine/general)
für weitere Details an.

---

*Haben Sie ein Multi-Laser-Setup? Wir würden gerne von Ihrer Erfahrung hören!
Teilen Sie Ihre Projekte und Ihr Feedback auf
[GitHub](https://github.com/barebaric/rayforge).*
