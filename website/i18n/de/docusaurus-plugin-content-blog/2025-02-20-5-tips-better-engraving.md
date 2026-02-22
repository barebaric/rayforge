---
slug: 5-tips-better-engraving
title: 5 Tipps für bessere Lasergravierergebnisse mit Rayforge
authors: rayforge_team
tags: [engraving, optimization, quality, workflow]
---

![3D Preview](/screenshots/main-3d.png)

Professionelle Lasergravierergebnisse erfordern mehr als nur gute Hardware –
auch Ihre Software-Einstellungen und Ihr Workflow sind wichtig. Hier sind fünf
Tipps, die Ihnen helfen, das Beste aus Rayforge herauszuholen.

<!-- truncate -->

## 1. Verwenden Sie Overscan für glattere Rastergravur

Bei der Rastergravur ist ein häufiges Problem sichtbare Linien oder
Unregelmäßigkeiten an den Rändern, wo der Laser die Richtung wechselt. Dies
passiert, weil der Laserkopf abbremsen und beschleunigen muss, was die
Gravurqualität beeinträchtigen kann.

**Lösung**: Aktivieren Sie **Overscan** in Ihren Raster-Operationseinstellungen.

Overscan verlängert den Fahrweg des Lasers über den tatsächlichen Gravurbereich
hinaus, sodass der Kopf volle Geschwindigkeit erreicht, bevor er in den
Arbeitsbereich eintritt, und diese Geschwindigkeit beibehält. Dies führt zu
einer viel glatteren, gleichmäßigeren Gravur.

So aktivieren Sie Overscan:

1. Wählen Sie Ihre Raster-Operation
2. Öffnen Sie die Operationseinstellungen
3. Aktivieren Sie "Overscan" und stellen Sie den Abstand ein (typischerweise
   funktionieren 3-5mm gut)

Erfahren Sie mehr in unserem [Overscan-Leitfaden](/docs/features/overscan).

## 2. Optimieren Sie die Fahrzeit mit Pfadsortierung

Bei Kontur-Operationen mit vielen separaten Pfaden kann die Reihenfolge, in der
der Laser jede Form besucht, die Gesamtauftragszeit erheblich beeinflussen.

**Lösung**: Verwenden Sie Rayforges integrierte **Fahrzeitoptimierung**.

Rayforge kann Pfade automatisch neu anordnen, um die Nicht-Schneide-Fahrzeit zu
minimieren. Dies ist besonders nützlich für Aufträge mit vielen kleinen Objekten
oder Text mit mehreren Buchstaben.

Die Pfadoptimierung ist typischerweise standardmäßig aktiviert, aber Sie können
sie in den Kontur-Operationseinstellungen überprüfen und anpassen.

## 3. Fügen Sie HalteLaschen hinzu, um Teilbewegung zu verhindern

Nichts ist frustrierender als ein fast fertiger Schnittauftrag, der ruiniert wird,
weil sich das Teil im letzten Moment verschoben oder durch das Maschinenbett
gefallen ist.

**Lösung**: Verwenden Sie **HalteLaschen**, um Teile an Ort und Stelle zu halten,
bis der Auftrag abgeschlossen ist.

HalteLaschen sind kleine ungeschnittene Bereiche, die Ihr Teil mit dem
umliegenden Material verbinden. Nach Abschluss des Auftrags können Sie das Teil
einfach entfernen und die Laschen mit einem Messer oder Schleifpapier säubern.

Rayforge unterstützt sowohl manuelle als auch automatische Laschen-Platzierung:

- **Manuell**: Klicken Sie genau dort, wo Sie Laschen auf der Arbeitsfläche
  möchten
- **Automatisch**: Geben Sie die Anzahl der Laschen an und lassen Sie Rayforge
  sie gleichmäßig verteilen

Schauen Sie sich die [HalteLaschen-Dokumentation](/docs/features/holding-tabs)
für eine vollständige Anleitung an.

## 4. Vorschau Ihres Auftrags in 3D vor dem Ausführen

Eines der wertvollsten Features von Rayforge ist die 3D-G-Code-Vorschau. Es ist
verlockend, diesen Schritt zu überspringen und den Auftrag direkt an die Maschine
zu senden, aber einen Moment zur Vorschau zu nehmen, kann Ihnen Zeit und Material
sparen.

**Worauf Sie in der Vorschau achten sollten**:

- Überprüfen Sie, ob alle Operationen in der richtigen Reihenfolge ausgeführt
  werden
- Prüfen Sie auf unerwartete Werkzeugwege oder Überschneidungen
- Bestätigen Sie, dass Multi-Pass-Operationen die richtige Anzahl von Durchläufen
  haben
- Stellen Sie sicher, dass die Auftraggrenzen in Ihr Material passen

Um die 3D-Vorschau zu öffnen, klicken Sie nach dem Generieren Ihres G-Codes auf
die **3D-Vorschau**-Schaltfläche in der Hauptsymbolleiste.

Erfahren Sie mehr über die 3D-Vorschau in unserer
[UI-Dokumentation](/docs/ui/3d-preview).

## 5. Verwenden Sie eigene G-Code-Hooks für konsistente Workflows

Wenn Sie feststellen, dass Sie vor oder nach jedem Auftrag dieselben G-Code-Befehle
ausführen – wie Referenzfahrt, Einschalten einer Luftunterstützung oder Ausführen
eines Fokus-Routines – können Sie dies mit **G-Code-Makros & Hooks** automatisieren.

**Häufige Anwendungsfälle**:

- **Pre-Job-Hook**: Referenzfahrt der Maschine, Luftunterstützung einschalten,
  Auto-Fokus-Routine ausführen
- **Post-Job-Hook**: Luftunterstützung ausschalten, zur Home-Position zurückkehren,
  einen Abschluss-Sound abspielen
- **Ebenenspezifische Makros**: Fokushöhe zwischen Operationen ändern,
  Lasermodule wechseln

Hooks unterstützen Variablensubstitution, sodass Sie Auftragseigenschaften wie
Materialstärke, Operationstyp und mehr referenzieren können.

Beispiel Pre-Job-Hook:

```gcode
G28 ; Home all axes
M8 ; Turn on air assist
G0 Z{focus_height} ; Move to focus height
```

Siehe unseren [G-Code-Makros & Hooks Leitfaden](/docs/machine/hooks-macros) für
detaillierte Beispiele und Variablenreferenz.

---

## Bonustipp: Testen Sie zuerst auf Restmaterial

Obwohl dies nicht spezifisch für Rayforge ist, ist es wert zu wiederholen: Testen
Sie immer neue Einstellungen, Operationen oder Materialien zuerst auf Reststücken.
Verwenden Sie Rayforges Materialprofile und Operationsvoreinstellungen, um Ihre
getesteten Einstellungen für zukünftige Verwendung zu speichern.

---

*Haben Sie Ihre eigenen Rayforge-Tipps und Tricks? Teilen Sie sie mit der
Community auf [GitHub Discussions](https://github.com/barebaric/rayforge/discussions)!*
