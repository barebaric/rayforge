---
slug: 5-tips-better-engraving
title: 5 Tipps für bessere Lasergravierergebnisse mit Rayforge
authors: rayforge_team
tags: [engraving, optimization, quality, workflow]
---

![3D Preview](/screenshots/main-3d.png)

Professionelle Lasergravierergebnisse erfordern mehr als nur gute Hardware –
auch deine Software-Einstellungen und dein Workflow sind wichtig. Hier sind fünf
Tipps, die dir helfen, das Beste aus Rayforge herauszuholen.

<!-- truncate -->

## 1. Verwende Overscan für glattere Rastergravur

Bei der Rastergravur ist ein häufiges Problem sichtbare Linien oder
Unregelmäßigkeiten an den Rändern, wo der Laser die Richtung wechselt. Dies
passiert, weil der Laserkopf abbremsen und beschleunigen muss, was die
Gravurqualität beeinträchtigen kann.

**Lösung**: Aktiviere **Overscan** in deinen Raster-Operationseinstellungen.

Overscan verlängert den Fahrweg des Lasers über den tatsächlichen Gravurbereich
hinaus, sodass der Kopf volle Geschwindigkeit erreicht, bevor er in den
Arbeitsbereich eintritt, und diese Geschwindigkeit beibehält. Dies führt zu
einer viel glatteren, gleichmäßigeren Gravur.

So aktivierst du Overscan:

1. Wähle deine Raster-Operation
2. Öffne die Operationseinstellungen
3. Aktiviere "Overscan" und stelle den Abstand ein (typischerweise
   funktionieren 3-5mm gut)

Erfahre mehr in unserem [Overscan-Leitfaden](/docs/features/overscan).

## 2. Optimiere die Fahrzeit mit Pfadsortierung

Bei Kontur-Operationen mit vielen separaten Pfaden kann die Reihenfolge, in der
der Laser jede Form besucht, die Gesamtauftragszeit erheblich beeinflussen.

**Lösung**: Verwende Rayforges integrierte **Fahrzeitoptimierung**.

Rayforge kann Pfade automatisch neu anordnen, um die Nicht-Schneide-Fahrzeit zu
minimieren. Dies ist besonders nützlich für Aufträge mit vielen kleinen Objekten
oder Text mit mehreren Buchstaben.

Die Pfadoptimierung ist typischerweise standardmäßig aktiviert, aber du kannst
sie in den Kontur-Operationseinstellungen überprüfen und anpassen.

## 3. Füge HalteLaschen hinzu, um Teilbewegung zu verhindern

Nichts ist frustrierender als ein fast fertiger Schnittauftrag, der ruiniert wird,
weil sich das Teil im letzten Moment verschoben oder durch das Maschinenbett
gefallen ist.

**Lösung**: Verwende **HalteLaschen**, um Teile an Ort und Stelle zu halten,
bis der Auftrag abgeschlossen ist.

HalteLaschen sind kleine ungeschnittene Bereiche, die dein Teil mit dem
umliegenden Material verbinden. Nach Abschluss des Auftrags kannst du das Teil
einfach entfernen und die Laschen mit einem Messer oder Schleifpapier säubern.

Rayforge unterstützt sowohl manuelle als auch automatische Laschen-Platzierung:

- **Manuell**: Klicke genau dort, wo du Laschen auf der Arbeitsfläche
  möchtest
- **Automatisch**: Gib die Anzahl der Laschen an und lass Rayforge
  sie gleichmäßig verteilen

Schau dir die [HalteLaschen-Dokumentation](/docs/features/holding-tabs)
für eine vollständige Anleitung an.

## 4. Vorschau deines Auftrags in 3D vor dem Ausführen

Eines der wertvollsten Features von Rayforge ist die 3D-G-Code-Vorschau. Es ist
verlockend, diesen Schritt zu überspringen und den Auftrag direkt an die Maschine
zu senden, aber einen Moment zur Vorschau zu nehmen, kann dir Zeit und Material
sparen.

**Worauf du in der Vorschau achten solltest**:

- Überprüfe, ob alle Operationen in der richtigen Reihenfolge ausgeführt
  werden
- Prüfe auf unerwartete Werkzeugwege oder Überschneidungen
- Bestätige, dass Multi-Pass-Operationen die richtige Anzahl von Durchläufen
  haben
- Stelle sicher, dass die Auftraggrenzen in dein Material passen

Um die 3D-Vorschau zu öffnen, klicke nach dem Generieren deines G-Codes auf
die **3D-Vorschau**-Schaltfläche in der Hauptsymbolleiste.

Erfahre mehr über die 3D-Vorschau in unserer
[UI-Dokumentation](/docs/ui/3d-preview).

## 5. Verwende eigene G-Code-Hooks für konsistente Workflows

Wenn du feststellst, dass du vor oder nach jedem Auftrag dieselben G-Code-Befehle
ausführst – wie Referenzfahrt, Einschalten einer Luftunterstützung oder Ausführen
eines Fokus-Routines – kannst du dies mit **G-Code-Makros & Hooks** automatisieren.

**Häufige Anwendungsfälle**:

- **Pre-Job-Hook**: Referenzfahrt der Maschine, Luftunterstützung einschalten,
  Auto-Fokus-Routine ausführen
- **Post-Job-Hook**: Luftunterstützung ausschalten, zur Home-Position zurückkehren,
  einen Abschluss-Sound abspielen
- **Ebenenspezifische Makros**: Fokushöhe zwischen Operationen ändern,
  Lasermodule wechseln

Hooks unterstützen Variablensubstitution, sodass du Auftragseigenschaften wie
Materialstärke, Operationstyp und mehr referenzieren kannst.

Beispiel Pre-Job-Hook:

```gcode
G28 ; Home all axes
M8 ; Turn on air assist
G0 Z{focus_height} ; Move to focus height
```

Siehe unseren [G-Code-Makros & Hooks Leitfaden](/docs/machine/hooks-macros) für
detaillierte Beispiele und Variablenreferenz.

---

## Bonustipp: Teste zuerst auf Restmaterial

Obwohl dies nicht spezifisch für Rayforge ist, ist es wert zu wiederholen: Teste
immer neue Einstellungen, Operationen oder Materialien zuerst auf Reststücken.
Verwende Rayforges Materialprofile und Operationsvoreinstellungen, um deine
getesteten Einstellungen für zukünftige Verwendung zu speichern.

---

*Hast du deine eigenen Rayforge-Tipps und Tricks? Teile sie mit der
Community auf [GitHub Discussions](https://github.com/barebaric/rayforge/discussions)!*
