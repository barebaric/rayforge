# Pfad-Optimierung

Pfad-Optimierung sortiert Schneidesegmente neu, um die Verfahrdistanz zu minimieren. Der Laser
bewegt sich effizient zwischen Schnitten, anstatt zufällig über den Arbeitsbereich zu springen.

## Funktionsweise

Ohne Optimierung werden Pfade in der Reihenfolge geschnitten, in der sie in deiner Designdatei
erscheinen. Die Optimierung analysiert alle Pfadsegmente und ordnet sie so an, dass der Laser die
kürzeste Gesamtdistanz zwischen Schnitten zurücklegt.

**Vor Optimierung** — der Laser springt hin und her über das Material:

![Die Leerfahrten springen in Zeichnungsreihenfolge zwischen den Schnitten hin und
her](/images/path-optimization-before.webp)

**Nach Optimierung** — der Laser bewegt sich sequenziell von Schnitt zu Schnitt:

![Die Leerfahrten verbinden benachbarte Schnitte](/images/path-optimization-after.webp)

## Einstellungen

![Die Optimierungseinstellungen auf der Nachbearbeitungsseite eines Konturschritts](/screenshots/step-settings-contour-post-optimize.webp)

### Optimierung aktivieren

Aktiviert oder deaktiviert die Pfad-Optimierung. Standardmäßig für die meisten Operationen
aktiviert.

### Besten Startpunkt wählen

Startet jeden geschlossenen Pfad am Eckpunkt, der dem Laserkopf am nächsten liegt, statt dort, wo
der Pfad gezeichnet wurde. Das verkürzt die Leerfahrten. Offene Pfade sowie Pfade mit Ein- und
Ausfahrten, Überschnitt oder Haltestegen behalten ihren Startpunkt. Standardmäßig deaktiviert.

### Ecken bevorzugen

Nur verfügbar, wenn **Besten Startpunkt wählen** aktiviert ist. Geschlossene Pfade beginnen an einer
scharfen Ecke, sofern der Pfad eine hat, damit die Nahtstelle weniger sichtbar ist. Pfade ohne
Ecken, etwa Kreise, beginnen am nächstgelegenen Eckpunkt. Standardmäßig deaktiviert.

Derselbe geschlossene Pfad, gestartet dort, wo er gezeichnet wurde (1), am dem Laserkopf nächsten
Eckpunkt mit **Besten Startpunkt wählen** (2) und an der nächsten scharfen Ecke mit **Ecken
bevorzugen** (3):

![Startpunkte eines geschlossenen Pfads: wie gezeichnet, am nächsten Eckpunkt, an der nächsten
scharfen Ecke](/images/path-optimization-start-point.webp)

## Wann Optimierung verwenden

**Aktivieren für:**

- Designs mit vielen separaten Formen
- Reduzierung der Gesamt-Job-Zeit
- Minimierung des Verschleißes des Bewegungssystems
- Komplexe verschachtelte Layouts

**Deaktivieren für:**

- Designs, bei denen die Schneidereihenfolge wichtig ist (z.B. Innen-Features vor Außen)
- Debugging von Pfadproblemen
- Wenn du vorhersehbare, wiederholbare Ausführungsreihenfolge benötigst

## Auswirkungen auf deinen Job

**Zeitersparnis:** Kann die Job-Zeit um 20-50% für Designs mit vielen separaten Schnitten
reduzieren.

**Bewegungseffizienz:** Weniger Eilgang-Bewegungen bedeuten weniger Verschleiß an Riemen, Motoren
und Lagern.

**Wärmeverteilung:** Optimierte Pfade können Hitze in einem Bereich konzentrieren. Bei
wärmeempfindlichen Materialien überlege, ob die Reihenfolge wichtig ist.

<!-- prettier-ignore-start -->
:::tip
Optimierung läuft automatisch. Aktiviere sie einfach und die Software erledigt den Rest.
:::
<!-- prettier-ignore-end -->

---

## Verwandte Themen

- [Kontur-Schneiden](operations/contour) - Primäre Schneideoperation
- [Halte-Laschen](holding-tabs) - Teile während des Schneidens sichern
