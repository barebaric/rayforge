# Pfad-Optimierung

Pfad-Optimierung sortiert Schneidesegmente neu, um die Verfahrdistanz zu minimieren. Der Laser bewegt sich effizient zwischen Schnitten, anstatt zufällig über den Arbeitsbereich zu springen.

## Funktionsweise

Ohne Optimierung werden Pfade in der Reihenfolge geschnitten, in der sie in deiner Designdatei erscheinen. Die Optimierung analysiert alle Pfadsegmente und ordnet sie so an, dass der Laser die kürzeste Gesamtdistanz zwischen Schnitten zurücklegt.

**Vor Optimierung:** Laser springt hin und her über das Material  
**Nach Optimierung:** Laser bewegt sich sequenziell von Schnitt zu Schnitt

## Einstellungen

### Optimierung aktivieren

Aktiviert oder deaktiviert die Pfad-Optimierung. Standardmäßig für die meisten Operationen aktiviert.

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

**Zeitersparnis:** Kann die Job-Zeit um 20-50% für Designs mit vielen separaten Schnitten reduzieren.

**Bewegungseffizienz:** Weniger Eilgang-Bewegungen bedeuten weniger Verschleiß an Riemen, Motoren und Lagern.

**Wärmeverteilung:** Optimierte Pfade können Hitze in einem Bereich konzentrieren. Bei wärmeempfindlichen Materialien überlege, ob die Reihenfolge wichtig ist.

:::tip
Optimierung läuft automatisch. Aktiviere sie einfach und die Software erledigt den Rest.
:::

---

## Verwandte Themen

- [Kontur-Schneiden](operations/contour) - Primäre Schneideoperation
- [Halte-Laschen](holding-tabs) - Teile während des Schneidens sichern
