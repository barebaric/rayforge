---
description: "Organisiere Laser-Aufträge in Ebenen mit unterschiedlichen Einstellungen. Verwalte Schnittreihenfolge, Operationen und Materialien mit dem Multi-Ebenen-System von Rayforge."
---

# Multi-Ebenen-Workflow

![Ebenen-Panel](/screenshots/bottom-panel-layers.png)

Das Multi-Ebenen-System von Rayforge ermöglicht es dir, Aufträge in
separate Verarbeitungsschritte zu organisieren. Jede Ebene ist ein
Container für Werkstücke und hat ihren eigenen Workflow — eine
Abfolge von Schritten, jeder mit unabhängigen Lasereinstellungen.

:::tip Wann du keine mehreren Ebenen brauchst
In vielen Fällen reicht eine einzige Ebene. Jeder Schritt innerhalb
einer Ebene hat eigene Laser-, Leistungs-, Geschwindigkeits- und
andere Einstellungen, sodass du gravieren und konturieren in derselben
Ebene kannst. Separate Ebenen sind nur nötig, wenn du verschiedene
Teile eines Bildes mit unterschiedlichen Einstellungen konturieren
willst oder wenn du unterschiedliche WCS- oder Rotationskonfigurationen
brauchst.
:::

## Ebenen erstellen und verwalten

### Ebene hinzufügen

Klicke auf die **+**-Schaltfläche im Ebenen-Panel. Neue Dokumente
starten mit drei leeren Ebenen.

### Ebenen neu anordnen

Ziehe Ebenen per Drag-and-Drop im Panel, um die Ausführungsreihenfolge
zu ändern. Ebenen werden von links nach rechts verarbeitet. Du kannst
**Mittelklick-Ziehen** verwenden, um in der Ebenenliste zu scrollen.

### Werkstücke neu anordnen

Werkstücke innerhalb einer Ebene können per Drag-and-Drop neu angeordnet
werden, um ihre Z-Reihenfolge zu steuern.

Du kannst mehrere Werkstücke mit **Strg+Klick** auswählen, um einzelne
Elemente umzuschalten, oder **Umschalt+Klick**, um einen Bereich
auszuwählen. Das Ziehen einer Auswahl verschiebt alle ausgewählten
Elemente gleichzeitig.

Ausgewählte Werkstücke werden in der Ebenenspalte hervorgehoben, und die
Auswahl bleibt mit der Canvas synchronisiert.

### Ebene löschen

Wähle die Ebene aus und klicke auf die Löschen-Schaltfläche. Alle
Werkstücke in der Ebene werden entfernt. Du kannst die Löschung bei
Bedarf rückgängig machen.

## Ebenen-Eigenschaften

Jede Ebene hat die folgenden Einstellungen, die über das Zahnrad-Symbol
in der Ebenenspalte erreichbar sind:

- **Name** — wird in der Ebenenüberschrift angezeigt
- **Farbe** — wird zum Rendern der Operationen der Ebene auf der
  Arbeitsfläche verwendet
- **Sichtbarkeit** — das Augensymbol schaltet um, ob die Ebene auf der
  Arbeitsfläche und in Vorschauen angezeigt wird. Ausgeblendete Ebenen
  sind weiterhin im erzeugten G-code enthalten.
- **Koordinatensystem (WCS)** — weist dieser Ebene ein
  Arbeitskoordinatensystem zu. Wenn auf ein bestimmtes WCS (z. B. G54,
  G55) gesetzt, schaltet die Maschine am Anfang der Ebene auf dieses
  Koordinatensystem um. Wähle **Standard**, um stattdessen das globale
  WCS zu verwenden.
- **Rotationsmodus** — aktiviert den Rotationsvorsatz-Modus für diese
  Ebene, sodass du Flachbett- und zylindrische Bearbeitung im selben
  Projekt mischen kannst. Konfiguriere das Rotationsmodul und den
  Objektdurchmesser in den Ebeneneinstellungen.

## Ebenen-Workflows

Jede Ebene hat einen **Workflow** — eine Abfolge von Schritten, die als
Pipeline von Symbolen in der Ebenenspalte angezeigt wird. Jeder Schritt
definiert eine einzelne Operation (z. B. Kontur, Rastergravur) mit
eigenen Laser-, Leistungs-, Geschwindigkeits- und anderen Einstellungen.

Klicke auf einen Schritt, um ihn zu konfigurieren. Verwende die
**+**-Schaltfläche in der Pipeline, um weitere Schritte zu einer Ebene
hinzuzufügen. Schritte können per Drag-and-Drop neu angeordnet werden.

## Vektordatei-Import

Beim Importieren von Vektordateien (SVG, DXF, PDF) bietet der
Import-Dialog drei Möglichkeiten, um Ebenen aus der Quelldatei zu
behandeln:

- **Bestehenden Ebenen zuordnen** — importiert jede Quell-Ebene in die
  entsprechende Dokument-Ebene nach Position
- **Neue Ebenen** — erstellt eine neue Dokument-Ebene für jede
  Quell-Ebene
- **Abflachen** — importiert alles in die aktive Ebene

Bei Verwendung von **Bestehenden Ebenen zuordnen** oder **Neue Ebenen**
zeigt der Dialog eine Liste der Ebenen aus der Quelldatei mit
Schaltern, um auszuwählen, welche importiert werden sollen.

## Werkstücke Ebenen zuordnen

**Drag-and-Drop:** Wähle Werkstück(e) auf der Arbeitsfläche oder im
Ebenen-Panel aus und ziehe sie auf die Zielebene. Mehrfachauswahl mit
Strg+Klick und Umschalt+Klick wird unterstützt, und du kannst Elemente
über Ebenen hinweg ziehen.

**Ausschneiden und Einfügen:** Schneide ein Werkstück aus der aktuellen
Ebene aus (Ctrl+X), wähle die Zielebene und füge es ein (Ctrl+V).

**Kontextmenü:** Rechtsklicke auf ein Werkstück im Ebenen-Tab, um ein
Kontextmenü mit Optionen zu öffnen, um es in eine andere Ebene zu
verschieben, zu löschen oder seine Eigenschaften zu öffnen.

## Ausführungsreihenfolge

Während eines Auftrags werden Ebenen von links nach rechts verarbeitet.
Innerhalb jeder Ebene werden alle Werkstücke verarbeitet, bevor zur
nächsten Ebene gewechselt wird. Der Standard-Workflow ist, zuerst zu
gravieren und zuletzt zu schneiden, damit die Teile während der
Gravur an ihrem Platz bleiben.

## Verwandte Seiten

- [Operationen](./operations/contour) - Operationstypen für
  Ebenen-Workflows
- [Simulationsmodus](./simulation-mode) - Vorschau der
  Multi-Ebenen-Ausführung
- [Makros & Hooks](../machine/hooks-macros) - Ebenenbezogene Hooks
  zur Automatisierung
- [3D-Vorschau](../ui/3d-preview) - Ebenen-Stack visualisieren
- [Asset-Browser](../ui/bottom-panel) - Assets mit Kontextmenüs verwalten
