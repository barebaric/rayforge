---
description:
  "Erstelle ein Materialtest-Raster, um optimale Leistungs- und Geschwindigkeitseinstellungen für
  jedes Material zu finden. Kalibriere deinen Lasercutter systematisch."
---

# Materialtest-Raster

Jedes Material — und oft jede Farbe und Dicke desselben Materials — reagiert unterschiedlich auf
Laserleistung und -geschwindigkeit. Das Materialtest-Raster nimmt der Suche nach der richtigen
Kombination das Rätselraten: Es erzeugt ein Muster aus Testzellen, in denen jede Zelle mit einer
leicht anderen Einstellung graviert oder geschnitten wird — alles in einem einzigen Job. Nach einem
Durchlauf siehst du auf einen Blick, welche Kombination das gewünschte Ergebnis liefert.

Erstelle eines über **Werkzeuge → Materialtest-Raster erstellen**. Rayforge fügt der Arbeitsfläche
ein spezielles Werkstück zusammen mit einer passenden Operation hinzu, und du konfigurierst das
Raster in dessen Einstellungsdialog.

![Materialtest-Raster-Einstellungen](/screenshots/material-test.webp)

## Presets

Der Einstellungsdialog bietet Presets für gängige Lasertypen. Sie füllen einen sinnvollen
Geschwindigkeitsbereich, Leistungsbereich und Testtyp aus, damit du mit einer vernünftigen Basis
starten kannst:

| Preset             | Geschwindigkeitsbereich | Leistungsbereich | Testtyp |
| ------------------ | ----------------------- | ---------------- | ------- |
| **Dioden-Gravur**  | 1000-10000 mm/min       | 10-100%          | Gravur  |
| **Dioden-Schnitt** | 100-5000 mm/min         | 50-100%          | Schnitt |
| **CO2-Gravur**     | 3000-20000 mm/min       | 10-50%           | Gravur  |
| **CO2-Schnitt**    | 1000-20000 mm/min       | 30-100%          | Schnitt |

Ein Preset ist nur ein Startpunkt — jeder Wert bleibt danach anpassbar, und Geschwindigkeitsbereiche
werden automatisch auf das begrenzt, was deine Maschine schafft.

## Raster-Modi

Ein Testraster variiert zwei Parameter gleichzeitig: einen über die Spalten und einen über die
Zeilen. Der Raster-Modus legt fest, welche beiden das sind. **Leistung vs. Geschwindigkeit** ist der
Standard und deckt die häufigste Frage ab — Leistung über die Spalten, Geschwindigkeit entlang der
Zeilen.

**Leistung vs. Durchgänge** und **Geschwindigkeit vs. Durchgänge** halten einen der beiden Parameter
fest und variieren stattdessen die Anzahl der Durchgänge, was beim Schneiden dickerer Materialien
nützlich ist. **Geschwindigkeit vs. Versatz** ist ein spezieller Kalibrierungsmodus für
bidirektionale Gravur: Er variiert den horizontalen Scan-Versatz, damit du zeilenweise
Fehlausrichtung ausgleichen kannst. Da das nur bei Rasterarbeit sinnvoll ist, schaltet die Auswahl
das Raster auf Gravur um und vergrößert den Zeilenabstand, sodass eine Fehlausrichtung leicht zu
erkennen ist. Innerhalb jeder Zeile wird die Leistung zusammen mit der Geschwindigkeit skaliert,
damit alle Zellen visuell vergleichbar bleiben.

## Raster konfigurieren

Der Einstellungsdialog gruppiert die Parameter in drei Bereiche.

Der Bereich **Raster** steuert den Test selbst. Der Testtyp bestimmt, ob jede Zelle den Umriss eines
Quadrats schneidet oder es mit Rasterlinien füllt. Die Rasterabmessungen legen fest, wie viele
Spalten und Zeilen getestet werden — jede Spalte steht für einen Schritt des ersten Parameters des
Modus und jede Zeile für einen Schritt des zweiten, vom Minimum bis zum Maximum des eingegebenen
Bereichs. Erlaubt sind 2 bis 20 Schritte pro Achse; 5×5 ist ein guter Standard. Formgröße (Standard
10 mm) und Abstand (Standard 2 mm) bestimmen, wie groß das Raster wird. Beim Testtyp Gravur steuert
der Zeilenabstand den Abstand zwischen den Scanlinien — kleinere Werte füllen dichter, dauern aber
länger. Lass ihn auf null stehen, um die Spotgröße deines Lasers zu verwenden, was für die meisten
Gravurarbeiten gut passt.

Der Bereich **Beschriftungen** steuert die Anmerkungen, die neben dem Raster graviert werden.
Beschriftungen sind standardmäßig aktiviert und werden zuerst graviert, damit das Testmuster sie
nicht verdecken kann. Sie haben ihre eigene Leistung (Standard 10%) und Geschwindigkeit (Standard
1000 mm/min), und Geschwindigkeitswerte werden in deiner bevorzugten Anzeigeeinheit angezeigt.

Der Bereich **Parameter** enthält die Bereiche, die das Raster variiert — Geschwindigkeit, Leistung,
Durchgänge oder Versatz, je nach gewähltem Modus. Modi, die einen Parameter festhalten (zum Beispiel
die Geschwindigkeit bei Leistung vs. Durchgänge), lassen diese Konstante hier ebenfalls einstellen.

## Das Raster-Layout verstehen

Im Standardmodus Leistung vs. Geschwindigkeit nimmt die Leistung von links nach rechts zu und die
Geschwindigkeit von oben nach unten:

```
                   Leistung (%)
                 10       55       100
Geschw.    100  [  ]     [  ]     [  ]
(mm/min)   300  [  ]     [  ]     [  ]
           500  [  ]     [  ]     [  ]
```

Beschriftungen an der linken und oberen Kante zeigen den exakten Wert jeder Zeile und Spalte, sodass
du Zellen nie zählen musst.

Die Gesamtgröße ergibt sich direkt aus den Rasterabmessungen: Jede Achse misst _Schritte × Formgröße
plus (Schritte − 1) × Abstand_, hinzu kommt Platz für die Beschriftungen links und oben (höchstens
15 mm, und nur wenn Beschriftungen aktiviert sind). Ein 5×5-Raster aus 20-mm-Quadraten mit 5 mm
Abstand ist ohne Beschriftungen 120 mm × 120 mm groß, mit Beschriftungen 135 mm × 135 mm.

## Wie das Raster ausgeführt wird {#how-the-grid-runs}

Zellen werden absichtlich **nicht** in Lesereihenfolge ausgeführt. Rayforge führt sie in einer
risikooptimierten Reihenfolge aus: die höchste Geschwindigkeit zuerst, innerhalb jeder
Geschwindigkeit die niedrigste Leistung und innerhalb jeder Leistung die wenigsten Durchgänge.
Langsame, leistungsstarke Kombinationen sind diejenigen, die am ehesten das Material verrußen lassen
oder einen Brand auslösen, daher laufen sie zuletzt. Diese Reihenfolge ist beabsichtigt und kann
nicht geändert werden.

## Den Test ausführen

Lege das Material ein, das du charakterisieren möchtest — Reststücke, nicht dein finales Werkstück —
und fokussiere den Laser wie bei einem echten Job, da die Fokusdistanz das Ergebnis verändert.
Starte den Job und bleib bei der Maschine: Wenn eine Zelle stark verrußt oder übermäßig raucht,
brich den Job ab, statt ihn zu Ende laufen zu lassen.

Wenn der Test abgeschlossen ist, untersuche jede Zelle. Wenn die Gravur zu hell ausfällt, geh in
Richtung mehr Leistung oder langsamerer Geschwindigkeit; wenn sie zu dunkel oder verrußt ausfällt,
geh in Richtung weniger Leistung oder höherer Geschwindigkeit. Bei Schnitttests such die Zelle, die
sauber durchschneidet und dabei am wenigsten verrußt. Um den Sweet Spot einzugrenzen, führe ein
zweites, feineres Raster aus: Wenn ein grobes 5×5-Testraster seine beste Zelle bei etwa 40% Leistung
und 4000 mm/min gefunden hat, grenzt ein nachfolgendes Raster mit 35-45% und 3000-5000 mm/min sie
genau ein.

<!-- prettier-ignore-start -->
:::tip[Als Rezept speichern]
Statt ein Notizbuch mit Gewinner-Einstellungen zu führen, speichere sie als
[Rezept](../../application-settings/recipes.md): Benenne es (zum Beispiel „3 mm Sperrholz
Schnitt"), binde es an die Maschine, die Operation, das Material und die Dicke, die du getestet
hast, und Rayforge schlägt genau diese Einstellungen vor, wenn du dasselbe Material das nächste Mal
schneidest.
:::
<!-- prettier-ignore-end -->

## Erweiterte Verwendung

Materialtest-Raster sind gewöhnliche Werkstücke, daher lassen sie sich frei mit anderen Operationen
kombinieren. Ein häufiges Muster ist, eine Kontur-Operation um das fertige Raster zu legen und das
Teststück nach Abschluss der Gravur aus dem Rohmaterial zu schneiden.

Dieselbe Rasterkonfiguration auf verschiedenen Materialien auszuführen, ist ein schneller Weg, um
eine Bibliothek erprobter Einstellungen aufzubauen — und Rezepte machen diese Bibliothek später nach
Material und Dicke durchsuchbar.

## Tipps & Best Practices

Ein paar Gewohnheiten machen Testergebnisse verlässlicher:

- Beginne mit einem Preset und passe von dort an, statt von Grund auf zu konfigurieren.
- Gib den Zellen etwas Raum: Quadrate von 15-20 mm sind viel leichter zu beurteilen als winzige.
- Ändere immer nur eine Variable, wenn du eingrenzt — ein feines Raster, das beide Achsen weit
  variiert, ist schwer zu interpretieren.
- Lass das Material zwischen aufeinanderfolgenden Tests auf demselben Stück abkühlen.
- Verwende für jeden Test dieselbe Fokusdistanz, auch für den finalen Job.

Und die üblichen Laser-Sicherheitsregeln gelten für Testraster doppelt, die absichtlich unbekanntes
Terrain erkunden:

- Lass einen laufenden Test niemals unbeaufsichtigt.
- Beginne mit konservativen Leistungsbereichen und arbeite dich nach oben.
- Stelle sicher, dass die Rauchabsaugung funktioniert, bevor du beginnst.
- Halte einen Feuerlöscher in Reichweite.

## Fehlerbehebung

**Die Zellen werden in einer seltsamen Reihenfolge ausgeführt.** Das ist die risikooptimierte
Ausführungsreihenfolge, die in [Wie das Raster ausgeführt wird](#how-the-grid-runs) beschrieben wird
— schnellste und schwächste Kombinationen zuerst. Sie ist beabsichtigt.

**Die Ergebnisse sind zwischen Durchläufen inkonsistent.** Stelle sicher, dass das Material flach
aufliegt und befestigt ist, dass der Fokus über das gesamte Raster hinweg identisch ist und dass
dein Netzteil stabile Leistung liefert. Wenn nur ein Bereich des Rasters falsch aussieht, kann das
Material selbst ungleichmäßig sein.

## Verwandte Themen

- **[3D-Vorschau](../../ui/3d-preview.md)** - Testausführung vor dem Starten in der Vorschau ansehen
- **[Rezepte](../../application-settings/recipes.md)** - Testergebnisse automatisch wiederverwenden
- **[Gravur](engrave)** - Gravur-Operationen verstehen
- **[Kontur-Schneiden](contour)** - Schneide-Operationen verstehen
