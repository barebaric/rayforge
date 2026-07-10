# Wellenfront

Das wellenfront-adaptive Ausräumen füllt geschlossene Vektorformen mit
konzentrischen Werkzeugpfaden, die sich wie Wellenringe in einem Teich
vom Taschenzentrum nach außen ausdehnen. Die expandierenden Ringe
behandeln innere Inseln automatisch und erzeugen sanfte, kontinuierliche
Werkzeugpfade ohne die scharfen Richtungswechsel des Raster-Scannings.

## Übersicht

Anders als die traditionelle Rastergravur, die in parallelen Linien
vor und zurück fegt, erzeugt Wellenfront konzentrische Durchgänge,
die vom Zentrum jeder Tasche ausstrahlen. Dies ergibt eine gleichmäßige,
wellenartige Oberflächenstruktur, die sich gut für Anwendungen eignet,
bei denen das Füllmuster selbst zum visuellen Ergebnis beiträgt.

Wellenfront-Operationen:

- Füllen geschlossene Vektorformen (Taschen) mit konzentrischen Durchgängen
- Expandieren von der Taschenmitte nach außen
- Routen automatisch um innere Inseln (Löcher innerhalb der Tasche)
- Erzeugen sanfte Werkzeugpfade ohne Richtungsumkehrungen

## Wann Wellenfront verwenden

Wellenfront ist ein alternatives Füllmuster für Taschenbereiche. Seine
konzentrischen Ringe können visuell ansprechender sein als parallele
Rasterlinien, und das expandierende Muster ergänzt natürlich kreisförmige
oder organische Formen.

Verwende wellenfront-adaptives Ausräumen für:

- Füllen von Taschen in Vektordesigns
- Stempel- und Matrizenbau — die Wellenfront räumt die Hintergrundtasche
  aus, während erhabene Merkmale als innere Inseln erhalten bleiben
- Anwendungen, bei denen die Fülltextur im fertigen Stück sichtbar ist

**Verwende Wellenfront nicht für:**

- Schneiden entlang von Umrissen (verwende stattdessen [Kontur](contour))
- Füllen von Bitmap-Bildern (verwende stattdessen [Gravur](engrave))
- Dünne Wandabschnitte, bei denen keine Tasche vorhanden ist

## Eine Wellenfront-Operation erstellen

### Schritt 1: Objekte auswählen

1. Geschlossene Vektorformen auf der Arbeitsfläche importieren oder zeichnen
2. Die Objekte auswählen, die die Taschenbegrenzung definieren
3. Sicherstellen, dass Formen geschlossene Pfade sind

### Schritt 2: Wellenfront-Operation hinzufügen

- **Menü:** Operationen → Wellenfront hinzufügen
- **Rechtsklick:** Kontextmenü → Operation hinzufügen → Wellenfront

### Schritt 3: Einstellungen konfigurieren

Schrittweite und Offset an Material und gewünschte Oberfläche anpassen.

![Wellenfront-Operation Ergebnis](/screenshots/operations-wavefront.png)

## Haupt-Einstellungen

### Schrittweite (Step Over)

Der Abstand zwischen aufeinanderfolgenden Wellenfront-Durchgängen (mm).
Kleinere Werte ergeben dichtere Abdeckung mit mehr Durchgängen und
längeren Jobzeiten. Größere Werte platzieren Durchgänge weiter
auseinander für schnellere Fertigstellung.

**Schrittweite standardmäßig auf Laserpunktgröße** mit einem Bereich von
0,05–50,0 mm.

| Schrittweite | Liniendichte             | Jobzeit    |
| ------------ | ------------------------ | ---------- |
| 0,1 mm       | Dicht, viele Linien      | Langsamste |
| 0,3 mm       | Moderat                  | Mittel     |
| 1,0 mm+      | Spärlich, weniger Linien | Schnell    |

Typische Werte liegen bei 0,1–0,5 mm für die meisten Anwendungen.

### Offset

Zusätzlicher Abstand zur Taschenwand (mm). Erzeugt einen Rand zwischen
dem äußersten Wellenfront-Durchgang und der Begrenzungskontur. Dies ist
nützlich, wenn ein separater [Kontur](contour)-Durchgang die Kante
fertigstellt oder wenn ein bewusster Rand um die Tasche bleiben soll.

Bereich: 0,0–20,0 mm. Standard ist 0,0 (Wellenfront-Durchgänge reichen
bis zur Begrenzung).

## Wie Wellenfront funktioniert

1. **Einstich-Durchgang** — Ein spiralförmiger Einstich taucht in die
   Mitte der Tasche ein, um einen anfänglichen ausgeräumten Bereich zu
   schaffen
2. **Wellenfront-Expansion** — Ausgehend vom freigeräumten Zentrum
   expandieren konzentrische Ringe nach außen. Jeder Ring dehnt sich um
   die konfigurierte Schrittweite über den vorherigen hinaus aus
3. **Insel-Behandlung** — Während die Wellenfront wächst, trifft sie auf
   innere Inseln und routet um diese herum, sodass sie stehen bleiben
4. **Fertigstellung** — Die Expansion wird fortgesetzt, bis der gesamte
   Taschenbereich abgedeckt ist

## Nachbearbeitung

Wellenfront-Operationen unterstützen:

- **[Pfad-Glättung](../smooth.md)** — Gezackte Kanten in den Werkzeugpfaden
  reduzieren
- **[Pfad-Optimierung](../path-optimization.md)** — Verfahrweg zwischen
  Durchgängen minimieren

## Tipps & Best Practices

### Wahl der Schrittweite

- Dichtere Abdeckung (kleine Schrittweite) bedeutet mehr Durchgänge und
  längere Jobzeiten
- Spärliche Abdeckung (große Schrittweite) ist schneller, lässt aber mehr
  Material zwischen den Durchgängen
- Balance zwischen Dichte und Jobzeit für deine Anwendung finden

### Stempel- und Matrizenbau

Wellenfront eignet sich gut für den Stempelbau. Die expandierenden
konzentrischen Ringe räumen natürlich die Hintergrundtasche aus, während
sie um erhabene Merkmale navigieren, die als innere Inseln behandelt
werden.

### Kombination mit Kontur

Ein üblicher Arbeitsablauf ist, das Tascheninnere mit Wellenfront
auszuräumen und dann die Begrenzung mit einem [Kontur](contour)-Durchgang
für eine saubere Kante fertigzustellen. Den Offset so einstellen, dass
genügend Rand für den Konturschnitt bleibt.

## Verwandte Themen

- **[Kontur](contour)** — Schneiden entlang von Vektorumrissen
- **[Gravur](engrave)** — Bereiche mit Rastergravurmustern füllen
- **[Schrumpf-Wicklung](shrink-wrap)** — Begrenzungsschnitt um Objekte
- **[Pfad-Glättung](../smooth.md)** — Werkzeugpfadkanten verfeinern
