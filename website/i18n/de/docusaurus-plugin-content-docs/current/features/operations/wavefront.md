# Wellenfront

Das wellenfront-adaptive Ausräumen füllt geschlossene Vektorformen mit
konzentrischen Werkzeugpfaden, die sich wie Wellenringe in einem Teich
vom Taschenzentrum nach außen ausdehnen. Die expandierenden Ringe
behandeln innere Inseln automatisch und erzeugen sanfte, kontinuierliche
Werkzeugpfade ohne die scharfen Richtungswechsel des Raster-Scannings.

## Übersicht

Anders als die traditionelle Rastergravur, die in parallelen Linien vor
und zurück fegt, erzeugt Wellenfront konzentrische Durchgänge, die vom
Zentrum jeder Tasche ausstrahlen. Dies ergibt eine gleichmäßige,
wellenartige Oberfläche, die sich gut für Anwendungen eignet, bei denen
das Füllmuster selbst zum visuellen Ergebnis beiträgt.

Wellenfront-Operationen:

- Füllen geschlossene Vektorformen (Taschen) mit konzentrischen Durchgängen
- Expandieren vom Taschenzentrum nach außen
- Umgehen automatisch innere Inseln (Löcher innerhalb der Tasche)
- Erzeugen sanfte Werkzeugpfade ohne Richtungswechsel

## Wann Wellenfront verwenden

Wellenfront ist ein alternatives Füllmuster für Taschenbereiche. Seine
konzentrischen Ringe können optisch ansprechender sein als parallele
Rasterlinien, und das expandierende Muster ergänzt natürlich kreisförmige
oder organische Formen.

Verwende wellenfront-adaptives Ausräumen für:

- Füllen von Taschen in Vektordesigns
- Stempel- und Matrizenbau — die Wellenfront räumt die Hintergrundtasche
  aus, während erhabene Merkmale als innere Inseln erhalten bleiben
- Anwendungen, bei denen die Füllstruktur im fertigen Stück sichtbar ist

**Verwende Wellenfront nicht für:**

- Schneiden entlang von Umrissen (verwende stattdessen [Kontur](contour))
- Füllen von Bitmap-Bildern (verwende stattdessen [Gravur](engrave))
- Dünne Wandabschnitte ohne vorhandene Tasche

## Eine Wellenfront-Operation erstellen

### Schritt 1: Objekte auswählen

1. Geschlossene Vektorformen auf der Arbeitsfläche importieren oder zeichnen
2. Die Objekte auswählen, die die Taschenbegrenzung definieren
3. Sicherstellen, dass die Formen geschlossene Pfade sind

### Schritt 2: Wellenfront-Operation hinzufügen

- **Menü:** Operationen → Wellenfront hinzufügen
- **Rechtsklick:** Kontextmenü → Operation hinzufügen → Wellenfront

### Schritt 3: Einstellungen konfigurieren

Schrittweite und Offset an dein Material und das gewünschte Finish anpassen.

![Wellenfront-Operation Ergebnis](/screenshots/operations-wavefront.webp)

## Haupt-Einstellungen

Der Schritt-Einstellungen-Dialog hat drei Registerkarten: **Schritt-Einstellungen**, **Laser** und **Nachbearbeitung**. Die Einstellungen werden unten in Registerkarten-Reihenfolge beschrieben.

### Wellenfront-Einstellungen

![Wellenfront-Schritt-Einstellungen](/screenshots/step-settings-wavefront-general.webp)

Die Gruppe **Wellenfront** auf der Registerkarte _Schritt-Einstellungen_ steuert das Füllmuster.

#### Schrittweite

Der Abstand zwischen aufeinanderfolgenden Wellenfront-Durchgängen (mm).
Kleinere Werte ergeben eine dichtere Abdeckung mit mehr Durchgängen und
längeren Jobzeiten. Größere Werte platzieren die Durchgänge weiter
auseinander für eine schnellere Fertigstellung.

**Die Schrittweite ist standardmäßig auf die Laserpunktgröße eingestellt**
und hat einen Bereich von 0,05–50,0 mm.

| Schrittweite | Liniendichte             | Jobzeit    |
| ------------ | ------------------------ | ---------- |
| 0,1 mm       | Dicht, viele Linien      | Langsamste |
| 0,3 mm       | Moderat                  | Mittel     |
| 1,0 mm+      | Spärlich, weniger Linien | Schnell    |

Typische Werte liegen bei 0,1–0,5 mm für die meisten Anwendungen.

#### Offset

Zusätzlicher Abstand zur Taschenwand (mm). Erzeugt einen Rand zwischen
dem äußersten Wellenfront-Durchgang und der Begrenzungskontur. Dies ist
nützlich, wenn ein separater [Kontur](contour)-Durchgang die Kante
fertigstellt oder wenn du einen bewussten Rand um die Tasche lassen
möchtest.

Bereich: 0,0–20,0 mm. Standard ist 0,0 (die Wellenfront-Durchgänge
reichen bis zur Begrenzung).

### Laser-Einstellungen

![Laser-Einstellungen](/screenshots/step-settings-wavefront-laser.webp)

Leistung, Geschwindigkeit und Laserkopf-Auswahl befinden sich auf der Seite **Laser** des Schritt-Einstellungen-Dialogs.

**Leistung (%):**

- Laserintensität zum Schneiden
- An die Schneideanforderungen deines Materials anpassen

**Geschwindigkeit (mm/min):**

- Wie schnell sich der Laser bewegt
- An die Schnittgeschwindigkeit deines Materials anpassen

## Wie Wellenfront funktioniert

1. **Einstich-Durchgang** — Ein spiralförmiger Einstich taucht in die
   Mitte der Tasche ein, um einen anfänglich ausgeräumten Bereich zu
   schaffen
2. **Wellenfront-Expansion** — Ausgehend vom freigeräumten Zentrum
   expandieren konzentrische Ringe nach außen. Jeder Ring dehnt sich um
   die konfigurierte Schrittweite über den vorherigen hinaus aus
3. **Insel-Behandlung** — Während die Wellenfront wächst, trifft sie auf
   innere Inseln und umgeht sie, sodass sie stehen bleiben
4. **Fertigstellung** — Die Expansion wird fortgesetzt, bis der gesamte
   Taschenbereich abgedeckt ist

## Nachbearbeitung

![Wellenfront-Nachbearbeitungseinstellungen](/screenshots/step-settings-wavefront-post.webp)

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
- Balanciere Dichte gegen Jobzeit für deine Anwendung

### Stempel- und Matrizenbau

Wellenfront eignet sich gut für den Stempelbau. Die expandierenden
konzentrischen Ringe räumen natürlich die Hintergrundtasche aus, während
sie um erhabene Merkmale navigieren, die als innere Inseln behandelt
werden.

### Kombination mit Kontur

Ein üblicher Workflow ist es, das Tascheninnere mit Wellenfront
auszuräumen und dann die Begrenzung mit einem [Kontur](contour)-Durchgang
für eine saubere Kante fertigzustellen. Stelle den Offset so ein, dass
genügend Rand für den Konturschnitt bleibt.

## Verwandte Themen

- **[Kontur](contour)** — Schneiden entlang von Vektorumrissen
- **[Gravur](engrave)** — Bereiche mit Rastergravurmustern füllen
- **[Schrumpfumhüllung](shrink-wrap)** — Begrenzungsschnitt um Objekte
- **[Pfad-Glättung](../smooth.md)** — Werkzeugpfadkanten verfeinern
