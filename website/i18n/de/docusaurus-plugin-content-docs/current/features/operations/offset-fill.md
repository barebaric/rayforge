# Offset-Füllung

Die Offset-Füllung füllt geschlossene Vektorformen mit Ringen, die parallel zu ihrem Umriss
verlaufen. Der erste Ring folgt dem Umriss selbst; jeder weitere Ring liegt einen Linienabstand
weiter innen, bis die Form gefüllt ist. Löcher in einer Form bleiben frei: Die Ringe folgen auch
deren Kanten.

## Übersicht

Während die [Gravur](engrave) in geraden Linien vor und zurück fährt und die
[Wellenfront](wavefront) sich vom Taschenzentrum ausbreitet, arbeitet die Offset-Füllung vom Umriss
nach innen. Jeder Ring ist ein geschlossener Pfad, der entlang seiner Mittellinie geschnitten wird,
sodass die Füllung die Form des Designs annimmt.

Offset-Füllung-Operationen:

- Füllen geschlossene Vektorformen mit Ringen parallel zum Umriss
- Lassen Löcher innerhalb einer Form frei
- Ignorieren offene Pfade
- Schneiden die Ringe von außen nach innen oder von innen nach außen

## Wann Offset-Füllung verwenden

Verwende die Offset-Füllung für:

- Füllen von Text, Logos und anderen Vektorformen mit einer Füllung, die ihren Konturen folgt
- Ausräumen von Taschen, bei denen die Ringe eng an der Formkante liegen sollen
- Designs, bei denen das Füllmuster im fertigen Stück sichtbar ist

**Nicht verwenden für:**

- Nur Schneiden entlang von Umrissen (verwende stattdessen [Kontur](contour))
- Füllen von Bitmap-Bildern (verwende stattdessen [Gravur](engrave))
- Offene Pfade, die keine zu füllende Fläche umschließen

## Eine Offset-Füllung-Operation erstellen

### Schritt 1: Formen vorbereiten

1. Geschlossene Vektorformen importieren oder auf der Arbeitsfläche zeichnen
2. Sicherstellen, dass die Formen geschlossene Pfade sind; offene Pfade werden übersprungen

### Schritt 2: Einen Offset-Füllung-Schritt hinzufügen

1. Öffne den Workflow der Ebene im rechten Panel.
2. Klicke auf die Schaltfläche **Schritt hinzufügen** und wähle **Offset-Füllung**.

### Schritt 3: Einstellungen konfigurieren

Linienabstand, Offset und Füllrichtung an dein Material und das gewünschte Finish anpassen.

## Haupt-Einstellungen

Der Schritt-Einstellungen-Dialog hat drei Registerkarten: **Schritt-Einstellungen**, **Laser** und
**Nachbearbeitung**. Die Einstellungen werden unten in Registerkarten-Reihenfolge beschrieben.

### Offset-Füllung-Einstellungen

Die Gruppe **Offset-Füllung** auf der Registerkarte _Schritt-Einstellungen_ steuert das Füllmuster.

#### Linienabstand (Line Interval)

Der Abstand zwischen benachbarten Füllringen. Kleinere Werte ergeben eine dichtere Füllung mit mehr
Ringen und längeren Jobzeiten; größere Werte sind schneller, lassen aber mehr Platz zwischen den
Ringen.

**Der Linienabstand entspricht standardmäßig der Laserpunktgröße** und hat einen Bereich von
0,01–50,0 mm.

#### Offset

Der Abstand des ersten Rings innerhalb des Umrisses. Beim Standardwert 0,0 liegt der erste Ring auf
dem Umriss; ein größerer Wert lässt einen Rand zwischen Umriss und Füllung. Das ist nützlich, wenn
ein separater [Kontur](contour)-Durchgang die Kante fertigstellt.

Bereich: 0,0–20,0 mm. Standard ist 0,0.

#### Füllrichtung (Fill Direction)

Die Reihenfolge, in der die Ringe geschnitten werden:

- **Außen-Innen** (Standard): Der äußerste Ring wird zuerst geschnitten und die Füllung arbeitet
  sich zur Mitte vor
- **Innen-Außen**: Der innerste Ring wird zuerst und der Umrissring zuletzt geschnitten

### Laser-Einstellungen

Leistung, Geschwindigkeit und Laserkopf-Auswahl befinden sich auf der Seite **Laser** des
Schritt-Einstellungen-Dialogs. Ein neuer Offset-Füllung-Schritt beginnt mit 80 % Leistung.

**Leistung (%):**

- Laserintensität für die Füllung
- An die Anforderungen deines Materials anpassen

**Geschwindigkeit (mm/min):**

- Wie schnell sich der Laser bewegt
- An die Gravurgeschwindigkeit deines Materials anpassen

## Wie Offset-Füllung funktioniert

1. **Geschlossene Formen** — Offene Pfade werden verworfen; jede geschlossene Form wird zusammen mit
   ihren Löchern behandelt
2. **Erster Ring** — Der Umriss, um den eingestellten Offset nach innen verschoben
3. **Weitere Ringe** — Jeder nächste Ring liegt einen Linienabstand innerhalb des vorherigen; Ringe
   um Löcher wachsen gleichzeitig von den Löchern nach außen
4. **Abschluss** — Es werden Ringe hinzugefügt, bis die Form gefüllt ist
5. **Schneiden** — Die Ringe werden entlang ihrer Mittellinie in der gewählten Füllrichtung
   geschnitten

## Nachbearbeitung

Offset-Füllung-Operationen unterstützen:

- **[Pfad-Glättung](../smooth.md)** — Gezackte Kanten in den Werkzeugpfaden reduzieren
  (standardmäßig aus)
- **[Auf Rohmaterial zuschneiden](../crop-to-stock.md)** — Werkzeugpfade auf die Materialgrenze
  beschränken (standardmäßig aus)
- **[Pfad-Optimierung](../path-optimization.md)** — Verfahrweg zwischen den Ringen minimieren
- **[Mehrfach-Durchgang](../multi-pass.md)** — Die Füllung mehrmals wiederholen

## Tipps & Best Practices

### Wahl des Linienabstands

- Beginne mit dem Standardwert (der Laserpunktgröße) für eine geschlossene, gleichmäßige Füllung
- Erhöhe den Abstand für ein sichtbares Ringmuster oder einen schnelleren Job

### Kombination mit Kontur

Räume das Innere einer Form mit der Offset-Füllung aus und stelle die Kante dann mit einem
[Kontur](contour)-Durchgang fertig. Stelle den Offset so ein, dass die Füllung genug Platz für den
Konturschnitt lässt.

## Verwandte Themen

- **[Wellenfront](wavefront)** — Konzentrische Füllung, die sich vom Taschenzentrum ausbreitet
- **[Kontur](contour)** — Schneiden entlang von Vektorumrissen
- **[Gravur](engrave)** — Bereiche mit Rastergravurmustern füllen
- **[Pfad-Glättung](../smooth.md)** — Werkzeugpfadkanten verfeinern
