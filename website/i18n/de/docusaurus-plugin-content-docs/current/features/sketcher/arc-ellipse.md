---
description:
  "Zeichne Bögen und Ellipsen (einschließlich Kreise) im Rayforge-Sketcher, mit Zusatztasten und
  Maß-Eingabe."
---

# Bogen und Ellipse

Der Sketcher bietet zwei Werkzeuge für gekrümmte Formen: das **Bogen-Werkzeug** für kreisförmige
Bögen und das **Ellipsen-Werkzeug** für Ellipsen und Kreise.

![Ein Bogen und eine Ellipse, wie sie von ihren Werkzeugen erstellt werden](/screenshots/addons-sketcher-tool-arc-ellipse.webp)

## Bogen-Werkzeug

Das Bogen-Werkzeug (`G+A`) erstellt einen Bogen in drei Klicks:

1. Klicke auf den **Mittelpunkt**.
2. Klicke auf den **Startpunkt** — sein Abstand zum Mittelpunkt legt den Radius fest.
3. Bewege den Cursor, um den Bogen zwischen den beiden Punkten überstreichen zu lassen, und klicke
   auf die **Endposition**.

Solange die Vorschau aktiv ist, kannst du eine Zahl eingeben, um den Radius exakt festzulegen;
drücke `Tab` oder `Enter`, um sie anzuwenden. `Tab` vor der Eingabe schaltet das magnetische
Einrasten um.

## Ellipsen-Werkzeug

Das Ellipsen-Werkzeug (`G+C`) erstellt Ellipsen und Kreise mit zwei Klicks: Der erste setzt den
Mittelpunkt, der zweite den Randpunkt. Du kannst auch am Mittelpunkt drücken, ziehen und am Rand
loslassen — beide Gesten funktionieren gleichwertig.

- Halte `Ctrl` gedrückt, um die Form auf einen perfekten Kreis zu beschränken.
- Halte `Shift` gedrückt, um den Startpunkt als Mittelpunkt der Ellipse zu verwenden.

## Zwei Klicks oder Ziehen

Wie die [Rechteck](rectangle.md)-Werkzeuge akzeptiert auch das Ellipsen-Werkzeug zwei Gesten
gleichwertig: Klicke den ersten Punkt, bewege und klicke den zweiten Punkt, oder drücke am ersten
Punkt, ziehe und lasse am zweiten los. Ein kurzer Klick ohne Bewegung aktiviert das Werkzeug
lediglich und wartet auf den zweiten Punkt, sodass versehentliche Klicks keine degenerierte
Geometrie hinterlassen. Während eine Vorschau aktiv ist, zeigt die Statusleiste die verfügbaren
Zusatztasten an, und `Esc` bricht die Vorschau ab.
