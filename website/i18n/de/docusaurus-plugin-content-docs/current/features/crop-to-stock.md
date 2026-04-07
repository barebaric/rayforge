# Auf Rohmaterial zuschneiden

Auf Rohmaterial zuschneiden begrenzt Schneidepfade auf deine Materialgrenze. Alle Schnitte, die über den Rohmaterialbereich hinausreichen, werden abgeschnitten und verhindern, dass der Laser außerhalb deines Materials schneidet.

## Funktionsweise

Der Transformator vergleicht deine Schneidepfade mit der definierten Rohmaterialgrenze. Pfadsegmente außerhalb dieser Grenze werden entfernt oder auf die Materialkante beschnitten.

Wenn in deinem Dokument keine Rohmaterial-Elemente definiert sind, wird stattdessen der Arbeitsbereich der Maschine als Zuschneidegrenze verwendet.

## Einstellungen

### Auf-Rohmaterial-Zuschneiden aktivieren

Aktiviert oder deaktiviert das Zuschneiden. Standardmäßig deaktiviert.

### Versatz

Passt die effektive Rohmaterialgrenze vor dem Zuschneiden an (-100 bis +100 mm).

- **Positive Werte:** Grenze verkleinern (schneidet konservativer)
- **Negative Werte:** Grenze erweitern (erlaubt Schnitte näher an der Kante)
- **0 mm:** Exakte Rohmaterialgrenze verwenden

Verwende Versatz, wenn du einen Sicherheitsabstand zur Materialkante möchtest, oder wenn deine Materialplatzierung nicht perfekt ausgerichtet ist.

## Wann Auf-Rohmaterial-Zuschneiden verwenden

**Teilweise Designs:** Dein Design ist größer als dein Material, aber du möchtest nur den Teil schneiden, der passt.

**Sicherheitsabstand:** Verhindere versehentliche Schnitte über Materialkanten hinaus.

**Verschachtelte Bögen:** Schneide nur die Teile, die auf dein aktuelles Materialstück passen.

**Testschnitte:** Beschränke einen Test auf einen bestimmten Bereich deines Materials.

## Beispiel

Du hast ein großes Design, aber nur ein kleines Stück Material:

1. Definiere deine Rohmaterialgröße passend zu deinem Material
2. Aktiviere Auf-Rohmaterial-Zuschneiden
3. Setze Versatz auf 2mm als Sicherheitsabstand
4. Nur die Teile innerhalb deiner Materialgrenze werden geschnitten

---

## Verwandte Themen

- [Rohmaterial-Handhabung](stock-handling) - Materialgrenzen einrichten
- [Kontur-Schneiden](operations/contour) - Primäre Schneideoperation
