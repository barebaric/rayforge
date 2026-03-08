# Pfad-Glättung

Pfad-Glättung reduziert gezackte Kanten und scharfe Übergänge in deinen Schneidepfaden, was zu saubereren Kurven und sanfteren Maschinenbewegungen führt.

## Funktionsweise

Die Glättung wendet einen Filter auf deine Pfadgeometrie an, der spitze Ecken abrundet und raue Kanten glättet. Der Laser folgt einer sanfteren Bahn, anstatt abrupte Richtungsänderungen zu machen.

## Einstellungen

### Glättung aktivieren

Aktiviert oder deaktiviert die Glättung für diese Operation. Die Glättung ist standardmäßig deaktiviert.

### Glätte

Steuert, wie stark der Pfad geglättet wird (0-100). Höhere Werte erzeugen rundere Kurven, können aber mehr vom ursprünglichen Pfad abweichen.

- **Niedrig (0-30):** Minimale Glättung, erhält scharfe Details
- **Mittel (30-60):** Ausgewogene Glättung für die meisten Designs
- **Hoch (60-100):** Starke Glättung, am besten für organische Formen

### Eckenwinkel-Schwellenwert

Winkel, die schärfer als dieser Wert sind, werden als Ecken erhalten, anstatt geglättet zu werden (0-179 Grad). Dies verhindert, dass wichtige scharfe Merkmale abgerundet werden.

- **Niedrigere Werte:** Mehr Ecken werden geglättet, runderes Ergebnis
- **Höhere Werte:** Mehr Ecken bleiben erhalten, schärferes Ergebnis

## Wann Glättung verwenden

**Gut für:**

- Aus Pixel-Quellen importierte Designs mit Stufeneffekt
- Reduzierung mechanischer Belastung bei schnellen Richtungsänderungen
- Verbesserung der Schnittqualität bei Kurven
- Designs mit vielen kleinen Liniensegmenten

**Nicht nötig für:**

- Saubere Vektorgrafiken mit glatten Bézier-Kurven
- Designs, bei denen scharfe Ecken exakt erhalten bleiben müssen
- Technische Zeichnungen, die präzise Geometrie erfordern

---

## Verwandte Themen

- [Kontur-Schneiden](operations/contour) - Primäre Schneideoperation
- [Pfad-Optimierung](path-optimization) - Verfahrdistanz reduzieren
