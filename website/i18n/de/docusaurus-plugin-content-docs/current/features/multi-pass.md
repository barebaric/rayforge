# Mehrfach-Durchgang

Mehrfach-Durchgang wiederholt den Schneide- oder Gravurpfad mehrmals, optional mit Z-Abstieg zwischen den Durchgängen. Dies ist nützlich für dicke Materialien oder um tiefere Gravuren zu erstellen.

## Funktionsweise

Jeder Durchgang verfolgt denselben Pfad erneut. Mit aktiviertem Z-Abstieg bewegt sich der Laser zwischen den Durchgängen näher zum Material und schneidet dabei progressiv tiefer.

## Einstellungen

### Anzahl der Durchgänge

Wie oft der gesamte Schritt wiederholt werden soll (1-100). Jeder Durchgang folgt demselben Pfad.

- **1 Durchgang:** Einzelschnitt (Standard)
- **2-3 Durchgänge:** Üblich für mittel-dicke Materialien
- **4+ Durchgänge:** Sehr dicke oder harte Materialien

### Z-Abstieg pro Durchgang

Distanz, um die Z-Achse zwischen den Durchgängen abzusenken (0-50 mm). Funktioniert nur, wenn deine Maschine Z-Achsen-Steuerung hat.

- **0 mm:** Alle Durchgänge auf gleicher Tiefe (Standard)
- **Materialstärke ÷ Durchgänge:** Progressives Tiefenschneiden
- **Kleine Schritte (0,1-0,5mm):** Feinkontrolle für tiefe Gravur

:::warning Z-Achse erforderlich
Z-Abstieg funktioniert nur mit Maschinen, die über motorisierte Z-Achsen-Steuerung verfügen. Bei Maschinen ohne Z-Achse erfolgen alle Durchgänge auf der gleichen Fokushöhe.
:::

## Wann Mehrfach-Durchgang verwenden

**Dicke Materialien schneiden:**

Mehrere Durchgänge auf der gleichen Tiefe schneiden oft sauberer als ein einzelner langsamer Durchgang. Der erste Durchgang erzeugt eine Schnittfuge, und nachfolgende Durchgänge folgen demselben Pfad effizienter.

**Tiefe Gravur:**

Mit Z-Abstieg kannst du tiefe Relief-Muster oder Gravuren schnitzen, die in einem einzigen Durchgang unmöglich wären.

**Verbesserte Kantenqualität:**

Mehrere schnellere Durchgänge erzeugen oft sauberere Kanten als ein langsamer Durchgang, besonders bei Materialien, die leicht verbrennen.

## Tipps

- Beginne mit 2-3 Durchgängen bei deiner normalen Schnittgeschwindigkeit
- Erhöhe bei dicken Materialien die Durchgänge, anstatt zu verlangsamen
- Aktiviere Z-Abstieg nur, wenn deine Maschine es unterstützt
- Teste auf Abfallmaterial, um die optimale Durchgang-Anzahl zu finden

---

## Verwandte Themen

- [Kontur-Schneiden](operations/contour) - Primäre Schneideoperation
- [Gravur](operations/engrave) - Gravur-Operationen
