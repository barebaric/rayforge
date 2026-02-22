# Schnittbreiten-Kompensation

Schnittbreite ist das Material, das vom Laserstrahl während des Schneidens entfernt wird. Die Schnittbreiten-Kompensation passt Werkzeugwege an, um dies zu berücksichtigen und stellt sicher, dass geschnittene Teile ihren entworfenen Abmessungen entsprechen.

## Was ist Schnittbreite?

**Schnittbreite** = die Breite des durch den Schneideprozess entfernten Materials.

**Beispiel:**
- Laser-Punktgröße: 0.2mm
- Material-Interaktion: fügt ~0.1mm auf jeder Seite hinzu
- **Gesamte Schnittbreite:** ~0.4mm

---

## Wie Schnittbreiten-Kompensation funktioniert

Schnittbreiten-Kompensation **offsetet den Werkzeugweg** nach innen oder außen, um Materialentfernung zu berücksichtigen:

**Für Außen-Schnitte (ein Teil schneiden):**
- Pfad um die Hälfte der Schnittbreite **nach außen** offseten
- Ergebnis: Endteil hat die korrekte Größe

**Für Innen-Schnitte (ein Loch schneiden):**
- Pfad um die Hälfte der Schnittbreite **nach innen** offseten
- Ergebnis: Endloch hat die korrekte Größe

**Beispiel mit 0.4mm Schnittbreite:**

```
Ursprünglicher Pfad:  50mm Quadrat
Kompensation:         Um 0.2mm nach außen offseten (halbe Schnittbreite)
Laser folgt:          50.4mm Quadrat
Nach dem Schneiden:   Teil misst 50.0mm (perfekt!)
```

---

## Schnittbreite messen

**Verfahren zur genauen Schnittbreiten-Messung:**

1. **Eine Testdatei erstellen:**
   - Ein 50mm x 50mm Quadrat zeichnen
   - Einen Kreis zeichnen (beliebige Größe, für Innen-Schnitt-Test)

2. **Den Test schneiden:**
   - Ihre normalen Schneideeinstellungen verwenden
   - Vollsändig durchschneiden
   - Material abkühlen lassen

3. **Messen:**
   - **Äußeres Quadrat (Teil):** Mit Schieblehre messen
     - Wenn < 50mm, wurde Schnittbreite nach außen entfernt
     - Schnittbreite = (50 - gemessen) × 2
   - **Innerer Kreis (Loch):** Durchmesser messen
     - Wenn > entworfener Durchmesser, wurde Schnittbreite nach innen entfernt
     - Schnittbreite = (gemessen - entworfen) / 2

4. **Mitteln:** Den Durchschnitt mehrerer Messungen verwenden

**Variablen, die die Schnittbreite beeinflussen:**
- Laserleistung (höher = breiter)
- Schneidegeschwindigkeit (langsamer = breiter)
- Materialtyp und -dichte
- Fokusdistanz
- Luftunterstützungsdruck

---

## Manuelle Schnittbreiten-Kompensation

Wenn automatische Schnittbreiten-Kompensation nicht verfügbar ist, kompensieren Sie in Ihrer Design-Software:

**Inkscape:**

1. **Den Pfad auswählen**
2. **Pfad → Dynamischer Offset** (Strg+J)
3. **Zum Offset ziehen** um die Hälfte Ihrer Schnittbreiten-Messung
   - Nach außen für Teile (um Pfad größer zu machen)
   - Nach innen für Löcher (um Pfad kleiner zu machen)
4. **Pfad → Objekt zu Pfad** zum Fertigstellen

**Illustrator:**

1. **Den Pfad auswählen**
2. **Objekt → Pfad → Pfad offseten**
3. **Offset-Wert eingeben:** (Schnittbreite / 2)
   - Positiv nach außen, negativ nach innen
4. **OK** zum Anwenden

**Fusion 360 / CAD:**

- Skizzen-Elemente vor dem Export offseten
- Die Schnittbreiten-/Offset-Dimension verwenden

---

## Verwandte Seiten

- [Kontur-Operation](operations/contour) - Schneide-Operationen
- [Materialtest-Raster](operations/material-test-grid) - Optimale Einstellungen finden
