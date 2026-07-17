---
description: "Erstelle Array-Kopien mit den Modi Gitter, Punktrotation und Kreis. Jeder Modi bietet eine Live-Vorschau auf der Leinwand und interaktive Platzierung."
---

# Arrays

Mit der Array-Funktion kannst du mehrere Kopien von ausgewählten Werkstücken
in drei verschiedenen Layout-Modi erstellen. Jeder Modus öffnet einen
nicht-modalen Dialog, sodass du weiterhin mit der Leinwand interagieren
kannst, während du die Parameter anpasst — die Vorschau aktualisiert sich in
Echtzeit.

Um einen Array-Dialog zu öffnen, wähle ein oder mehrere Werksstücke auf der
Leinwand aus und wähle dann den Array-Modus aus der Werkzeugleiste oder dem
Kontextmenü aus.

:::tip
Alle Array-Modi sind nicht-modal. Du kannst Werksstücke auf der Leinwand
ziehen, während der Dialog geöffnet ist, und die Vorschau aktualisiert sich
live, um die neuen Positionen widerzuspiegeln.
:::

---

## Gitter

Der Gitter-Modus ordnet Kopien in einer rechteckigen Matrix aus Zeilen und
Spalten an, mit konfigurierbarem horizontalen und vertikalem Abstand.

![Gitter-Array](/screenshots/main-array-grid.png)

### Einstellungen

| Einstellung | Beschreibung |
|-------------|--------------|
| **Zeilen** | Anzahl der Zeilen (1–360) |
| **Spalten** | Anzahl der Spalten (1–360) |
| **Abstandsmodus** | Wähle zwischen *Lücke* (Abstand zwischen Kopien) oder *Steigung* (Abstand von Kante zu Kante jedes Exemplars) |
| **Spaltenabstand** | Horizontaler Abstand zwischen den Spalten |
| **Zeilenabstand** | Vertikaler Abstand zwischen den Zeilen |

---

## Punktrotation

Der Punktrotations-Modus erstellt Kopien, indem er sie an ihrem eigenen
Mittelpunkt rotiert. Dies ist nützlich für die Erstellung von
kreisförmigen Mustern, bei denen jede Kopie an ihrem Originalort bleibt,
aber um einen Bruchteil des Gesamtwinkels gedreht wird.

![Punktrotations-Array](/screenshots/main-array-point-rotation.png)

### Einstellungen

| Einstellung | Beschreibung |
|-------------|--------------|
| **Anzahl** | Anzahl der Kopien (1–360) |
| **Gesamtwinkel (Grad)** | Gesamter Winkelbereich aller Kopien (−360° bis 360°) |

:::info
Da die Rotation um den eigenen Mittelpunkt der Auswahl erfolgt, verschieben
sich beim Ziehen des Werksstücks auf der Leinwand alle Kopien gemeinsam,
während der Dialog geöffnet bleibt.
:::

---

## Kreis

Der Kreis-Modus platziert Kopien entlang eines Kreisbogens um einen
Mittelpunkt. Ein Fadenkreuz-Marker auf der Leinwand zeigt den Mittelpunkt
an, und du kannst ihn an eine neue Position ziehen, während der Dialog
geöffnet ist.

![Kreis-Array](/screenshots/main-array-circular.png)

### Einstellungen

| Einstellung | Beschreibung |
|-------------|--------------|
| **Anzahl** | Anzahl der Kopien (1–360) |
| **Gesamtwinkel (Grad)** | Winkelbereich des Bogens (−360° bis 360°) |
| **Mitte X** | X-Koordinate des Kreismittelpunkts |
| **Mitte Y** | Y-Koordinate des Kreismittelpunkts |
| **Radius** | Radius des kreisförmigen Pfads |
| **Kopien drehen** | Wenn aktiviert, wird jede Kopie entlang der Bogentangente gedreht |

:::tip Mittelpunkt ziehen
Das Fadenkreuz auf der Leinwand repräsentiert den Kreismittelpunkt. Ziehe es,
um das Array interaktiv umzupositionieren — die Felder Mitte X und Mitte Y
im Dialog werden automatisch aktualisiert.
:::

:::tip Werksstücke ziehen
Du kannst auch das ursprüngliche Werkstück auf der Leinwand ziehen. Der
Radius wird automatisch aktualisiert, um die Kopien bei ihrem aktuellen
Abstand vom Mittelpunkt zu halten.
:::
