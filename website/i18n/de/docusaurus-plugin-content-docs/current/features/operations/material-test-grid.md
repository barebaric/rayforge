# Materialtest-Raster

Der Materialtest-Raster-Generator erstellt parametrische Testmuster, um optimale Lasereinstellungen für verschiedene Materialien zu finden.

## Übersicht

Material-Testen ist essentiell für Laserarbeit - verschiedene Materialien erfordern verschiedene Leistungs- und Geschwindigkeitseinstellungen. Der Materialtest-Raster automatisiert diesen Prozess durch:

- Generieren von Testrastern mit konfigurierbaren Geschwindigkeits-/Leistungsbereichen
- Bereitstellen von Presets für gängige Lasertypen (Diode, CO2)
- Optimieren der Ausführungsreihenfolge zur Sicherheit (schnellste Geschwindigkeiten zuerst)
- Hinzufügen von Beschriftungen zur Identifizierung der Einstellungen jeder Testzelle

## Ein Materialtest-Raster erstellen

### Schritt 1: Den Generator öffnen

Zugriff auf den Materialtest-Raster-Generator:

- Menü: **Werkzeuge → Materialtest-Raster**
- Dies erstellt ein spezielles Werkstück, das das Testmuster generiert

### Schritt 2: Ein Preset wählen (Optional)

Rayforge enthält Presets für gängige Szenarien:

| Preset               | Geschwindigkeitsbereich    | Leistungsbereich | Verwendung für               |
| -------------------- | -------------------------- | ---------------- | ---------------------------- |
| **Dioden-Gravur**    | 1000-10000 mm/min          | 10-100%          | Diodenlaser-Gravur           |
| **Dioden-Schnitt**   | 100-5000 mm/min            | 50-100%          | Diodenlaser-Schneiden        |
| **CO2-Gravur**       | 3000-20000 mm/min          | 10-50%           | CO2-Laser-Gravur             |
| **CO2-Schnitt**      | 1000-20000 mm/min          | 30-100%          | CO2-Laser-Schneiden          |

Presets sind Startpunkte - Sie können alle Parameter nach Auswahl anpassen.

### Schritt 3: Parameter konfigurieren

Passen Sie die Testraster-Parameter im Einstellungsdialog an:

![Materialtest-Raster-Einstellungen](/screenshots/material-test-grid.png)

#### Testtyp

- **Gravur**: Füllt Quadrate mit Rastermuster
- **Schnitt**: Schneidet Umrisse von Quadraten

#### Geschwindigkeitsbereich

- **Min. Geschwindigkeit**: Langsamste zu testende Geschwindigkeit (mm/min)
- **Max. Geschwindigkeit**: Schnellste zu testende Geschwindigkeit (mm/min)
- Spalten im Raster repräsentieren verschiedene Geschwindigkeiten

#### Leistungsbereich

- **Min. Leistung**: Niedrigste zu testende Leistung (%)
- **Max. Leistung**: Höchste zu testende Leistung (%)
- Zeilen im Raster repräsentieren verschiedene Leistungsstufen

#### Raster-Abmessungen

- **Spalten**: Anzahl der Geschwindigkeits-Variationen (typisch 3-7)
- **Zeilen**: Anzahl der Leistungs-Variationen (typisch 3-7)

#### Größe & Abstand

- **Form-Größe**: Größe jedes Testquadrats in mm (Standard: 20mm)
- **Abstand**: Lücke zwischen Quadraten in mm (Standard: 5mm)

#### Beschriftungen

- **Beschriftungen einschließen**: Achsenbeschriftungen ein/aus, die Geschwindigkeits- und Leistungswerte anzeigen
- Beschriftungen erscheinen an linken und oberen Kanten
- Beschriftungen werden bei 10% Leistung, 1000 mm/min graviert

### Schritt 4: Das Raster generieren

Klicken Sie auf **Generieren**, um das Testmuster zu erstellen. Das Raster erscheint auf Ihrer Arbeitsfläche als spezielles Werkstück.

## Das Raster-Layout verstehen

### Raster-Organisation

```
Leistung (%)     Geschwindigkeit (mm/min) →
    ↓      1000   2500   5000   7500   10000
  100%     [  ]   [  ]   [  ]   [  ]   [  ]
   75%     [  ]   [  ]   [  ]   [  ]   [  ]
   50%     [  ]   [  ]   [  ]   [  ]   [  ]
   25%     [  ]   [  ]   [  ]   [  ]   [  ]
   10%     [  ]   [  ]   [  ]   [  ]   [  ]
```

- **Spalten**: Geschwindigkeit nimmt von links nach rechts zu
- **Zeilen**: Leistung nimmt von unten nach oben zu
- **Beschriftungen**: Zeigen exakte Werte für jede Zeile/Spalte

### Raster-Größen-Berechnung

**Ohne Beschriftungen:**

- Breite = Spalten × (form_größe + abstand) - abstand
- Höhe = Zeilen × (form_größe + abstand) - abstand

**Mit Beschriftungen:**

- 15mm Rand links und oben für Beschriftungsplatz hinzufügen

**Beispiel:** 5×5 Raster mit 20mm Quadraten und 5mm Abstand:

- Ohne Beschriftungen: 120mm × 120mm
- Mit Beschriftungen: 135mm × 135mm

## Ausführungsreihenfolge (Risiko-Optimierung)

Rayforge führt Testzellen in einer **risiko-optimierten Reihenfolge** aus, um Materialschäden zu verhindern:

1. **Höchste Geschwindigkeit zuerst**: Schnelle Geschwindigkeiten sind sicherer (weniger Hitzestau)
2. **Niedrigste Leistung innerhalb der Geschwindigkeit**: Minimiert Risiko bei jeder Geschwindigkeitsstufe

Dies verhindert, dass Verrußen oder Feuer mit langsamen, hochleistungs-Kombinationen beginnen.

**Beispiel-Ausführungsreihenfolge für 3×3 Raster:**

```
Reihenfolge:  1  2  3
              4  5  6  ← Höchste Geschwindigkeit, zunehmende Leistung
              7  8  9

(Schnellste Geschwindigkeit/niedrigste Leistung zuerst ausgeführt)
```

## Materialtest-Ergebnisse verwenden

### Schritt 1: Den Test ausführen

1. Ihr Material in den Laser laden
2. Den Laser richtig fokussieren
3. Den Materialtest-Raster-Job ausführen
4. Den Test überwachen - stoppen, wenn eine Zelle Probleme verursacht

### Schritt 2: Ergebnisse bewerten

Nach Abschluss des Tests untersuchen Sie jede Zelle:

- **Zu hell:** Leistung erhöhen oder Geschwindigkeit verringern
- **Zu dunkel/verrußt:** Leistung verringern oder Geschwindigkeit erhöhen
- **Perfekt:** Die Geschwindigkeits-/Leistungskombination notieren

### Schritt 3: Einstellungen aufzeichnen

Dokumentieren Sie Ihre erfolgreichen Einstellungen zur späteren Referenz:

- Materialtyp und -dicke
- Operationstyp (gravieren oder schneiden)
- Geschwindigkeits- und Leistungskombination
- Anzahl der Durchgänge
- Alle speziellen Hinweise

:::tip Materialdatenbank
Erwägen Sie, ein Referenzdokument mit Ihren Materialtest-Ergebnissen zu erstellen, um in zukünftigen Projekten schnell nachzuschlagen.
:::

## Erweiterte Verwendung

### Mit anderen Operationen kombinieren

Materialtest-Raster sind reguläre Werkstücke - Sie können sie mit anderen Operationen kombinieren:

**Beispiel-Workflow:**

1. Materialtest-Raster erstellen
2. Kontur-Schnitt um das gesamte Raster hinzufügen
3. Test ausführen, frei schneiden, Ergebnisse bewerten

Dies ist nützlich, um das Teststück aus Rohmaterial zu schneiden.

### Benutzerdefinierte Test-Bereiche

Für Feinabstimmung erstellen Sie Testbereiche mit engen Grenzen:

**Grobtest** (Bereich finden):
- Geschwindigkeit: 1000-10000 mm/min (5 Spalten)
- Leistung: 10-100% (5 Zeilen)

**Feinabstimmungs-Test** (optimieren):
- Geschwindigkeit: 4000-6000 mm/min (5 Spalten)
- Leistung: 35-45% (5 Zeilen)

### Verschiedene Materialien, gleiches Raster

Führen Sie dieselbe Rasterkonfiguration auf verschiedenen Materialien aus, um Ihre Materialbibliothek schneller aufzubauen.

## Tipps & Best Practices

### Raster-Design

✅ **Mit Presets beginnen** - Gute Startpunkte für gängige Szenarien
✅ **5×5-Raster verwenden** - Gute Balance aus Detail und Testzeit
✅ **Beschriftungen aktivieren** - Essentiell zur Identifizierung von Ergebnissen
✅ **Quadrate ≥20mm halten** - Einfacher Ergebnisse zu sehen und zu messen

### Test-Strategie

✅ **Zuerst Abfall testen** - Niemals auf Endmaterial testen
✅ **Eine Variable nach der anderen** - Geschwindigkeit ODER Leistungsbereich testen, nicht beide Extreme
✅ **Abkühlen lassen** - Zwischen Tests auf gleichem Material warten
✅ **Konsistenter Fokus** - Gleiche Fokusdistanz für alle Tests

### Sicherheit

⚠️ **Tests überwachen** - Laufende Tests niemals unbeaufsichtigt lassen
⚠️ **Konservativ beginnen** - Mit niedrigeren Leistungsbereichen beginnen
⚠️ **Belüftung überprüfen** - Sicherstellen, dassproper Rauchabsaugung vorhanden ist
⚠️ **Feuerwache** - Feuerlöscher bereit haben

## Fehlerbehebung

### Testzellen werden in falscher Reihenfolge ausgeführt

- Rayforge verwendet risiko-optimierte Reihenfolge (schnellste Geschwindigkeiten zuerst)
- Dies ist beabsichtigt und kann nicht geändert werden
- Siehe [Ausführungsreihenfolge](#ausführungsreihenfolge-risiko-optimierung) oben

### Ergebnisse sind inkonsistent

- **Überprüfen:** Material ist flach und richtig befestigt
- **Überprüfen:** Fokus ist über den gesamten Testbereich konsistent
- **Überprüfen:** Laserleistung ist stabil (Netzteil überprüfen)
- **Versuchen:** Kleineres Raster, um Testbereich zu reduzieren

## Verwandte Themen

- **[Simulationsmodus](../simulation-mode)** - Testausführung vor dem Ausführen vorschauen
- **[Gravur](engrave)** - Gravur-Operationen verstehen
- **[Kontur-Schneiden](contour)** - Schneide-Operationen verstehen
