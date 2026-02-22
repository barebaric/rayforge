# Overscan

Overscan erweitert Rastergravur-Linien über den tatsächlichen Inhaltsbereich hinaus, um sicherzustellen, dass der Laser während des Gravierens konstante Geschwindigkeit erreicht und Beschleunigungs-Artefakte eliminiert.

## Das Problem: Beschleunigungs-Markierungen

Ohne Overscan leidet Rastergravur unter **Beschleunigungs-Artefakten**:

- **Helle Kanten** wo Beschleunigung beginnt (Laser bewegt sich zu schnell für Leistungsstufe)
- **Dunkle Kanten** wo Verlangsamung auftritt (Laser verweilt länger)
- **Inkonsistente Gravur-Tiefe/Dunkelheit** entlang jeder Zeile
- Sichtbare Bänderung oder Streifenbildung an Zeilenenden

## Wie Overscan funktioniert

Overscan **erweitert den Werkzeugweg** vor und nach jeder Rasterlinie:

**Prozess:**

1. **Einlauf:** Laser bewegt sich an eine Position _vor_ der Zeile beginnt
2. **Beschleunigen:** Laser beschleunigt auf Zielgeschwindigkeit (Laser AUS)
3. **Gravieren:** Laser schaltet ein und graviert bei konstanter Geschwindigkeit
4. **Verlangsamen:** Laser schaltet aus und verlangsamt _nach_ der Zeile endet

**Ergebnis:** Der gesamte gravierte Bereich erhält konstante Leistung bei konstanter Geschwindigkeit.

**Vorteile:**

- Gleichmäßige Gravur-Tiefe über gesamte Rasterlinie
- Keine hellen/dunklen Kanten
- Höherwertige Fotogravur
- Professionell aussehende Ergebnisse

## Overscan konfigurieren

Overscan ist ein **Transformator** im Rayforge-Workflow-Pipeline.

**Zum Aktivieren:**

1. **Die Ebene auswählen** mit Rastergravur
2. **Workflow-Einstellungen öffnen** (oder Operationseinstellungen)
3. **Overscan-Transformator hinzufügen** falls noch nicht vorhanden
4. **Distanz konfigurieren**

**Einstellungen:**

| Einstellung          | Beschreibung              | Typischer Wert    |
| -------------------- | ------------------------- | ----------------- |
| **Aktiviert**        | Overscan ein/aus schalten | AN (für Raster)   |
| **Distanz (mm)**     | Wie weit Linien erweitern | 2-5 mm            |

## Overscan-Distanz wählen

Die Overscan-Distanz sollte der Maschine ermöglichen, **vollständig auf Zielgeschwindigkeit zu beschleunigen**.

**Praktische Richtlinien:**

| Max. Geschwindigkeit     | Beschleunigung | Empfohlener Overscan |
| ------------------------ | -------------- | -------------------- |
| 3000 mm/min (50 mm/s)    | Niedrig        | 5 mm                 |
| 3000 mm/min (50 mm/s)    | Mittel         | 3 mm                 |
| 3000 mm/min (50 mm/s)    | Hoch           | 2 mm                 |
| 6000 mm/min (100 mm/s)   | Niedrig        | 10 mm                |
| 6000 mm/min (100 mm/s)   | Mittel         | 6 mm                 |
| 6000 mm/min (100 mm/s)   | Hoch           | 4 mm                 |

**Faktoren, die erforderliche Distanz beeinflussen:**

- **Geschwindigkeit:** Höhere Geschwindigkeit = mehr Distanz zum Beschleunigen nötig
- **Beschleunigung:** Niedrigere Beschleunigung = mehr Distanz nötig
- **Maschinen-Mechanik:** Riemenantrieb vs Direktantrieb beeinflusst Beschleunigung

**Abstimmung:**

- **Zu wenig:** Beschleunigungs-Markierungen noch an Kanten sichtbar
- **Zu viel:** Verschwendet Zeit, kann Maschinen-Grenzen treffen
- **Mit 3mm beginnen** und basierend auf Ergebnissen anpassen

## Overscan-Einstellungen testen

**Testverfahren:**

1. **Eine Testgravur erstellen:**
   - Gefülltes Rechteck (50mm x 20mm)
   - Ihre typischen Gravureinstellungen verwenden
   - Overscan bei 3mm aktivieren

2. **Den Test gravieren:**
   - Den Job ausführen
   - Fertigstellen lassen

3. **Die Kanten untersuchen:**
   - Linke und rechte Kanten des Rechtecks ansehen
   - Auf Dunkelheits-Variation an Kanten prüfen
   - Kanten-Dunkelheit mit Zentrum-Dunkelheit vergleichen

4. **Anpassen:**
   - **Wenn Kanten heller/dunkler sind:** Overscan erhöhen
   - **Wenn Kanten mit Zentrum übereinstimmen:** Overscan ist ausreichend
   - **Wenn Kanten perfekt sind:** Versuchen, Overscan leicht zu reduzieren, um Zeit zu sparen

## Wann Overscan verwenden

**Immer verwenden für:**

- Fotogravur (Raster)
- Füllmuster
- Jegliche hochdetaillierte Rasterarbeit
- Graustufen-Bildgravur
- Textgravur (Rastermodus)

**Optional für:**

- Vektorschneiden (nicht benötigt)
- Sehr langsames Gravieren (Beschleunigung weniger bemerkbar)
- Große einfache Formen (Kanten weniger kritisch)

**Deaktivieren für:**

- Vektor-Operationen
- Sehr kleine Arbeitsbereiche (könnte Grenzen überschreiten)
- Wenn Kantenqualität nicht wichtig ist

---

## Verwandte Themen

- [Gravur-Operationen](./operations/engrave) - Gravureinstellungen konfigurieren
- [Materialtest-Raster](./operations/material-test-grid) - Optimale Leistung/Geschwindigkeit finden
