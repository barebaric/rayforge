# Mehrschicht-Workflow

Das Mehrschicht-System von Rayforge ermöglicht es dir, komplexe Jobs in separate Verarbeitungsstufen zu organisieren, jede mit eigenen Operationen und Einstellungen. Dies ist essentiell für das Kombinieren verschiedener Prozesse wie Gravieren und Schneiden oder für die Arbeit mit mehreren Materialien.

## Was sind Ebenen?

Eine **Ebene** in Rayforge ist:

- **Ein Container** für Werkstücke (importierte Formen, Bilder, Text)
- **Ein Workflow**, der definiert, wie diese Werkstücke verarbeitet werden
- **Ein Schritt**, der während Jobs sequenziell verarbeitet wird

**Schlüsselkonzept:** Ebenen werden in Reihenfolge verarbeitet, eine nach der anderen, was es dir ermöglicht, die Sequenz der Operationen zu steuern.

:::note Ebenen und Werkstücke
Eine Ebene enthält ein oder mehrere Werkstücke. Beim Importieren von SVG-Dateien mit Ebenen wird jede Ebene aus deinem Design zu einer separaten Ebene in Rayforge. Dies lässt dich dein Design genau so organisiert halten, wie du es erstellt hast.
:::


---

## Warum mehrere Ebenen verwenden?

### Häufige Anwendungsfälle

**1. Erst gravieren, dann schneiden**

Der häufigste Mehrschicht-Workflow:

- **Ebene 1:** Design rastergravieren
- **Ebene 2:** Umriss konturschneiden

**Warum separate Ebenen?**

- Gravieren zuerst stellt sicher, dass das Stück sich während des Gravierens nicht bewegt
- Schneiden zuletzt verhindert, dass Teile fallen, bevor Gravieren abgeschlossen ist
- Unterschiedliche Leistungs-/Geschwindigkeitseinstellungen für jede Operation

**2. Mehrfach-Durchgang-Schneiden**

Für dicke Materialien:

- **Ebene 1:** Erster Durchgang bei moderater Leistung
- **Ebene 2:** Zweiter Durchgang bei voller Leistung (gleiche Geometrie)
- **Ebene 3:** Optionaler dritter Durchgang, falls nötig

**Vorteile:**

- Reduziert Verrußen im Vergleich zu einem Hochleistungs-Durchgang
- Jede Ebene kann unterschiedliche Geschwindigkeits-/Leistungseinstellungen haben

**3. Multi-Material-Projekte**

Verschiedene Materialien in einem Job:

- **Ebene 1:** Acrylteile schneiden
- **Ebene 2:** Holzteile gravieren
- **Ebene 3:** Metallteile markieren

**Anforderungen:**

- Jede Ebene zielt auf unterschiedliche Bereiche des Bettes ab
- Unterschiedliche Geschwindigkeit/Leistung/Fokus für jedes Material

**4. SVG-Ebenen-Import**

SVG-Dateien mit bestehender Ebenenstruktur importieren:

- **Ebene 1:** Gravur-Elemente aus SVG
- **Ebene 2:** Schneide-Elemente aus SVG
- **Ebene 3:** Ritz-Elemente aus SVG

**Workflow:**

- Eine SVG-Datei importieren, die Ebenen hat
- "Originalvektoren verwenden" im Import-Dialog aktivieren
- Zu importierende Ebenen aus der erkannten Ebenen-Liste auswählen
- Jede Ebene wird zu einer separaten Ebene in Rayforge

**Anforderungen:**

- Deine SVG-Datei muss Ebenen verwenden (erstellt in Inkscape oder ähnlicher Software)
- "Originalvektoren verwenden" beim Importieren aktivieren
- Ebenennamen werden aus deiner Design-Software beibehalten

---

## Ebenen erstellen und verwalten

### Eine neue Ebene hinzufügen

1. **Auf die "+"-Taste klicken** im Ebenen-Panel
2. **Die Ebene benennen** beschreibend (z.B. "Gravur-Ebene", "Schnitt-Ebene")
3. **Die Ebene erscheint** in der Ebenenliste

**Standard:** Neue Dokumente beginnen mit einer Ebene.

### Ebenen-Eigenschaften

Jede Ebene hat:

| Eigenschaft     | Beschreibung                                              |
| --------------- | --------------------------------------------------------- |
| **Name**        | Der Name, der in der Ebenenliste angezeigt wird           |
| **Sichtbar**    | Sichtbarkeit in Arbeitsfläche und Vorschau umschalten     |
| **Rohmaterial** | Optionale Material-Zuordnung                              |
| **Workflow**    | Die Operation(en), die auf Werkstücke in dieser Ebene angewendet werden |
| **Werkstücke**  | Die Formen/Bilder, die in dieser Ebene enthalten sind     |

:::note Ebenen als Container
Ebenen sind Container für deine Werkstücke. Beim Importieren von SVG-Dateien mit Ebenen wird jede Ebene aus deinem Design zu einer separaten Ebene in Rayforge.
:::


### Ebenen neu anordnen

**Ausführungsreihenfolge = Ebenenreihenfolge in der Liste (von oben nach unten)**

Zum Neuordnen:

1. **Ziehen und ablegen** von Ebenen im Ebenen-Panel
2. **Reihenfolge ist wichtig** - Ebenen werden von oben nach unten ausgeführt

**Beispiel:**

```
Ebenen-Panel:
1. Gravur-Ebene      Wird zuerst ausgeführt
2. Ritz-Ebene        Wird als zweites ausgeführt
3. Schnitt-Ebene     Wird zuletzt ausgeführt (empfohlen)
```

### Ebenen löschen

1. **Die Ebene auswählen** im Ebenen-Panel
2. **Auf die Löschen-Taste klicken** oder Entf drücken
3. **Löschen bestätigen** (alle Werkstücke in der Ebene werden entfernt)

:::warning Löschen ist dauerhaft
Das Löschen einer Ebene entfernt alle ihre Werkstücke und Workflow-Einstellungen. Verwende Rückgängig, wenn du versehentlich löschst.
:::


---

## Werkstücke Ebenen zuweisen

### Manuelle Zuweisung

1. **Ein Werkstück importieren oder erstellen**
2. **Das Werkstück ziehen** zur gewünschten Ebene im Ebenen-Panel
3. **Oder das Eigenschaften-Panel verwenden**, um die Ebene des Werkstücks zu ändern

### SVG-Ebenen-Import

Beim Importieren von SVG-Dateien mit aktiviertem "Originalvektoren verwenden":

1. **"Originalvektoren verwenden" aktivieren** im Import-Dialog
2. **Rayforge erkennt Ebenen** aus deiner SVG-Datei
3. **Welche Ebenen auswählen** zum Importieren mit den Ebenen-Schaltern
4. **Jede ausgewählte Ebene** wird zu einer separaten Ebene mit eigenem Werkstück

:::note Ebenen-Erkennung
Rayforge erkennt automatisch Ebenen aus deiner SVG-Datei. Jede Ebene, die du in deiner Design-Software erstellt hast, wird als separate Ebene in Rayforge erscheinen.
:::


:::note Nur Vektor-Import
Ebenenauswahl ist nur bei direktem Vektor-Import verfügbar. Bei Verwendung des Trace-Modus wird das gesamte SVG als ein Werkstück verarbeitet.
:::


### Werkstücke zwischen Ebenen verschieben

**Ziehen und ablegen:**

- Werkstück(e) in der Arbeitsfläche oder im Dokument-Panel auswählen
- Zur Zielebene im Ebenen-Panel ziehen

**Ausschneiden und einfügen:**

- Werkstück aus aktueller Ebene ausschneiden (Strg+X)
- Zielebene auswählen
- Einfügen (Strg+V)

### SVG-Import-Dialog

Beim Importieren von SVG-Dateien bietet der Import-Dialog Optionen, die die Ebenen-Handhabung beeinflussen:

**Import-Modus:**

- **Originalvektoren verwenden:** Behält deine Vektorpfade und Ebenenstruktur. Wenn aktiviert, erscheint ein "Ebenen"-Abschnitt, der alle Ebenen aus deiner Datei zeigt.
- **Trace-Modus:** Konvertiert das SVG in eine Bitmap und zeichnet die Umrisse nach. Ebenenauswahl ist in diesem Modus deaktiviert.

**Ebenen-Abschnitt (Nur Vektor-Import):**

- Zeigt alle Ebenen aus deiner SVG-Datei
- Jede Ebene hat einen Umschalter zum Aktivieren/Deaktivieren des Imports
- Ebenennamen aus deiner Design-Software werden beibehalten
- Nur ausgewählte Ebenen werden als separate Ebenen importiert

:::tip SVG-Dateien für Ebenen-Import vorbereiten
Um SVG-Ebenen-Import zu verwenden, erstelle dein Design mit Ebenen in Software wie Inkscape. Verwende das Ebenen-Panel, um dein Design zu organisieren, und Rayforge wird diese Struktur beibehalten.
:::


---

## Ebenen-Workflows

Jede Ebene hat einen **Workflow**, der definiert, wie ihre Werkstücke verarbeitet werden.

### Ebenen-Workflows einrichten

Für jede Ebene wählst du einen Operationstyp und konfigurierst seine Einstellungen:

**Operationstypen:**

- **Kontur** - Folgt Umrissen (zum Schneiden oder Ritzen)
- **Rastergravur** - Graviert Bilder und füllt Bereiche
- **Tiefengravur** - Erzeugt varying Tiefe Gravuren

**Optionale Erweiterungen:**

- **Laschen** - Kleine Brücken, um Teile während des Schneidens an Ort zu halten
- **Overscan** - Erweitert Schnitte über die Form hinaus für sauberere Kanten
- **Schnittbreiten-Anpassung** - Kompensiert die Schneidebreite des Lasers

### Häufige Ebenen-Setups

**Gravur-Ebene:**

- Operation: Rastergravur
- Einstellungen: 300-500 DPI, moderate Geschwindigkeit
- Normalerweise keine zusätzlichen Optionen benötigt

**Schneide-Ebene:**

- Operation: Kontur-Schneiden
- Optionen: Laschen (um Teile zu halten), Overscan (für saubere Kanten)
- Einstellungen: Langsamere Geschwindigkeit, höhere Leistung

**Ritz-Ebene:**

- Operation: Kontur (niedrige Leistung, schneidet nicht durch)
- Einstellungen: Niedrige Leistung, schnelle Geschwindigkeit
- Zweck: Falzlinien, dekorative Linien

---

## Ebenen-Sichtbarkeit

Steuern, welche Ebenen auf der Arbeitsfläche und in Vorschauen angezeigt werden:

### Arbeitsflächen-Sichtbarkeit

- **Augen-Symbol** im Ebenen-Panel schaltet Sichtbarkeit um
- **Versteckte Ebenen:**
  - Nicht in 2D-Arbeitsfläche angezeigt
  - Nicht in 3D-Vorschau angezeigt
  - **Immer noch im generierten G-Code enthalten**

**Anwendungsfälle:**

- Komplexe Gravur-Ebenen verstecken, während Schneide-Ebenen positioniert werden
- Arbeitsfläche entcluttern, wenn an spezifischen Ebenen gearbeitet wird
- Fokus auf eine Ebene nach der anderen

### Sichtbarkeit vs. Aktiviert

| Zustand                       | Arbeitsfläche | Vorschau | G-Code |
| ----------------------------- | ------------- | -------- | ------ |
| **Sichtbar & Aktiviert**      | Ja            | Ja       | Ja     |
| **Versteckt & Aktiviert**     | Nein          | Nein     | Ja     |
| **Sichtbar & Deaktiviert**    | Ja            | Ja       | Nein   |
| **Versteckt & Deaktiviert**   | Nein          | Nein     | Nein   |

:::note Ebenen deaktivieren
:::

Um eine Ebene temporär von Jobs auszuschließen, ohne sie zu löschen, schalte die Operation der Ebene aus oder deaktiviere sie in den Ebeneneinstellungen.

---

## Ebenen-Ausführungsreihenfolge

### Wie Ebenen verarbeitet werden

Während der Job-Ausführung verarbeitet Rayforge jede Ebene in Reihenfolge von oben nach unten. Innerhalb jeder Ebene werden alle Werkstücke verarbeitet, bevor zur nächsten Ebene gewechselt wird.

### Reihenfolge ist wichtig

**Falsche Reihenfolge:**

```
1. Schnitt-Ebene
2. Gravur-Ebene
```

**Problem:** Geschnittene Teile können herausfallen oder sich bewegen, bevor graviert wird!

**Korrekte Reihenfolge:**

```
1. Gravur-Ebene
2. Schnitt-Ebene
```

**Warum:** Gravieren geschieht, während das Teil noch befestigt ist, dann schneidet das Schneiden es frei.

### Mehrere Durchgänge

Für dicke Materialien erstellst du mehrere Schneide-Ebenen:

```
1. Gravur-Ebene
2. Schnitt-Ebene (Durchgang 1) - 50% Leistung
3. Schnitt-Ebene (Durchgang 2) - 75% Leistung
4. Schnitt-Ebene (Durchgang 3) - 100% Leistung
```

**Tipp:** Verwende dieselbe Geometrie für alle Schneide-Durchgänge (dupliziere die Ebene).

---

## Erweiterte Techniken

### Ebenen-Gruppierung nach Material

Verwende Ebenen, um nach Material zu organisieren, wenn du gemischte Jobs ausführst:

```
Material 1 (3mm Acryl):
  - Acryl-Gravur-Ebene
  - Acryl-Schnitt-Ebene

Material 2 (3mm Sperrholz):
  - Holz-Gravur-Ebene
  - Holz-Schnitt-Ebene
```

**Workflow:**

1. Alle Material-1-Ebenen verarbeiten
2. Materialien wechseln
3. Alle Material-2-Ebenen verarbeiten

**Alternative:** Verwenden Sie separate Dokumente für verschiedene Materialien.

### Pausieren zwischen Ebenen

Du kannst Rayforge so konfigurieren, dass zwischen Ebenen pausiert wird. Dies ist nützlich, wenn du:

- Materialien mitten im Job wechseln musst
- Den Fortschritt vor dem Fortfahren inspizieren möchtest
- Den Fokus für verschiedene Operationen anpassen musst

Um Ebenen-Pausen einzurichten, verwende die Hooks-Funktion in deinen Maschineneinstellungen.

### Ebenen-spezifische Einstellungen

Der Workflow jeder Ebene kann einzigartige Einstellungen haben:

| Ebene   | Operation  | Geschwindigkeit | Leistung | Durchgänge |
| ------- | ---------- | --------------- | -------- | ---------- |
| Gravur  | Raster     | 300 mm/min      | 20%      | 1          |
| Ritzen  | Kontur     | 500 mm/min      | 10%      | 1          |
| Schnitt | Kontur     | 100 mm/min      | 90%      | 2          |

---

## Best Practices

### Namenskonventionen

**Gute Ebenennamen:**

- "Gravur - Logo"
- "Schnitt - Außenkontur"
- "Ritzen - Falzlinien"
- "Durchgang 1 - Grobschnitt"
- "Durchgang 2 - Feinschnitt"

**Schlechte Ebenennamen:**

- "Ebene 1", "Ebene 2" (nicht beschreibend)
- Lange Beschreibungen (kurz halten)

### Ebenen-Organisation

1. **Von oben nach unten = Ausführungsreihenfolge**
2. **Gravieren vor Schneiden** (allgemeine Regel)
3. **Verwandte Operationen gruppieren** (alles Schneiden, alles Gravieren)
4. **Sichtbarkeit verwenden**, um auf aktuelle Arbeit zu fokussieren
5. **Unbenutzte Ebenen löschen**, um Projekte sauber zu halten

### SVG-Dateien für Ebenen-Import vorbereiten

**Für beste Ergebnisse beim Importieren von SVG-Ebenen:**

1. **Das Ebenen-Panel verwenden** in deiner Design-Software, um dein Design zu organisieren
2. **Aussagekräftige Namen zuweisen** zu jeder Ebene (z.B. "Gravieren", "Schneiden")
3. **Ebenen flach halten** - vermeide, Ebenen in andere Ebenen zu stecken
4. **Deine Datei speichern** und in Rayforge importieren
5. **Ebenen-Erkennung verifizieren** durch Überprüfen des Import-Dialogs

Rayforge funktioniert am besten mit SVG-Dateien, die in Inkscape oder ähnlicher Vektor-Design-Software erstellt wurden, die Ebenen unterstützt.

### Leistung

**Viele Ebenen:**

- Keine signifikante Leistungsbeeinträchtigung
- 10-20 Ebenen sind üblich für komplexe Jobs
- Logisch organisieren, nicht um Ebenenanzahl zu minimieren

**Bei Bedarf vereinfachen:**

- Ähnliche Operationen in einer Ebene kombinieren, wenn möglich
- Weniger Rastergravuren verwenden (ressourcenintensiv)

---

## Fehlerbehebung

### Ebene generiert keinen G-Code

**Problem:** Ebene erscheint im Dokument, aber nicht im generierten G-Code.

**Lösungen:**

1. **Überprüfen, dass Ebene Werkstücke hat** - Leere Ebenen werden übersprungen
2. **Überprüfen, dass Workflow konfiguriert ist** - Ebene benötigt eine Operation
3. **Operationseinstellungen verifizieren** - Leistung > 0, gültige Geschwindigkeit, usw.
4. **Werkstück-Sichtbarkeit überprüfen** - Versteckte Werkstücke werden möglicherweise nicht verarbeitet
5. **G-Code neu generieren** - Eine kleine Änderung vornehmen, um Neu-Generierung zu erzwingen

### Falsche Ebenenreihenfolge

**Problem:** Operationen werden in unerwarteter Reihenfolge ausgeführt.

**Lösung:** Ebenen im Ebenen-Panel neu ordnen. Denke daran: oben = zuerst.

### Ebenen überlappen in der Vorschau

**Problem:** Mehrere Ebenen zeigen überlappenden Inhalt in der Vorschau.

**Klärung:** Dies ist normal, wenn Ebenen denselben XY-Bereich teilen.

**Lösungen:**

- Ebenen-Sichtbarkeit verwenden, um andere Ebenen temporär zu verstecken
- 3D-Vorschau überprüfen, um Tiefe/Reihenfolge zu sehen
- Verifiziere, dass dies beabsichtigt ist (z.B. gravieren dann schneiden derselben Form)

### Werkstück in falscher Ebene

**Problem:** Werkstück wurde falscher Ebene zugewiesen.

**Lösung:** Werkstück zur korrekten Ebene im Ebenen-Panel oder Dokument-Baum ziehen.

### SVG-Ebenen nicht erkannt

**Problem:** Importieren einer SVG-Datei, aber keine Ebenen erscheinen im Import-Dialog.

**Lösungen:**

1. **SVG-Struktur überprüfen** - Öffne deine Datei in Inkscape oder ähnlicher Software, um zu verifizieren, dass sie Ebenen hat
2. **"Originalvektoren verwenden" aktivieren** - Ebenenauswahl ist nur in diesem Import-Modus verfügbar
3. **Verifiziere, dass dein Design Ebenen hat** - Stelle sicher, dass du Ebenen in deiner Design-Software erstellt hast, nicht nur Gruppen
4. **Auf verschachtelte Ebenen prüfen** - Ebenen innerhalb anderer Ebenen werden möglicherweise nicht richtig erkannt
5. **Deine Datei neu speichern** - Manchmal hilft das erneute Speichern mit einer aktuellen Version deiner Design-Software

### SVG-Ebenen-Import zeigt falschen Inhalt

**Problem:** Importierte Ebene zeigt Inhalt von anderen Ebenen oder ist leer.

**Lösungen:**

1. **Ebenenauswahl überprüfen** - Verifiziere, dass die richtigen Ebenen im Import-Dialog aktiviert sind
2. **Dein Design verifizieren** - Öffne die Originaldatei in deiner Design-Software, um zu bestätigen, dass jede Ebene den richtigen Inhalt enthält
3. **Auf gemeinsame Elemente prüfen** - Elemente, die in mehreren Ebenen erscheinen, können Verwirrung stiften
4. **Trace-Modus versuchen** - Verwende Trace-Modus als Fallback, wenn Vektor-Import Probleme hat

---

## Verwandte Seiten

- [Operationen](./operations/contour) - Operationstypen für Ebenen-Workflows
- [Simulationsmodus](./simulation-mode) - Mehrschicht-Ausführung vorschauen
- [Makros & Hooks](../machine/hooks-macros) - Ebenen-Level-Hooks zur Automatisierung
- [3D-Vorschau](../ui/3d-preview) - Ebenen-Stack visualisieren
