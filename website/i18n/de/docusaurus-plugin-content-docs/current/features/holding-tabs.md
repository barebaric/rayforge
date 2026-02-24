# Halte-Laschen

Halte-Laschen (auch Brücken oder Laschen genannt) sind kleine ungeschnittene Abschnitte entlang von Schneidepfaden, die Teile am umgebenden Material befestigt halten. Dies verhindert, dass geschnittene Teile sich während des Jobs bewegen, was Fehlausrichtungen, Beschädigungen oder Feuergefahren verursachen könnte.

## Warum Halte-Laschen verwenden?

Beim Schneiden durch Material kann das geschnittene Teil:

- **Position verschieben** mitten im Job, was nachfolgende Operationen fehl ausrichtet
- **Durchfallen** durch das Bett-Gitter oder kippen, wenn nur an Kanten unterstützt
- **Kollidieren** mit dem Laserkopf während der Bewegung
- **Feuer fangen** wenn es auf heißen Abfall darunter fällt
- **Beschädigt werden** durch Fallen oder Vibration

Halte-Laschen lösen diese Probleme, indem sie das Teil befestigt halten, bis Sie bereit sind, es zu entfernen.

---

## Wie Halte-Laschen funktionieren

Rayforge implementiert Laschen durch Erstellen von **kleinen Lücken im Schneidepfad**:

1. Sie markieren Positionen entlang des Schneidepfads, wo Laschen sein sollen
2. Während der G-Code-Generierung unterbricht Rayforge den Schnitt an jeder Lasche
3. Der Laser hebt ab (oder schaltet aus), überspringt die Laschen-Breite, und setzt das Schneiden fort
4. Nach Job-Abschluss brechen oder schneiden Sie die Laschen manuell, um das Teil zu befreien

---

## Halte-Laschen hinzufügen

### Schnelles Hinzufügen

1. **Wähle das Werkstück** aus, dem Sie Laschen hinzufügen möchten (muss eine Schnitt-/Kontur-Operation sein)
2. **Klicke auf das Laschen-Werkzeug** in der Symbolleiste oder drücken Sie die Laschen-Tastenkombination
3. **Klicke auf den Pfad** wo Sie Laschen möchten:
   - Laschen erscheinen als kleine Griffe auf der Pfad-Umrandung
   - Mehrere Male klicken, um mehr Laschen hinzuzufügen
   - Typischerweise 3-4 Laschen für kleine Teile, mehr für größere Stücke
4. **Laschen aktivieren** falls nicht bereits aktiviert (Umschalten im Eigenschaften-Panel)

### Das Laschen-hinzufügen-Popover verwenden

Für mehr Kontrolle:

1. **Rechtsklick** auf das Werkstück oder **Bearbeiten → Laschen hinzufügen** verwenden
2. **Laschen-Platzierungsmethode wählen:**
   - **Manuell:** Individuelle Positionen klicken
   - **Äquidistant:** Laschen automatisch gleichmäßig um den Pfad verteilen
3. **Laschen-Einstellungen konfigurieren:**
   - **Anzahl der Laschen:** Wie viele Laschen zu erstellen (für äquidistant)
   - **Laschen-Breite:** Länge jedes ungeschnittenen Abschnitts (typisch 2-5mm)
4. **Auf Anwenden klicken**

---

## Laschen-Eigenschaften

### Laschen-Breite

Die **Breite** ist die Länge des ungeschnittenen Abschnitts entlang des Pfads.

**Empfohlene Breiten:**

| Material     | Dicke  | Laschen-Breite |
|--------------|-------- | -------------- |
| **Karton**   | 1-3mm   | 2-3mm          |
| **Sperrholz**| 3mm     | 3-4mm          |
| **Sperrholz**| 6mm     | 4-6mm          |
| **Acryl**    | 3mm     | 2-3mm          |
| **Acryl**    | 6mm     | 3-5mm          |
| **MDF**      | 3mm     | 3-4mm          |
| **MDF**      | 6mm     | 5-7mm          |

**Richtlinien:**
- **Dickere Materialien** benötigen breitere Laschen für Stabilität
- **Schwerere Teile** benötigen mehr und/oder breitere Laschen
- **Spröde Materialien** (Acryl) können kleinere Laschen verwenden (leichter zu brechen)
- **Faserige Materialien** (Holz) benötigen möglicherweise breitere Laschen

:::warning Laschen-Breite vs Material-Dicke
Laschen müssen breit genug sein, um das Teil zu stützen, aber klein genug, um sauber zu entfernen. Zu schmal = Teil kann abbrechen; zu breit = schwierig zu entfernen oder beschädigt das Teil.
:::

### Laschen-Position

Laschen werden mit zwei Parametern positioniert:

- **Segment-Index:** Welches Linien-/Bogen-Segment des Pfads
- **Position (0.0 - 1.0):** Wo entlang dieses Segments (0 = Start, 1 = Ende)

**Manuelle Platzierungs-Tipps:**
- Laschen auf **geraden Abschnitten** platzieren, wenn möglich (leichter zu entfernen)
- Laschen auf **engen Kurven** vermeiden (Spannungskonzentration)
- Laschen **gleichmäßig** um das Teil verteilen
- Laschen auf **Ecken** für maximale Unterstützung platzieren, falls nötig

### Äquidistante Laschen

Die **äquidistante** Funktion platziert Laschen automatisch in gleichen Abständen:

**Vorteile:**
- Gleichmäßige Gewichtsverteilung
- Vorhersehbares Brechverhalten
- Schnelles Setup für regelmäßige Formen

---

## Mit Laschen arbeiten

### Laschen bearbeiten

**Eine Lasche bewegen:**
1. Das Werkstück auswählen
2. Den Laschen-Griff entlang des Pfads ziehen
3. Loslassen, um neue Position zu setzen

**Eine Lasche skalieren:**
- Das Eigenschaften-Panel verwenden, um die Breite anzupassen
- Alle Laschen auf einem Werkstück teilen sich dieselbe Breite

**Eine Lasche löschen:**
1. Den Laschen-Griff zum Auswählen klicken
2. Entf drücken oder das Kontextmenü verwenden
3. Oder alle Laschen löschen und neu beginnen

### Laschen aktivieren/deaktivieren

Laschen ein/aus schalten, ohne sie zu löschen:

- **Werkstück-Eigenschaften-Panel:** "Laschen aktivieren" Kontrollkästchen
- **Symbolleiste:** Laschen-Sichtbarkeit Umschalt-Symbol

**Wenn deaktiviert:**
- Laschen werden nicht im G-Code generiert
- Laschen-Griffe sind auf der Arbeitsfläche versteckt
- Der Pfad schneidet vollständig durch

**Anwendungsfall:** Laschen temporär deaktivieren, um den Schnitt zu testen, dann für Produktion wieder aktivieren.

---

## Laschen nach dem Schneiden entfernen

**Werkzeuge:**
- Bastelmesser oder Teppichmesser
- Flachzange
- Meißel (für Holz)
- Feinsäge für dicke Materialien

**Technik:**

1. **Die Lasche ritzen** von beiden Seiten, falls zugänglich
2. **Sanft biegen**, um die Lasche zu belasten
3. **Durchschneiden** des verbleibenden Materials
4. **Schleifen oder feilen** Sie den Laschen-Rest bündig mit der Kante

**Für spröde Materialien (Acryl):**
- Minimale Laschen verwenden (sie brechen leicht)
- Vor dem Brechen tief ritzen
- Das Teil stützen, während Laschen gebrochen werden, um Risse zu vermeiden

**Für Holz:**
- Laschen erfordern möglicherweise Schneiden (brechen nicht sauber)
- Ein scharfes Messer oder Meißel verwenden
- Bündig schneiden, dann glatt schleifen

---

## Verwandte Seiten

- [Kontur-Schneiden](operations/contour) - Primäre Operation, die Laschen verwendet
- [Mehrschicht-Workflow](multi-layer) - Laschen über mehrere Ebenen verwalten
- [3D-Vorschau](../ui/3d-preview) - Laschen in der Vorschau visualisieren
- [Simulationsmodus](simulation-mode) - Schnitte mit Laschen-Lücken vorschauen
