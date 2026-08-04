# Farbregeln

Mit Farbregeln kannst du einem bestimmten Schritt-Typ eine Farbe zuweisen, sodass
die passende Operation beim Import einer SVG-, PDF- oder anderen Vektordatei
automatisch ausgewählt wird. Statt für jede importierte Ebene manuell Schritte zu
erstellen, liest Rayforge die Farbe jeder Form und wendet die passende Regel an.

## So funktioniert es

Beim Import einer Vektordatei kann Rayforge die eingehenden Formen nach ihrer
Farbe gruppieren. Jede eindeutige Farbe wird zu einer Ebene. Existiert für diese
Farbe eine Farbregel, wird der Ebene automatisch der Schritt-Typ der Regel
zugewiesen. Farben ohne Regel erhalten das Standardverhalten (Kontur für
Konturlinien, plus Gravur, wenn die Formen Füllungen haben).

Nachdem der Schritt-Typ zugewiesen wurde, läuft das normale
[Rezept-Matching](recipes)-System darüber — Farbregeln bestimmen also, *welche*
Operation ausgeführt wird, und Rezepte bestimmen, *wie* sie ausgeführt wird
(Leistung, Geschwindigkeit, Durchgänge, usw.).

## Farbregeln erstellen

### 1. Die Seite "Farbregeln" öffnen

Menü: **Bearbeiten → Einstellungen**, dann in der Seitenleiste **Farbregeln**
auswählen.

### 2. Eine Regel hinzufügen

Klicke auf **Farbregel hinzufügen**, um den Bearbeitungsdialog zu öffnen:

- **Farbe** — Wähle die SVG-Farbe, die diese Regel auslösen soll. Verwende die
  Farbauswahl, um die Strich- oder Füllfarbe aus deiner Design-Software zu
  übernehmen.
- **Beschriftung** *(optional)* — Ein Anzeigename, der in der Regelliste
  angezeigt wird (z.B. "Rot schneiden", "Blau gravieren"). Wenn das Feld leer
  bleibt, wird der Hex-Wert verwendet.
- **Schritt-Typ** — Die Operation, die erstellt wird, wenn diese Farbe
  importiert wird. Jeder registrierte Schritt-Typ ist verfügbar, einschließlich
  der von [Addons](addons) bereitgestellten (z.B. Shrink Wrap, Material Test
  Grid).

### 3. Speichern

Klicke auf **Hinzufügen**, um die Regel zu speichern. Sie wird beim nächsten
Import sofort wirksam. Regeln werden in deiner Benutzerkonfiguration gespeichert
und bleiben über Sitzungen hinweg erhalten.

:::tip Farben exakt abgleichen
Farbregeln gleichen nach exaktem Hex-Wert ab. Notiere beim Auswählen einer
Farbe in deiner Design-Software (Inkscape, Illustrator, usw.) den exakten
Hex-Code und gib denselben Wert in Rayforge ein. Zum Beispiel muss `#e34c4c`
in deiner SVG-Datei auch `#e34c4c` in der Regel sein — schon eine um eine
Ziffer abweichende Farbe verhindert den Abgleich.
:::

## Regeln verwalten

Jede Regel in der Liste zeigt ein Farbmuster, die Beschriftung, den Schritt-Typ
sowie Bearbeiten-/Löschen-Schaltflächen.

- **Bearbeiten** — Farbe, Beschriftung oder Schritt-Typ ändern. Das Ändern der
  Farbe einer bestehenden Regel ersetzt sie (die alte Farbe wird entfernt).
- **Löschen** — Die Regel dauerhaft entfernen.
- **Nicht verfügbare Schritt-Typen** — Wenn das Addon des Schritt-Typs
  deinstalliert wurde, erscheint neben der Regel ein Warnsymbol. Die Regel wird
  aufbewahrt, damit du sie beheben oder das Addon neu installieren kannst.
  Ebenen, die während des Imports mit einer Regel mit nicht verfügbarem
  Schritt-Typ übereinstimmen, fallen auf das Standardverhalten zurück.

## Importverhalten

### Automatische Farbgruppierung

Wenn Farbregeln existieren, wechselt der Import-Dialog automatisch auf
**Farben** als Ebenenquelle für Dateien, die eindeutige Farben enthalten. So
stellt Rayforge sicher, dass jede Farbe eine eigene Ebene wird, damit die Regeln
angewendet werden können. Du kannst im Dialog bei Bedarf weiterhin auf
**SVG-Ebenen** oder andere Quellen umschalten.

### Was eine Regel auslöst

Eine Farbregel wird angewendet, wenn:

1. Die Datei mit **Farben** als Ebenenquelle importiert wird.
2. Die Strich- oder Füllfarbe einer Form exakt der Farbe der Regel entspricht.
3. Der Schritt-Typ der Regel aktuell registriert ist.

Regeln gelten **nicht** für Dateien, die mit den Ebenenquellen **SVG-Ebenen**
oder **Abflachen** importiert werden, da diese Quellen nicht nach Farbe
gruppieren.

## Beispiel-Workflow

Ein typisches Setup für mehrfarbige SVG-Designs:

1. **In deiner Design-Software** verschiedenen Operationen eindeutige Farben
   zuweisen:
   - Rot (`#ff0000`) für Schnittkonturen
   - Blau (`#0000ff`) für Gravuren
   - Grün (`#00ff00`) für Ritzungen

2. **In Rayforge** drei Farbregeln erstellen:
   - `#ff0000` → Kontur
   - `#0000ff` → Gravur
   - `#00ff00` → Kontur (mit anderen Rezept-Einstellungen)

3. **SVG importieren.** Der Import-Dialog wählt automatisch Farben aus, und jede
   Farbgruppe erhält automatisch ihren Schritt-Typ.

4. **Feinabstimmung** mit [Rezepten](recipes), um Leistung, Geschwindigkeit und
   andere Parameter pro Schritt-Typ festzulegen.

## Farbregeln und Rezepte

Farbregeln und Rezepte ergänzen sich:

| Funktion    | Was festgelegt wird                 | Wann sie angewendet wird |
| ----------- | ----------------------------------- | ------------------------ |
| Farbregeln  | Schritt-Typ (Kontur, usw.)          | Beim Import              |
| Rezepte     | Schritt-Einstellungen (Leistung, usw.) | Bei der Schritterstellung |

Ein typisches Setup verwendet Farbregeln, um die Operation auszuwählen, und
Rezepte, um die Parameter zu konfigurieren. Beispielsweise wird eine rote
Farbregel auf Kontur abgebildet, und ein Rezept, das auf deinem aktuellen
Material auf den Kontur-Schritt-Typ ausgerichtet ist, wendet die richtige
Schnittgeschwindigkeit und Leistung an.

---

**Verwandte Themen**:

- [Rezepte](recipes) - Leistungs-, Geschwindigkeits- und Parameter-Presets anwenden
- [Dateien importieren](../files/importing.md) - SVG- und Vektor-Importoptionen
- [Mehr-Ebenen-Workflow](../features/multi-layer.md) - Ebenenorganisation
- [Operationen](../features/operations/contour.md) - Schritt-Typ-Referenz
