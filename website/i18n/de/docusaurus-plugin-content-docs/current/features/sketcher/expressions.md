---
description:
  "Skizzen-Parameter, Einschränkungs-Ausdrücke und Textfeld-Vorlagen im Rayforge-Sketcher: Geometrie
  und Beschriftungen mit benannten Werten und Formeln antreiben."
---

# Ausdrücke und Parameter

Eine Skizze wird wirklich parametrisch, wenn ihre Abmessungen von benannten Werten angetrieben
werden statt von fest einprogrammierten Zahlen. Diese Seite beschreibt den vollständigen
Arbeitsablauf: Parameter erstellen, Geometrie über Ausdrücke antreiben und pro Instanz Werte aus dem
Hauptfenster zuweisen. Außerdem werden Vorlagenausdrücke in Textfeldern behandelt.

## Parameter hinzufügen und bearbeiten

Jede Skizze führt ihre eigene Liste von Parametern, angezeigt im Panel **Sketch-Parameter** links im
Sketch-Editor. **Parameter hinzufügen** erstellt einen, mit der Wahl zwischen einer Ganzzahl, einer
Gleitkommazahl, einem Schieberegler oder einer einzelnen Textzeile.

![Das Sketch-Parameter-Panel im Sketch-Editor](/screenshots/addons-sketcher-parameters-panel.webp)

Jeder Parameter ist eine aufklappbare Zeile. Klicke auf die Zeile, um die Definitionsfields
anzuzeigen:

- **Label** — der lesbare Name, der in Listen angezeigt wird.
- **Key** — der Bezeichner, auf den sich Ausdrücke beziehen (automatisch aus dem Label abgeleitet,
  sofern nicht manuell eingegeben). Halte ihn als gültigen Python-Namen, z.B. `width` oder
  `wall_thickness`.
- **Description** — eine optionale Notiz unter der Zeile.
- **Default Value** — der Anfangswert des Parameters.
- **Minimum / Maximum Value** — optionale Grenzen (Schalter für jeden aktivieren). Ein
  Schieberegler-Parameter hat immer einen begrenzten Bereich.

Ein typisches Setup für eine Box mit variabler Wandstärke sind zwei Parameter, `width` und
`thickness`. Nichts beschränkt die Geometrie noch; die Parameter sind nur Namen für Zahlen, bis ein
Ausdruck sie verwendet.

## Parameter in Ausdrücken verwenden

Doppelklicke auf eine dimensionale Einschränkung (siehe [Einschränkungen](constraints.md)) und gib
einen Ausdruck statt einer einfachen Zahl ein:

```
width / 2
```

Der Wert der Einschränkung wird zum Ergebnis dieses Ausdrucks, neu berechnet bei jeder Lösung der
Skizze. Im folgenden Beispiel ist die linke Kante auf `width / 2` beschränkt — ihre Markierung und
ihr Label werden in **Orange** dargestellt, um zu kennzeichnen, dass sie ausdrucksgesteuert ist —,
während die obere Kante eine einfache numerische Bemaßung behält:

![Eine ausdrucksgesteuerte Maßeinschränkung](/screenshots/addons-sketcher-parameters-expression.webp)

Ändere den Parameter `width`, und die eingeschränkte Geometrie folgt — eine Bearbeitung aktualisiert
jetzt jede Bemaßung, die darauf verweist.

Ausdrücke können Parameter mit Arithmetik und den Standard-Python-Mathematikfunktionen kombinieren:

```
width - 2 * thickness
sqrt(area) / 2
2 * pi * radius
```

Funktionen wie `sqrt`, `sin`, `cos` und `tan` sowie Konstanten wie `pi` stammen aus Pythons
`math`-Modul — genau dieses Modul plus die Parameter ist das, worauf ein Einschränkungs-Ausdruck
verweisen kann. Auch String-Parameter können referenziert werden, was vor allem in Textfeldern
nützlich ist.

## Werte im Hauptfenster zuweisen

Parameter, die in einer Skizze definiert werden, dienen als Standardwerte für deren Grenzen. Wenn
eine Skizze im Dokument platziert wird, trägt jedes Werkstück eine eigene Kopie jedes
Parameterwerts, und die Gruppe **Sketch-Parameter** im rechten Eigenschaftspanel erlaubt es, sie pro
Instanz zu überschreiben — dieselbe Skizze kann in mehreren Größen über ein Blatt verwendet werden,
jeweils mit eigener `width` und `thickness`.

Wähle das Skizzen-Werkstück im Hauptfenster, und die Gruppe erscheint im Eigenschaftspanel, eine
Zeile pro Parameter, jeweils mit dem Wert, den diese Instanz verwendet. Gib einen neuen Wert ein
oder verwende die Spin-Buttons; das Teil wird sofort neu generiert.

![Parameterwerte im Hauptfenster zuweisen](/screenshots/addons-sketcher-parameters.webp)

Die Bearbeitung der Parameter-_Definitionen_ (Parameter hinzufügen, Standardwert ändern oder Key
umbenennen) erfolgt im Sketch-Editor, wie oben beschrieben. Das Hauptfenster-Panel passt nur die
_Werte_ für die ausgewählte Instanz an — es spiegelt immer die Parametermenge der Skizze wider, und
eine neue Instanz verwendet die Standardwerte der Skizze, bis du sie überschreibst.

## Vorlagenausdrücke in Textfeldern {#template-expressions-in-text-boxes}

Textfelder lösen Ausdrücke in geschweiften Klammern zum Lösungszeitpunkt auf, sodass Beschriftungen
und gravierter Text live Werte anzeigen:

```
W = {width}, H = {height}
```

Jeder Parameter kann namentlich substituiert werden, und das Ergebnis kann mit einem
Python-Formatbezeichner nach einem Doppelpunkt formatiert werden:

- `{width}` — der aktuelle Wert des Parameters `width`
- `{name}` — der Wert eines String-Parameters
- `{width:.1f}` — eine Dezimalstelle
- `{timestamp():.0f}` — keine Dezimalen beim Ergebnis einer Funktion

Auch Mathematik funktioniert hier, entweder als Ausdruck wie `{width * 2}` oder über eine Funktion
wie `{sqrt(area):.2f}`. Verglichen mit Einschränkungs-Ausdrücken haben Textvorlagen einen
reichhaltigeren Werkzeugkasten: Neben dem Mathematikmodul stellen sie die folgenden eingebauten
Funktionen bereit, und es können benutzerdefinierte Funktionen für sie registriert werden (siehe
[unten](#custom-template-functions)).

### Eingebaute Vorlagenfunktionen

| Funktion        | Rückgabetyp | Beschreibung                                     |
| --------------- | ----------- | ------------------------------------------------ |
| `{today()}`     | `date`      | Aktuelles UTC-Datum (z.B. `2026-08-26`)          |
| `{date()}`      | `date`      | Alias für `today()`                              |
| `{now()}`       | `datetime`  | Aktuelles UTC-Datum und Uhrzeit                  |
| `{time()}`      | `time`      | Aktuelle UTC-Zeit (z.B. `15:30:00.123456+00:00`) |
| `{timestamp()}` | `float`     | Unix-Zeitstempel (Sekunden seit Epoche)          |
| `{uuid4()}`     | `str`       | 8-stelliger Hex-String (z.B. `a1b2c3d4`)         |
| `{uuid8()}`     | `str`       | Alias für `uuid4()`                              |
| `{uuid()}`      | `str`       | Vollständiger UUID v4-String (36 Zeichen)        |

Typische Anwendungen sind eindeutige Seriennummern pro Lösung (`Part # {uuid4()}`), live
Maßbeschriftungen (`W={width:.1f} H={height:.1f}`), Datumsstempel (`Date: {today()}`),
Produktionszähler (`{name} - {count:.0f}pcs`) oder Unix-Zeitstempel für die
Produktionsprotokollierung (`{timestamp():.0f}`).

## Benutzerdefinierte Vorlagenfunktionen {#custom-template-functions}

Du kannst eigene Funktionen registrieren, um sie in Textfeldvorlagen zu verwenden. Dies ist nützlich
zum Abrufen von Seriennummern aus einer Datenbank, zum Lesen externer Daten oder zum Erstellen
benutzerdefinierter Beschriftungen.

### Das Registrierungsskript schreiben

Erstelle eine Python-Datei (z.B. `~/.config/rayforge/my_functions.py`):

```python
"""Register custom template functions for text box expressions."""
import sqlite3

from sketcher.core.template_functions import register_template_function

DB_PATH = "/home/you/production.db"


def next_serial() -> str:
    """Fetch and reserve the next serial number from the database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "UPDATE counters SET value = value + 1 "
            "WHERE name = 'serial' RETURNING value"
        )
        row = cur.fetchone()
        conn.commit()
        return f"SN-{row[0]:06d}"
    finally:
        conn.close()

register_template_function("next_serial", next_serial)
```

Rufe `register_template_function(name, callable)` für jede Funktion auf. Die Funktion kann alles
tun, was Python kann — Dateien öffnen, Datenbankverbindungen herstellen, APIs aufrufen —, und sie
wird bei **jeder Berechnung** aufgerufen, daher sollte sie schnell sein (verwende Caching, wenn sich
die zugrunde liegenden Daten zwischen Berechnungen nicht ändern). Funktionen sind threadsicher, wenn
dein Callable es ist.

### Rayforge mit dem Skript starten

Verwende das `--script`-Flag, um deine Funktionen zu laden, bevor das Fenster geöffnet wird:

```bash
rayforge --script ~/.config/rayforge/my_functions.py mydoc.ryp
```

Dies führt dein Skript früh beim Start aus — bevor Addons geladen und bevor das Hauptfenster
erstellt wird —, sodass die Funktion verfügbar ist, wenn die Skizze zum ersten Mal gelöst wird.

### Die Funktion in einem Textfeld verwenden

Erstelle im Sketcher ein Textfeld mit:

```
{next_serial()}
```

Formatbezeichner funktionieren ebenfalls:

```
{next_serial():>20}
```

### Funktionen programmatisch registrieren

Wenn du ein Addon oder eine wiederverwendbare Bibliothek schreibst, kannst du
`register_template_function` aus jedem Python-Code aufrufen, der vor der Skizzenberechnung läuft:

```python
from sketcher.core.template_functions import register_template_function

register_template_function("part_number", lambda: f"P-{hash('x') % 10000:04d}")
```

### Eingebaute Funktionen können nicht entfernt werden

Die eingebauten Funktionen (`today`, `now`, `uuid` usw.) können nicht deregistriert werden. Wenn du
ihr Verhalten ändern möchtest, registriere eine Funktion mit einem anderen Namen.
