---
description: "Textfeld-Vorlagen und benutzerdefinierte Vorlagenfunktionen im parametrischen 2D-Sketcher von Rayforge."
---

# Textvorlagen

Textfelder unterstützen Vorlagenausdrücke in geschweiften Klammern. Diese
werden zum Lösungszeitpunkt mit den aktuellen Parameterwerten aufgelöst,
sodass sich der Text automatisch aktualisiert, wenn du eine Dimension oder
Eingabevariable änderst.

## Variablensubstitution

Referenziere beliebige Skizzenparameter oder Eingabevariablen namentlich:

- `{width}` — der aktuelle Wert des Parameters „width"
- `{name}` — der Wert eines String-Eingabeparameters
- `{count:.0f}` — formatiert mit einem Python-Formatbezeichner (keine Dezimalen)

## Mathematische Ausdrücke

Du kannst mathematische Funktionen in Vorlagen verwenden:

- `{sqrt(area):.2f}` — Quadratwurzel von „area", formatiert auf 2 Dezimalen
- `{width * 2}` — arithmetische Ausdrücke

Die Standard-Mathematikfunktionen (`sqrt`, `sin`, `cos`, `tan`, `pi` usw.)
sind verfügbar.

## Eingebaute Funktionen

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

## Formatspezifikationen

Python-Format-Spezifikationen funktionieren mit jedem
Ausdrucksergebnis:

- `{width:.1f}` — eine Dezimalstelle
- `{timestamp():.0f}` — keine Dezimalen beim Zeitstempel
- `{today()}` — Standardzeichenketten-Darstellung

## Anwendungsbeispiele

- `Teil #{uuid4()}` — eindeutige Seriennummer bei jeder Lösung
- `B={width:.1f} H={height:.1f}` — live Maßbeschriftungen
- `Datum: {today()}` — jedes Teil datumsstempeln
- `{name} - {count:.0f}Stk` — String- und numerische Parameter kombinieren
- `{timestamp():.0f}` — Unix-Zeitstempel für Produktionsprotokollierung

## Benutzerdefinierte Vorlagenfunktionen

Du kannst eigene Funktionen für Textfeldvorlagen registrieren.
Dies ist nützlich zum Abrufen von Seriennummern aus einer
Datenbank, zum Lesen externer Daten oder zum Erstellen
benutzerdefinierter Beschriftungen.

### Das Registrierungsskript schreiben

Erstelle eine Python-Datei (z.B.
`~/.config/rayforge/meine_funktionen.py`):

```python
"""Benutzerdefinierte Funktionen für Textfeldvorlagen registrieren."""
import sqlite3

from sketcher.core.template_functions import (
    register_template_function,
)

DB_PFAD = "/home/du/produktions.db"


def naechste_seriennummer() -> str:
    """Die nächste Seriennummer aus der Datenbank abrufen."""
    conn = sqlite3.connect(DB_PFAD)
    try:
        cur = conn.execute(
            "UPDATE zaehler SET wert = wert + 1 "
            "WHERE name = 'serial' RETURNING wert"
        )
        row = cur.fetchone()
        conn.commit()
        return f"SN-{row[0]:06d}"
    finally:
        conn.close()


register_template_function(
    "naechste_seriennummer", naechste_seriennummer
)
```

Wichtige Punkte:

- Rufe `register_template_function(name, callable)` für
  jede Funktion auf.
- Deine Funktion kann alles tun, was Python kann: Dateien
  öffnen, Datenbankverbindungen herstellen, APIs aufrufen
  usw.
- Die Funktion wird bei **jeder Berechnung** aufgerufen,
  daher sollte sie schnell sein.
- Funktionen sind threadsicher, wenn dein Callable es ist.

### Rayforge mit dem Skript starten

Verwende das `--script`-Flag, um deine Funktionen vor dem
Fensterladen zu laden:

```bash
rayforge --script ~/.config/rayforge/meine_funktionen.py \
    mein_dokument.ryp
```

Dies führt dein Skript früh beim Start aus — bevor Addons
geladen und bevor das Hauptfenster erstellt wird — sodass
die Funktion verfügbar ist, wenn die Skizze zum ersten Mal
gelöst wird.

### Die Funktion in einem Textfeld verwenden

Erstelle ein Textfeld mit:

```
{naechste_seriennummer()}
```

Format-Spezifikationen funktionieren ebenfalls:

```
{naechste_seriennummer():>20}
```

### Funktionen programmatisch registrieren

Wenn du ein Addon oder eine wiederverwendbare Bibliothek
schreibst, kannst du `register_template_function` aus jedem
Python-Code aufrufen, der vor der Skizzenberechnung läuft:

```python
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function(
    "teilnummer",
    lambda: f"T-{hash('x') % 10000:04d}"
)
```

### Eingebaute Funktionen können nicht entfernt werden

Die eingebauten Funktionen (`today`, `now`, `uuid` usw.)
können nicht deregistriert werden. Wenn du ihr Verhalten
ändern möchtest, registriere eine Funktion mit einem anderen
Namen.
