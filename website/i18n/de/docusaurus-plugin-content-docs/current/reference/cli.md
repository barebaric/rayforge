---
description: "Referenz der Kommandozeile für Rayforge."
---

# Kommandozeile

Vollständige Referenz für Rayforge-Kommandozeilenoptionen.

```
rayforge [Optionen] [Dateinamen...]
```

---

## Positionale Argumente

| Argument     | Beschreibung                      |
| ------------ | --------------------------------- |
| `Dateinamen` | SVG- oder Bilddateien beim Start. |

---

## Optionen

| Option              | Beschreibung                            |
| ------------------- | --------------------------------------- |
| `--version`         | Version ausgeben und beenden.           |
| `-h`, `--help`      | Hilfe ausgeben und beenden.             |
| `--loglevel STUFE`  | Log-Stufe. Standard: `INFO`.            |
| `--config VERZ`     | Benutzerdefiniertes Config-Verzeichnis. |
| `--exit`            | Nach Import beenden.                    |
| `--vector`          | Import als direkte Vektoren erzwingen.  |
| `--trace`           | Import durch Bitmap-Tracing erzwingen.  |
| `--script SKRIPT`   | Frühes Startskript.                     |
| `--uiscript SKRIPT` | UI-Skript (nach Laden).                 |

---

## Beispiele

### Datei öffnen

```bash
rayforge meinprojekt.ryp
```

### Mehrere Dateien öffnen

```bash
rayforge teil1.svg logo.png design.ryp
```

### Import mit Tracing

```bash
rayforge --trace foto.png
```

### Frühes Skript ausführen und beenden

```bash
rayforge --exit --script registrieren.py \
    meinprojekt.ryp
```

### UI-Skript (Automatisierung)

```bash
rayforge --exit --uiscript screenshot.py \
    meinprojekt.ryp
```

### Stapelverarbeitung

```bash
rayforge --exit --vector eingabe.svg
```

---

## Frühe Skripte (`--script`)

Das `--script`-Flag führt ein Python-Skript **synchron beim
Start** aus, bevor Addons geladen und bevor das Hauptfenster
erstellt wird. Geeignet für:

- Plugins beim `pluggy`-Plugin-Manager registrieren
- Anwendungskontext konfigurieren
- Vorlagenfunktionen für Textfelder registrieren
- Umgebungsvariablen vor dem Start setzen

Das Skript hat Zugriff auf den Kontext über `get_context()`:

```python
from rayforge.context import get_context

ctx = get_context()
# Plugins registrieren, Dienste konfigurieren, usw.
```

### Beispiel: Benutzerdefinierte Vorlagenfunktion registrieren

```python
"""Benutzerdefinierte Funktion für Textfeldausdrücke registrieren.

Ausführen mit: rayforge --script registrieren_fn.py
"""
from rayforge.context import get_context
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function("meine_id", lambda: "TEIL-001")
```

Jetzt funktioniert `{meine_id()}` in jedem Textfeld.

Siehe
[Benutzerdefinierte Vorlagenfunktionen](../features/sketcher.md#custom-template-functions)
in der Sketcher-Dokumentation für ein vollständiges Tutorial.

---

## UI-Skripte (`--uiscript`)

Das `--uiscript`-Flag führt ein Python-Skript **nach dem
vollständigen Laden des Hauptfensters** in einem Hintergrundthread
aus. Geeignet für:

- Automatisiertes UI-Testing
- Screenshots der Anwendung
- End-to-End-Workflows

Das Skript kann die Anwendung und Fenster direkt importieren:

```python
from rayforge.uiscript import app, win
```

Das Skript läuft in einem **Hintergrundthread** — achte auf
Thread-Sicherheit beim Zugriff auf GTK-Widgets
(verwende `GLib.idle_add` für GTK-Operationen).

### Beispiel: Screenshot aufnehmen

```python
"""Screenshot des Hauptfensters aufnehmen."""
from rayforge.uiscript import app, win

import gi
gi.require_version("Gtk", "4.0")
from gi.repository import GLib

def capture():
    surface = win.get_surface()
    if surface:
        surface.write_to_png("/tmp/rayforge_screenshot.png")
    return GLib.SOURCE_REMOVE

GLib.idle_add(capture)
```

---

## Beide Flags verwenden

Beide `--script` und `--uiscript` können zusammen verwendet
werden. Das `--script` läuft zuerst (synchron), dann wird das
Fenster geladen, und dann läuft `--uiscript`:

```bash
rayforge --script fruehes_setup.py \
    --uiscript automatisierung.py \
    meinprojekt.ryp
```

Dies ist nützlich, wenn du zuerst Plugins registrieren und
später die UI steuern möchtest.
