# Rayforge-Paket-Entwicklerleitfaden

Rayforge verwendet ein Paketsystem basierend auf [pluggy](https://pluggy.readthedocs.io/), um Entwicklern zu ermöglichen, Funktionalität zu erweitern, neue Maschinentreiber hinzuzufügen oder benutzerdefinierte Logik zu integrieren, ohne den Kern-Code zu ändern.

## 1. Schnellstart

Der schnellste Weg zum Starten ist die Verwendung des offiziellen Templates.

1. **Forke oder Klone** das [rayforge-package-template](https://github.com/barebaric/rayforge-package-template).
2. **Benenne** das Verzeichnis um und aktualisiere die Metadaten.

## 2. Paketstruktur

Der `PackageManager` scannt das `packages`-Verzeichnis. Ein gültiges Paket muss ein Verzeichnis sein, das mindestens zwei Dateien enthält:

1. `rayforge_package.yaml` (Metadaten)
2. Einen Python-Einstiegspunkt (z.B. `package.py`)

**Verzeichnisstruktur:**

```text
mein-rayforge-paket/
├── rayforge_package.yaml  <-- Erforderliches Manifest
├── package.py             <-- Einstiegspunkt (Logik)
├── assets/                <-- Optionale Ressourcen
└── README.md
```

## 3. Das Manifest (`rayforge_package.yaml`)

Diese Datei teilt Rayforge mit, wie Ihr Paket geladen werden soll.

```yaml
# rayforge_package.yaml

# Eindeutiger Bezeichner für Ihr Paket
name: mein_benutzerdefiniertes_paket

# Menschenlesbarer Anzeigename
display_name: "Mein Benutzerdefiniertes Paket"

# Versionszeichenfolge
version: 0.1.0

# Im UI angezeigte Beschreibung
description: "Fügt Unterstützung für den XYZ-Laserschneider hinzu."

# Abhängigkeiten (Paket und Versionseinschränkungen)
depends:
  - rayforge>=0.27.0,~0.27

# Die zu ladende Python-Datei (relativ zum Paketordner)
entry_point: package.py

# Autorenmetadaten
author: Max Mustermann
url: https://github.com/username/mein-benutzerdefiniertes-paket
```

## 4. Den Paketcode schreiben

Rayforge verwendet `pluggy`-Hooks. Um sich in Rayforge einzuklinken, definiere Funktionen, die mit `@pluggy.HookimplMarker("rayforge")` dekoriert sind.

### Grundlegendes Boilerplate (`package.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

# Den Hook-Implementierungs-Marker definieren
hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Wird aufgerufen, wenn Rayforge vollständig initialisiert ist.
    Dies ist Ihr Haupteinstiegspunkt für den Zugriff auf Manager.
    """
    logger.info("Mein Benutzerdefiniertes Paket wurde gestartet!")

    # Auf Kernsysteme über den Kontext zugreifen
    machine = context.machine
    camera = context.camera_mgr

    if machine:
        logger.info(f"Paket läuft auf Maschine: {machine.id}")

@hookimpl
def register_machines(machine_manager):
    """
    Wird während des Starts aufgerufen, um neue Maschinentreiber zu registrieren.
    """
    # from .my_driver import MyNewMachine
    # machine_manager.register("my_new_machine", MyNewMachine)
    pass
```

### Verfügbare Hooks

Definiert in `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Haupteinstiegspunkt.** Wird nach dem Laden von Konfiguration, Kamera und Hardware aufgerufen.
  Verwende dies für Logik, UI-Injektionen oder Listener.

**`register_machines`** (`machine_manager`)
: Wird früh im Boot-Prozess aufgerufen. Verwende dies, um neue Hardwareklassen/Treiber zu registrieren.

## 5. Auf Rayforge-Daten zugreifen

Der `rayforge_init`-Hook stellt den **`RayforgeContext`** bereit. Über dieses Objekt kannst du auf Folgendes zugreifen:

- **`context.machine`**: Die aktuell aktive Maschineninstanz.
- **`context.config`**: Globale Konfigurationseinstellungen.
- **`context.camera_mgr`**: Zugriff auf Kamera-Feeds und Computer-Vision-Tools.
- **`context.material_mgr`**: Zugriff auf die Materialbibliothek.
- **`context.recipe_mgr`**: Zugriff auf Verarbeitungsrezepte.

## 6. Entwicklung & Testen

Um dein Paket lokal zu testen ohne es zu veröffentlichen:

1.  **Lokalisieren dein Konfigurationsverzeichnis:**
    Rayforge verwendet `platformdirs`.

    - **Windows:** `C:\Users\<User>\AppData\Local\rayforge\rayforge\packages`
    - **macOS:** `~/Library/Application Support/rayforge/packages`
    - **Linux:** `~/.config/rayforge/packages`
       _(Prüfe die Logs beim Start auf `Config dir is ...`)_

2.  **Verlinke dein Paket symbolisch:**
    Anstatt Dateien hin- und her zu kopieren, erstelle einen symbolischen Link von deinem Entwicklungsordner zum Rayforge-Paketeordner.

    _Linux/macOS:_

    ```bash
    ln -s /pfad/zu/mein-rayforge-paket ~/.config/rayforge/packages/mein-rayforge-paket
    ```

3.  **Rayforge neu starten:**
    Die Anwendung scannt das Verzeichnis beim Start. Prüfe die Konsolen-Logs auf:
    > `Loaded package: mein_benutzerdefiniertes_paket`

## 7. Veröffentlichung

Um dein Paket mit der Community zu teilen:

1.  **Auf Git hosten:** Pushe deinen Code in ein öffentliches Git-Repository (GitHub, GitLab, etc.).
2.  **Bei der Registry einreichen:**
    - Gehe zu [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Forke das Repository.
    - Füge die Git-URL und Metadaten deines Pakets zur Registry-Liste hinzu.
    - Reiche einen Pull Request ein.

Sobald akzeptiert, können Benutzer dein Paket direkt über die Rayforge-UI oder mit der Git-URL installieren.
