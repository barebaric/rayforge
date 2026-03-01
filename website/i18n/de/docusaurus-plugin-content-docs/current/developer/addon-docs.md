# Rayforge-Addon-Entwicklerleitfaden

Rayforge verwendet ein Addonsystem basierend auf [pluggy](https://pluggy.readthedocs.io/), um Entwicklern zu ermöglichen, Funktionalität zu erweitern, neue Maschinentreiber hinzuzufügen oder benutzerdefinierte Logik zu integrieren, ohne den Kern-Code zu ändern.

## 1. Schnellstart

Der schnellste Weg zum Starten ist die Verwendung des offiziellen Templates.

1. **Forke oder Klone** das [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template).
2. **Benenne** das Verzeichnis um und aktualisiere die Metadaten.

## 2. Addonstruktur

Der `AddonManager` scannt das `addons`-Verzeichnis. Ein gültiges Addon muss ein Verzeichnis sein, das mindestens zwei Dateien enthält:

1. `rayforge-addon.yaml` (Metadaten)
2. Einen Python-Einstiegspunkt (z.B. `addon.py`)

**Verzeichnisstruktur:**

```text
mein-rayforge-addon/
├── rayforge-addon.yaml  <-- Erforderliches Manifest
├── addon.py             <-- Einstiegspunkt (Logik)
├── assets/              <-- Optionale Ressourcen
└── README.md
```

## 3. Das Manifest (`rayforge-addon.yaml`)

Diese Datei teilt Rayforge mit, wie Ihr Addon geladen werden soll.

```yaml
# rayforge-addon.yaml

# Eindeutiger Bezeichner für Ihr Addon
name: mein_benutzerdefiniertes_addon

# Menschenlesbarer Anzeigename
display_name: "Mein Benutzerdefiniertes Addon"

# Im UI angezeigte Beschreibung
description: "Fügt Unterstützung für den XYZ-Laserschneider hinzu."

# Abhängigkeiten (Addon und Versionseinschränkungen)
depends:
  - rayforge>=0.27.0,~0.27

# Die zu ladende Python-Datei (relativ zum Addonordner)
entry_point: addon.py

# Autorenmetadaten
author: Max Mustermann
url: https://github.com/username/mein-benutzerdefiniertes-addon
```

## 4. Den Addoncode schreiben

Rayforge verwendet `pluggy`-Hooks. Um sich in Rayforge einzuklinken, definiere Funktionen, die mit `@pluggy.HookimplMarker("rayforge")` dekoriert sind.

### Grundlegendes Boilerplate (`addon.py`)

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
    logger.info("Mein Benutzerdefiniertes Addon wurde gestartet!")

    # Auf Kernsysteme über den Kontext zugreifen
    machine = context.machine
    camera = context.camera_mgr

    if machine:
        logger.info(f"Addon läuft auf Maschine: {machine.id}")

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

Um dein Addon lokal zu testen ohne es zu veröffentlichen:

1.  **Lokalisieren dein Konfigurationsverzeichnis:**
    Rayforge verwendet `platformdirs`.

    - **Windows:** `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`
    - **macOS:** `~/Library/Application Support/rayforge/addons`
    - **Linux:** `~/.config/rayforge/addons`
       _(Prüfe die Logs beim Start auf `Config dir is ...`)_

2.  **Verlinke dein Addon symbolisch:**
    Anstatt Dateien hin- und her zu kopieren, erstelle einen symbolischen Link von deinem Entwicklungsordner zum Rayforge-Addonsordner.

    _Linux/macOS:_

    ```bash
    ln -s /pfad/zu/mein-rayforge-addon ~/.config/rayforge/addons/mein-rayforge-addon
    ```

3.  **Rayforge neu starten:**
    Die Anwendung scannt das Verzeichnis beim Start. Prüfe die Konsolen-Logs auf:
    > `Loaded addon: mein_benutzerdefiniertes_addon`

## 7. Veröffentlichung

Um dein Addon mit der Community zu teilen:

1.  **Auf Git hosten:** Pushe deinen Code in ein öffentliches Git-Repository (GitHub, GitLab, etc.).
2.  **Bei der Registry einreichen:**
    - Gehe zu [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Forke das Repository.
    - Füge die Git-URL und Metadaten deines Addons zur Registry-Liste hinzu.
    - Reiche einen Pull Request ein.

Sobald akzeptiert, können Benutzer dein Addon direkt über die Rayforge-UI oder mit der Git-URL installieren.
