# Addon-Entwicklung

Rayforge verwendet ein Addonsystem basierend auf [pluggy](https://pluggy.readthedocs.io/), um Entwicklern zu ermöglichen, Funktionalität zu erweitern, neue Maschinentreiber hinzuzufügen oder benutzerdefinierte Logik zu integrieren, ohne den Kern-Code zu ändern.

## 1. Schnellstart

Der schnellste Weg zum Starten ist die Verwendung des offiziellen Templates.

1. **Forke oder Klone** das [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template).
2. **Benenne** das Verzeichnis um und aktualisiere die Metadaten.

## 2. Addonstruktur

Der `AddonManager` scannt das `addons`-Verzeichnis. Ein gültiges Addon muss ein Verzeichnis mit einer Manifest-Datei sein:

**Verzeichnisstruktur:**

```text
mein-rayforge-addon/
├── rayforge-addon.yaml  <-- Erforderliches Manifest
├── mein_addon/          <-- Python-Paket
│   ├── __init__.py
│   ├── backend.py       <-- Backend-Einstiegspunkt
│   └── frontend.py      <-- Frontend-Einstiegspunkt (optional)
├── assets/              <-- Optionale Ressourcen
├── locales/             <-- Optionale Übersetzungen (.po-Dateien)
└── README.md
```

## 3. Das Manifest (`rayforge-addon.yaml`)

Diese Datei teilt Rayforge mit, wie Ihr Addon geladen werden soll.

```yaml
# rayforge-addon.yaml

# Eindeutiger Bezeichner für Ihr Addon (Verzeichnisname)
name: mein_benutzerdefiniertes_addon

# Menschenlesbarer Anzeigename
display_name: "Mein Benutzerdefiniertes Addon"

# Im UI angezeigte Beschreibung
description: "Fügt Unterstützung für den XYZ-Laserschneider hinzu."

# API-Version (muss mit Rayforge's PLUGIN_API_VERSION übereinstimmen)
api_version: 1

# Abhängigkeiten von Rayforge-Version
depends:
  - rayforge>=0.27.0,<2.0.0

# Optional: Abhängigkeiten von anderen Addons
requires:
  - ein-anderes-addon>=1.0.0

# Was das Addon bereitstellt
provides:
  # Backend-Modul (in Haupt- und Worker-Prozessen geladen)
  backend: mein_addon.backend
  # Frontend-Modul (nur im Hauptprozess geladen, für UI)
  frontend: mein_addon.frontend
  # Optionale Asset-Dateien
  assets:
    - path: assets/profiles.json
      type: profiles

# Autorenmetadaten
author:
  name: Max Mustermann
  email: max@example.com

url: https://github.com/username/mein-benutzerdefiniertes-addon
```

### Erforderliche Felder

- `name`: Eindeutiger Bezeichner (sollte mit Verzeichnisname übereinstimmen)
- `display_name`: Menschenlesbarer Name, der in der UI angezeigt wird
- `description`: Kurze Beschreibung der Addon-Funktionalität
- `api_version`: Muss `1` sein (entspricht Rayforge's `PLUGIN_API_VERSION`)
- `depends`: Liste von Versionseinschränkungen für Rayforge
- `author`: Objekt mit `name` (erforderlich) und `email` (optional)

### Optionale Felder

- `requires`: Liste von anderen Addon-Abhängigkeiten
- `provides`: Einstiegspunkte und Assets
- `url`: Projekt-Homepage oder Repository

## 4. Einstiegspunkte

Addons können zwei Arten von Einstiegspunkten bereitstellen:

### Backend (`provides.backend`)

Wird sowohl im Hauptprozess als auch in Worker-Prozessen geladen. Verwenden Sie dies für:
- Maschinentreiber
- Schritttypen
- Ops-Produzenten
- Kernfunktionalität ohne UI-Abhängigkeiten

### Frontend (`provides.frontend`)

Wird nur im Hauptprozess geladen. Verwenden Sie dies für:
- UI-Komponenten
- GTK-Widgets
- Menüeinträge
- Aktionen, die das Hauptfenster benötigen

Einstiegspunkte werden als gepunktete Modulpfade angegeben (z.B. `mein_addon.backend`).

## 5. Den Addoncode schreiben

Rayforge verwendet `pluggy`-Hooks. Um sich in Rayforge einzuklinken, definiere Funktionen, die mit `@pluggy.HookimplMarker("rayforge")` dekoriert sind.

### Grundlegendes Boilerplate (`backend.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Wird aufgerufen, wenn Rayforge vollständig initialisiert ist.
    Dies ist Ihr Haupteinstiegspunkt für den Zugriff auf Manager.
    """
    logger.info("Mein Benutzerdefiniertes Addon wurde gestartet!")

    machine = context.machine
    if machine:
        logger.info(f"Addon läuft auf Maschine: {machine.id}")

@hookimpl
def on_unload():
    """
    Wird aufgerufen, wenn das Addon deaktiviert oder entladen wird.
    Ressourcen bereinigen, Verbindungen schließen, Handler abmelden.
    """
    logger.info("Mein Benutzerdefiniertes Addon wird beendet")

@hookimpl
def register_machines(machine_manager):
    """
    Wird während des Starts aufgerufen, um neue Maschinentreiber zu registrieren.
    """
    from .my_driver import MyNewMachine
    machine_manager.register("my_new_machine", MyNewMachine)

@hookimpl
def register_steps(step_registry):
    """
    Wird aufgerufen, um benutzerdefinierte Schritttypen zu registrieren.
    """
    from .my_step import MyCustomStep
    step_registry.register("my_custom_step", MyCustomStep)

@hookimpl
def register_producers(producer_registry):
    """
    Wird aufgerufen, um benutzerdefinierte Ops-Produzenten zu registrieren.
    """
    from .my_producer import MyProducer
    producer_registry.register("my_producer", MyProducer)

@hookimpl
def register_step_widgets(widget_registry):
    """
    Wird aufgerufen, um benutzerdefinierte Schritteinstellungs-Widgets zu registrieren.
    """
    from .my_widget import MyStepWidget
    widget_registry.register("my_custom_step", MyStepWidget)

@hookimpl
def register_menu_items(menu_registry):
    """
    Wird aufgerufen, um Menüeinträge zu registrieren.
    """
    from .menu_items import register_menus
    register_menus(menu_registry)

@hookimpl
def register_commands(command_registry):
    """
    Wird aufgerufen, um Editor-Befehle zu registrieren.
    """
    from .commands import register_commands
    register_commands(command_registry)

@hookimpl
def register_actions(window):
    """
    Wird aufgerufen, um Fensteraktionen zu registrieren.
    """
    from .actions import setup_actions
    setup_actions(window)
```

### Verfügbare Hooks

Definiert in `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Haupteinstiegspunkt.** Wird nach dem Laden von Konfiguration, Kamera und Hardware aufgerufen.
  Verwende dies für Logik, UI-Injektionen oder Listener.

**`on_unload`** ()
: Wird aufgerufen, wenn ein Addon deaktiviert oder entladen wird. Verwenden Sie dies, um
  Ressourcen zu bereinigen, Verbindungen zu schließen, Handler abzumelden usw.

**`register_machines`** (`machine_manager`)
: Wird während des Starts aufgerufen, um neue Maschinentreiber zu registrieren.

**`register_steps`** (`step_registry`)
: Wird aufgerufen, um benutzerdefinierte Schritttypen zu registrieren.

**`register_producers`** (`producer_registry`)
: Wird aufgerufen, um benutzerdefinierte Ops-Produzenten zu registrieren.

**`register_step_widgets`** (`widget_registry`)
: Wird aufgerufen, um benutzerdefinierte Schritteinstellungs-Widgets zu registrieren.

**`register_menu_items`** (`menu_registry`)
: Wird aufgerufen, um Menüeinträge zu registrieren.

**`register_commands`** (`command_registry`)
: Wird aufgerufen, um Editor-Befehle zu registrieren.

**`register_actions`** (`window`)
: Wird aufgerufen, um Fensteraktionen zu registrieren.

## 6. Auf Rayforge-Daten zugreifen

Der `rayforge_init`-Hook stellt den **`RayforgeContext`** bereit. Über dieses Objekt kannst du auf Folgendes zugreifen:

- **`context.machine`**: Die aktuell aktive Maschineninstanz.
- **`context.config`**: Globale Konfigurationseinstellungen.
- **`context.config_mgr`**: Konfigurationsmanager.
- **`context.machine_mgr`**: Maschinenmanager (alle Maschinen).
- **`context.camera_mgr`**: Zugriff auf Kamera-Feeds und Computer-Vision-Tools.
- **`context.material_mgr`**: Zugriff auf die Materialbibliothek.
- **`context.recipe_mgr`**: Zugriff auf Verarbeitungsrezepte.
- **`context.dialect_mgr`**: G-Code-Dialekt-Manager.
- **`context.language`**: Aktueller Sprachcode für lokalisierte Inhalte.
- **`context.addon_mgr`**: Addon-Manager-Instanz.
- **`context.plugin_mgr`**: Plugin-Manager-Instanz.
- **`context.debug_dump_manager`**: Debug-Dump-Manager.
- **`context.artifact_store`**: Pipeline-Artefakt-Speicher.

## 7. Lokalisierung

Addons können Übersetzungen mit `.po`-Dateien bereitstellen:

```text
mein-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── mein_addon.po
│   └── es/
│       └── LC_MESSAGES/
│           └── mein_addon.po
```

`.po`-Dateien werden automatisch zu `.mo`-Dateien kompiliert, wenn das Addon
installiert oder geladen wird.

## 8. Entwicklung & Testen

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

## 9. Veröffentlichung

Um dein Addon mit der Community zu teilen:

1.  **Auf Git hosten:** Pushe deinen Code in ein öffentliches Git-Repository (GitHub, GitLab, etc.).
2.  **Bei der Registry einreichen:**
    - Gehe zu [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Forke das Repository.
    - Füge die Git-URL und Metadaten deines Addons zur Registry-Liste hinzu.
    - Reiche einen Pull Request ein.

Sobald akzeptiert, können Benutzer dein Addon direkt über die Rayforge-UI oder mit der Git-URL installieren.
