# Addon-Entwicklung Übersicht

Rayforge verwendet ein auf [pluggy](https://pluggy.readthedocs.io/) basierendes Addon-System, mit dem du Funktionen erweitern, neue Maschinentreiber hinzufügen oder benutzerdefinierte Logik integrieren kannst, ohne den Kerncode zu ändern.

## Schnellstart

Der schnellste Weg zum Einstieg ist das offizielle [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template). Forke oder klone es, benenne das Verzeichnis um und aktualisiere die Metadaten entsprechend deinem Addon.

## Wie Addons funktionieren

Der `AddonManager` durchsucht das Verzeichnis `addons` nach gültigen Addons. Ein Addon ist einfach ein Verzeichnis, das eine `rayforge-addon.yaml`-Manifestdatei zusammen mit deinem Python-Code enthält.

So sieht ein typisches Addon aus:

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Erforderliches Manifest
├── my_addon/            <-- Dein Python-Paket
│   ├── __init__.py
│   ├── backend.py       <-- Backend-Einstiegspunkt
│   └── frontend.py      <-- Frontend-Einstiegspunkt (optional)
├── assets/              <-- Optionale Ressourcen
├── locales/             <-- Optionale Übersetzungen (.po-Dateien)
└── README.md
```

## Dein erstes Addon

Lass uns ein einfaches Addon erstellen, das einen benutzerdefinierten Maschinentreiber registriert. Erstelle zuerst das Manifest:

```yaml title="rayforge-addon.yaml"
name: my_laser_driver
display_name: "My Laser Driver"
description: "Adds support for the XYZ laser cutter."
api_version: 9

author:
  name: Jane Doe
  email: jane@example.com

provides:
  backend: my_addon.backend
```

Erstelle nun das Backend-Modul, das deinen Treiber registriert:

```python title="my_addon/backend.py"
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def register_machines(machine_manager):
    """Register our custom machine driver."""
    from .my_driver import MyLaserMachine
    machine_manager.register("my_laser", MyLaserMachine)
```

Das war's! Dein Addon wird jetzt geladen, wenn Rayforge startet, und dein Maschinentreiber steht den Benutzern zur Verfügung.

Die [Manifest](./addon-manifest.md)-Dokumentation behandelt alle verfügbaren Konfigurationsoptionen.

## Einstiegspunkte verstehen

Addons können zwei Einstiegspunkte bereitstellen, die zu unterschiedlichen Zeiten geladen werden:

Der **Backend**-Einstiegspunkt wird sowohl im Hauptprozess als auch in Worker-Prozessen geladen. Verwende diesen für Maschinentreiber, Schritttypen, Ops-Produzenten und Transformatoren oder jede Kernfunktionalität, die keine UI-Abhängigkeiten benötigt.

Der **Frontend**-Einstiegspunkt wird nur im Hauptprozess geladen. Hier platzierst du UI-Komponenten, GTK-Widgets, Menüeinträge und alles, was Zugriff auf das Hauptfenster benötigt.

Beide werden als gepunktete Modulpfade wie `my_addon.backend` angegeben.

## Verbindung zu Rayforge mit Hooks

Rayforge verwendet `pluggy`-Hooks, um Addons in die Anwendung zu integrieren. Dekoriere einfach deine Funktionen mit `@pluggy.HookimplMarker("rayforge")`:

```python
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def rayforge_init(context: RayforgeContext):
    """Called when Rayforge is fully initialized."""
    # Your setup code here
    pass

@hookimpl
def on_unload():
    """Called when the addon is being disabled or unloaded."""
    # Clean up resources here
    pass
```

Die [Hooks](./addon-hooks.md)-Dokumentation beschreibt jeden verfügbaren Hook und wann er aufgerufen wird.

## Deine Komponenten registrieren

Die meisten Hooks erhalten ein Registrierungsobjekt, mit dem du deine benutzerdefinierten Komponenten registrierst:

```python
@hookimpl
def register_steps(step_registry):
    from .my_step import MyCustomStep
    step_registry.register(MyCustomStep)

@hookimpl
def register_actions(action_registry):
    from .actions import setup_actions
    setup_actions(action_registry)
```

Die [Registries](./addon-registries.md)-Dokumentation erklärt jede Registrierung und wie du sie verwendest.

## Auf Rayforges Daten zugreifen

Der `rayforge_init`-Hook gibt dir Zugriff auf ein `RayforgeContext`-Objekt. Über diesen Kontext kannst du auf alles in Rayforge zugreifen:

Du kannst die aktuell aktive Maschine über `context.machine` abrufen oder auf alle Maschinen über `context.machine_mgr` zugreifen. Das Objekt `context.config` enthält globale Einstellungen, während `context.camera_mgr` Zugriff auf Kamera-Feeds bietet. Für Materialien verwendest du `context.material_mgr` und für Verarbeitungsrezepte `context.recipe_mgr`. Der G-Code-Dialekt-Manager ist als `context.dialect_mgr` verfügbar, und KI-Funktionen laufen über `context.ai_provider_mgr`. Für die Lokalisierung prüfst du `context.language` auf den aktuellen Sprachcode. Der Addon-Manager selbst ist als `context.addon_mgr` verfügbar, und wenn du kostenpflichtige Addons erstellst, übernimmt `context.license_validator` die Lizenzvalidierung.

## Übersetzungen hinzufügen

Addons können Übersetzungen mit Standard-`.po`-Dateien bereitstellen. Organisiere sie wie folgt:

```text
my-rayforge-addon/
├── locales/
│   ├── de/
│   │   └── LC_MESSAGES/
│   │       └── my_addon.po
│   └── es/
│       └── LC_MESSAGES/
│           └── my_addon.po
```

Rayforge kompiliert `.po`-Dateien automatisch zu `.mo`-Dateien, wenn dein Addon geladen wird.

## Testen während der Entwicklung

Um dein Addon lokal zu testen, erstelle einen symbolischen Link von deinem Entwicklungsordner zum Addons-Verzeichnis von Rayforge.

Finde zuerst dein Konfigurationsverzeichnis. Unter Windows ist es `C:\Users\<Benutzer>\AppData\Local\rayforge\rayforge\addons`. Unter macOS suche in `~/Library/Application Support/rayforge/addons`. Unter Linux ist es `~/.config/rayforge/addons`.

Erstelle dann den Symlink:

```bash
ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
```

Starte Rayforge neu und prüfe die Konsole auf eine Meldung wie `Loaded addon: my_laser_driver`.

## Dein Addon teilen

Wenn du bereit bist, dein Addon zu teilen, pushe es in ein öffentliches Git-Repository auf GitHub oder GitLab. Reiche es dann beim [rayforge-registry](https://github.com/barebaric/rayforge-registry) ein, indem du das Repository forkest, die Metadaten deines Addons hinzufügst und einen Pull Request öffnest.

Nach der Annahme können Benutzer dein Addon direkt über den Addon-Manager von Rayforge installieren.
