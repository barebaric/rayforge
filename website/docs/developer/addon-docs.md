# Rayforge Addon Developer Guide

Rayforge uses an addon system based on [pluggy](https://pluggy.readthedocs.io/)
to allow developers to extend functionality, add new machine drivers, or
integrate custom logic without modifying the core codebase.

## 1. Quick Start

The fastest way to start is using the official template.

1. **Fork or Clone** the
   [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template).
2. **Rename** the directory and update the metadata.

## 2. Addon Structure

The `AddonManager` scans the `addons` directory. A valid addon must be a
directory containing a manifest file:

**Directory Layout:**

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Required Manifest
├── my_addon/            <-- Python package
│   ├── __init__.py
│   ├── backend.py       <-- Backend entry point
│   └── frontend.py      <-- Frontend entry point (optional)
├── assets/              <-- Optional resources
├── locales/             <-- Optional translations (.po files)
└── README.md
```

## 3. The Manifest (`rayforge-addon.yaml`)

This file tells Rayforge how to load your addon.

```yaml
# rayforge-addon.yaml

# Unique identifier for your addon (directory name)
name: my_custom_addon

# Human-readable display name
display_name: "My Custom Addon"

# Description displayed in the UI
description: "Adds support for the XYZ laser cutter."

# API version (must match Rayforge's PLUGIN_API_VERSION)
api_version: 1

# Dependencies on Rayforge version
depends:
  - rayforge>=0.27.0,<2.0.0

# Optional: Dependencies on other addons
requires:
  - some-other-addon>=1.0.0

# What the addon provides
provides:
  # Backend module (loaded in both main and worker processes)
  backend: my_addon.backend
  # Frontend module (loaded only in main process, for UI)
  frontend: my_addon.frontend
  # Optional asset files
  assets:
    - path: assets/profiles.json
      type: profiles

# Author metadata
author:
  name: Jane Doe
  email: jane@example.com

url: https://github.com/username/my-custom-addon
```

### Required Fields

- `name`: Unique identifier (should match directory name)
- `display_name`: Human-readable name shown in UI
- `description`: Brief description of addon functionality
- `api_version`: Must be `1` (matches Rayforge's `PLUGIN_API_VERSION`)
- `depends`: List of version constraints for Rayforge
- `author`: Object with `name` (required) and `email` (optional)

### Optional Fields

- `requires`: List of other addon dependencies
- `provides`: Entry points and assets
- `url`: Project homepage or repository

## 4. Entry Points

Addons can provide two types of entry points:

### Backend (`provides.backend`)

Loaded in both the main process and worker processes. Use this for:
- Machine drivers
- Step types
- Ops producers
- Core functionality without UI dependencies

### Frontend (`provides.frontend`)

Loaded only in the main process. Use this for:
- UI components
- GTK widgets
- Menu items
- Actions that require the main window

Entry points are specified as dotted module paths (e.g., `my_addon.backend`).

## 5. Writing the Addon Code

Rayforge uses `pluggy` hooks. To hook into Rayforge, define functions decorated
with `@pluggy.HookimplMarker("rayforge")`.

### Basic Boilerplate (`backend.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Called when Rayforge is fully initialized.
    This is your main entry point to access managers.
    """
    logger.info("My Custom Addon has started!")

    machine = context.machine
    if machine:
        logger.info(f"Addon running on machine: {machine.id}")

@hookimpl
def on_unload():
    """
    Called when the addon is being disabled or unloaded.
    Clean up resources, close connections, unregister handlers.
    """
    logger.info("My Custom Addon is shutting down")

@hookimpl
def register_machines(machine_manager):
    """
    Called during startup to register new machine drivers.
    """
    from .my_driver import MyNewMachine
    machine_manager.register("my_new_machine", MyNewMachine)

@hookimpl
def register_steps(step_registry):
    """
    Called to register custom step types.
    """
    from .my_step import MyCustomStep
    step_registry.register("my_custom_step", MyCustomStep)

@hookimpl
def register_producers(producer_registry):
    """
    Called to register custom ops producers.
    """
    from .my_producer import MyProducer
    producer_registry.register("my_producer", MyProducer)

@hookimpl
def register_step_widgets(widget_registry):
    """
    Called to register custom step settings widgets.
    """
    from .my_widget import MyStepWidget
    widget_registry.register("my_custom_step", MyStepWidget)

@hookimpl
def register_menu_items(menu_registry):
    """
    Called to register menu items.
    """
    from .menu_items import register_menus
    register_menus(menu_registry)

@hookimpl
def register_commands(command_registry):
    """
    Called to register editor commands.
    """
    from .commands import register_commands
    register_commands(command_registry)

@hookimpl
def register_actions(window):
    """
    Called to register window actions.
    """
    from .actions import setup_actions
    setup_actions(window)
```

### Available Hooks

Defined in `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Main Entry Point.** Called after config, camera, and hardware are loaded.
  Use this for logic, UI injections, or listeners.

**`on_unload`** ()
: Called when an addon is being disabled or unloaded. Use this to clean up
  resources, close connections, unregister handlers, etc.

**`register_machines`** (`machine_manager`)
: Called during startup to register new machine drivers.

**`register_steps`** (`step_registry`)
: Called to allow plugins to register custom step types.

**`register_producers`** (`producer_registry`)
: Called to allow plugins to register custom ops producers.

**`register_step_widgets`** (`widget_registry`)
: Called to allow plugins to register custom step settings widgets.

**`register_menu_items`** (`menu_registry`)
: Called to allow plugins to register menu items.

**`register_commands`** (`command_registry`)
: Called to allow plugins to register editor commands.

**`register_actions`** (`window`)
: Called to allow plugins to register window actions.

## 6. Accessing Rayforge Data

The `rayforge_init` hook provides the **`RayforgeContext`**. Through this object,
you can access:

- **`context.machine`**: The currently active machine instance.
- **`context.config`**: Global configuration settings.
- **`context.config_mgr`**: Configuration manager.
- **`context.machine_mgr`**: Machine manager (all machines).
- **`context.camera_mgr`**: Access camera feeds and computer vision tools.
- **`context.material_mgr`**: Access the material library.
- **`context.recipe_mgr`**: Access processing recipes.
- **`context.dialect_mgr`**: G-code dialect manager.
- **`context.language`**: Current language code for localized content.
- **`context.addon_mgr`**: Addon manager instance.
- **`context.plugin_mgr`**: Plugin manager instance.
- **`context.debug_dump_manager`**: Debug dump manager.
- **`context.artifact_store`**: Pipeline artifact store.

## 7. Localization

Addons can provide translations using `.po` files:

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

`.po` files are automatically compiled to `.mo` files when the addon is
installed or loaded.

## 8. Development & Testing

To test your addon locally without publishing it:

1.  **Locate your Configuration Directory:**
    Rayforge uses `platformdirs`.

    - **Windows:** `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`
    - **macOS:** `~/Library/Application Support/rayforge/addons`
    - **Linux:** `~/.config/rayforge/addons`
      _(Check the logs on startup for `Config dir is ...`)_

2.  **Symlink your addon:**
    Instead of copying files back and forth, create a symbolic link from your dev
    folder to the Rayforge addons folder.

    _Linux/macOS:_

    ```bash
    ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
    ```

3.  **Restart Rayforge:**
    The application scans the directory on startup. Check the console logs for:
    > `Loaded addon: my_custom_addon`

## 9. Publishing

To share your addon with the community:

1.  **Host on Git:** Push your code to a public Git repository (GitHub, GitLab,
    etc.).
2.  **Submit to Registry:**
    - Go to [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Fork the repository.
    - Add your addon's Git URL and metadata to the registry list.
    - Submit a Pull Request.

Once accepted, users can install your addon directly via the Rayforge UI or by
using the Git URL.
