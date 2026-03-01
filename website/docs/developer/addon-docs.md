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
directory containing at least two files:

1. `rayforge-addon.yaml` (Metadata)
2. A Python entry point (e.g., `addon.py`)

**Directory Layout:**

```text
my-rayforge-addon/
├── rayforge-addon.yaml  &lt;-- Required Manifest
├── addon.py             &lt;-- Entry point (logic)
├── assets/              &lt;-- Optional resources
└── README.md
```

## 3. The Manifest (`rayforge-addon.yaml`)

This file tells Rayforge how to load your addon.

```yaml
# rayforge-addon.yaml

# Unique identifier for your addon
name: my_custom_addon

# Human-readable display name
display_name: "My Custom Addon"

# Version string
version: 0.1.0

# Description displayed in the UI
description: "Adds support for the XYZ laser cutter."

# Dependencies (addon and version constraints)
depends:
  - rayforge&gt;=0.27.0,~0.27

# The python file to load (relative to the addon folder)
entry_point: addon.py

# Author metadata
author: Jane Doe
url: https://github.com/username/my-custom-addon
```

## 4. Writing the Addon Code

Rayforge uses `pluggy` hooks. To hook into Rayforge, define functions decorated
with `@pluggy.HookimplMarker("rayforge")`.

### Basic Boilerplate (`addon.py`)

```python
import logging
import pluggy
from rayforge.context import RayforgeContext

# Define the hook implementation marker
hookimpl = pluggy.HookimplMarker("rayforge")
logger = logging.getLogger(__name__)

@hookimpl
def rayforge_init(context: RayforgeContext):
    """
    Called when Rayforge is fully initialized.
    This is your main entry point to access managers.
    """
    logger.info("My Custom Addon has started!")

    # Access core systems via the context
    machine = context.machine
    camera = context.camera_mgr

    if machine:
        logger.info(f"Addon running on machine: {machine.id}")

@hookimpl
def register_machines(machine_manager):
    """
    Called during startup to register new machine drivers.
    """
    # from .my_driver import MyNewMachine
    # machine_manager.register("my_new_machine", MyNewMachine)
    pass
```

### Available Hooks

Defined in `rayforge/core/hooks.py`:

**`rayforge_init`** (`context`)
: **Main Entry Point.** Called after config, camera, and hardware are loaded.
  Use this for logic, UI injections, or listeners.

**`register_machines`** (`machine_manager`)
: Called early in the boot process. Use this to register new hardware
  classes/drivers.

## 5. Accessing Rayforge Data

The `rayforge_init` hook provides the **`RayforgeContext`**. Through this object,
you can access:

- **`context.machine`**: The currently active machine instance.
- **`context.config`**: Global configuration settings.
- **`context.camera_mgr`**: Access camera feeds and computer vision tools.
- **`context.material_mgr`**: Access the material library.
- **`context.recipe_mgr`**: Access processing recipes.

## 6. Development & Testing

To test your addon locally without publishing it:

1.  **Locate your Configuration Directory:**
    Rayforge uses `platformdirs`.

    - **Windows:** `C:\Users\&lt;User&gt;\AppData\Local\rayforge\rayforge\addons`
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

## 7. Publishing

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
