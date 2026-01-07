# Rayforge Package Developer Guide

Rayforge uses a package system based on [pluggy](https://pluggy.readthedocs.io/)
to allow developers to extend functionality, add new machine drivers, or
integrate custom logic without modifying the core codebase.

## 1. Quick Start

The fastest way to start is using the official template.

1. **Fork or Clone** the
   [rayforge-package-template](https://github.com/barebaric/rayforge-package-template).
2. **Rename** the directory and update the metadata.

## 2. Package Structure

The `PackageManager` scans the `packages` directory. A valid package must be a
directory containing at least two files:

1. `rayforge_package.yaml` (Metadata)
2. A Python entry point (e.g., `package.py`)

**Directory Layout:**

```text
my-rayforge-package/
├── rayforge_package.yaml  <-- Required Manifest
├── package.py             <-- Entry point (logic)
├── assets/                <-- Optional resources
└── README.md
```

## 3. The Manifest (`rayforge_package.yaml`)

This file tells Rayforge how to load your package.

```yaml
# rayforge_package.yaml

# Unique identifier for your package
name: my_custom_package

# Human-readable display name
display_name: "My Custom Package"

# Version string
version: 0.1.0

# Description displayed in the UI
description: "Adds support for the XYZ laser cutter."

# Dependencies (package and version constraints)
depends:
  - rayforge>=0.27.0,~0.27

# The python file to load (relative to the package folder)
entry_point: package.py

# Author metadata
author: Jane Doe
url: https://github.com/username/my-custom-package
```

## 4. Writing the Package Code

Rayforge uses `pluggy` hooks. To hook into Rayforge, define functions decorated
with `@pluggy.HookimplMarker("rayforge")`.

### Basic Boilerplate (`package.py`)

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
    logger.info("My Custom Package has started!")

    # Access core systems via the context
    machine = context.machine
    camera = context.camera_mgr

    if machine:
        logger.info(f"Package running on machine: {machine.id}")

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

To test your package locally without publishing it:

1.  **Locate your Configuration Directory:**
    Rayforge uses `platformdirs`.

    - **Windows:** `C:\Users\<User>\AppData\Local\rayforge\rayforge\packages`
    - **macOS:** `~/Library/Application Support/rayforge/packages`
    - **Linux:** `~/.config/rayforge/packages`
      _(Check the logs on startup for `Config dir is ...`)_

2.  **Symlink your package:**
    Instead of copying files back and forth, create a symbolic link from your dev
    folder to the Rayforge packages folder.

    _Linux/macOS:_

    ```bash
    ln -s /path/to/my-rayforge-package ~/.config/rayforge/packages/my-rayforge-package
    ```

3.  **Restart Rayforge:**
    The application scans the directory on startup. Check the console logs for:
    > `Loaded package: my_custom_package`

## 7. Publishing

To share your package with the community:

1.  **Host on Git:** Push your code to a public Git repository (GitHub, GitLab,
    etc.).
2.  **Submit to Registry:**
    - Go to [rayforge-registry](https://github.com/barebaric/rayforge-registry).
    - Fork the repository.
    - Add your package's Git URL and metadata to the registry list.
    - Submit a Pull Request.

Once accepted, users can install your package directly via the Rayforge UI or by
using the Git URL.
