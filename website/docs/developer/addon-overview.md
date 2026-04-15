---
description: "Develop add-ons for Rayforge. Learn the add-on system, hooks, manifests, and how to extend Rayforge with custom functionality."
---

# Addon Development Overview

Rayforge uses an addon system based on [pluggy](https://pluggy.readthedocs.io/) that lets you extend functionality, add new machine drivers, or integrate custom logic without modifying the core codebase.

## Quick Start

The fastest way to get started is with the official [rayforge-addon-template](https://github.com/barebaric/rayforge-addon-template). Fork or clone it, rename the directory, and update the metadata to match your addon.

## How Addons Work

The `AddonManager` scans the `addons` directory for valid addons. An addon is simply a directory containing a `rayforge-addon.yaml` manifest file along with your Python code.

Here's what a typical addon looks like:

```text
my-rayforge-addon/
├── rayforge-addon.yaml  <-- Required manifest
├── my_addon/            <-- Your Python package
│   ├── __init__.py
│   ├── backend.py       <-- Backend entry point
│   └── frontend.py      <-- Frontend entry point (optional)
├── assets/              <-- Optional resources
├── locales/             <-- Optional translations (.po files)
└── README.md
```

## Your First Addon

Let's create a simple addon that registers a custom machine driver. First, create the manifest:

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

Now create the backend module that registers your driver:

```python title="my_addon/backend.py"
import pluggy

hookimpl = pluggy.HookimplMarker("rayforge")

@hookimpl
def register_machines(machine_manager):
    """Register our custom machine driver."""
    from .my_driver import MyLaserMachine
    machine_manager.register("my_laser", MyLaserMachine)
```

That's it! Your addon will now be loaded when Rayforge starts, and your machine driver will be available to users.

The [Manifest](./addon-manifest.md) documentation covers all available configuration options.

## Understanding Entry Points

Addons can provide two entry points, each loaded at different times:

The **backend** entry point loads in both the main process and worker processes. Use this for machine drivers, step types, ops producers and transformers, or any core functionality that doesn't need UI dependencies.

The **frontend** entry point only loads in the main process. This is where you'd put UI components, GTK widgets, menu items, and anything that needs access to the main window.

Both are specified as dotted module paths like `my_addon.backend`.

## Connecting to Rayforge with Hooks

Rayforge uses `pluggy` hooks to let addons integrate with the application. Simply decorate your functions with `@pluggy.HookimplMarker("rayforge")`:

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

The [Hooks](./addon-hooks.md) documentation describes every available hook and when it's called.

## Registering Your Components

Most hooks receive a registry object that you use to register your custom components:

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

The [Registries](./addon-registries.md) documentation explains each registry and how to use them.

## Accessing Rayforge's Data

The `rayforge_init` hook gives you access to a `RayforgeContext` object. Through this context, you can reach everything in Rayforge:

You can get the currently active machine via `context.machine`, or access all machines through `context.machine_mgr`. The `context.config` object holds global settings, while `context.camera_mgr` provides access to camera feeds. For materials, use `context.material_mgr`, and for processing recipes, use `context.recipe_mgr`. The G-code dialect manager is available as `context.dialect_mgr`, and AI features go through `context.ai_provider_mgr`. For localization, check `context.language` for the current language code. The addon manager itself is available as `context.addon_mgr`, and if you're building paid addons, `context.license_validator` handles license validation.

## Adding Translations

Addons can provide translations using standard `.po` files. Organize them like this:

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

Rayforge automatically compiles `.po` files to `.mo` files when your addon is loaded.

## Testing During Development

To test your addon locally, create a symbolic link from your development folder to Rayforge's addons directory.

First, find your configuration directory. On Windows, it's `C:\Users\<User>\AppData\Local\rayforge\rayforge\addons`. On macOS, look in `~/Library/Application Support/rayforge/addons`. On Linux, it's `~/.config/rayforge/addons`.

Then create the symlink:

```bash
ln -s /path/to/my-rayforge-addon ~/.config/rayforge/addons/my-rayforge-addon
```

Restart Rayforge and check the console for a message like `Loaded addon: my_laser_driver`.

## Sharing Your Addon

When you're ready to share your addon, push it to a public Git repository on GitHub or GitLab. Then submit it to the [rayforge-registry](https://github.com/barebaric/rayforge-registry) by forking the repository, adding your addon's metadata, and opening a pull request.

Once accepted, users can install your addon directly through Rayforge's addon manager.
