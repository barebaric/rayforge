---
description: "Command-line interface reference for Rayforge."
---

# Command Line

Complete reference for Rayforge's command-line options.

```
rayforge [options] [filenames...]
```

---

## Positional Arguments

| Argument    | Description                           |
| ----------- | ------------------------------------- |
| `filenames` | SVG or image files to open on launch. |

---

## Options

| Option              | Description                     |
| ------------------- | ------------------------------- |
| `--version`         | Print version and exit.         |
| `-h`, `--help`      | Show help and exit.             |
| `--loglevel LEVEL`  | Logging level. Default: `INFO`. |
| `--config DIR`      | Custom config directory.        |
| `--exit`            | Exit after import settles.      |
| `--vector`          | Force direct vector import.     |
| `--trace`           | Force bitmap trace import.      |
| `--script SCRIPT`   | Early startup script.           |
| `--uiscript SCRIPT` | UI script (post-load).          |

---

## Examples

### Open a file

```bash
rayforge myproject.ryp
```

### Open multiple files

```bash
rayforge part1.svg logo.png design.ryp
```

### Import with tracing

```bash
rayforge --trace photo.png
```

### Run an early script and exit

```bash
rayforge --exit --script register_functions.py \
    myproject.ryp
```

### Run a UI script (automation)

```bash
rayforge --exit --uiscript screenshot.py \
    myproject.ryp
```

### Batch export

```bash
rayforge --exit --vector input.svg
```

---

## Early Scripts (`--script`)

The `--script` flag runs a Python script **synchronously during startup**, before addons are loaded
and before the main window is created. This makes it the right place for:

- Registering plugins with the `pluggy` plugin manager
- Configuring the application context
- Registering template functions for text boxes
- Setting environment variables before the app starts

The script has access to the context via `get_context()`:

```python
from rayforge.context import get_context

ctx = get_context()
# Register plugins, configure services, etc.
```

### Example: Register a custom template function

```python
"""Register a custom function for text box expressions.

Run with: rayforge --script register_fn.py
"""
from rayforge.context import get_context
from sketcher.core.template_functions import (
    register_template_function,
)

register_template_function("myid", lambda: "PART-001")
```

Now `{myid()}` works in any text box.

See [Custom Template Functions](../features/sketcher/expressions.md#custom-template-functions) in
the Sketcher docs for a full tutorial.

---

## UI Scripts (`--uiscript`)

The `--uiscript` flag runs a Python script **after the main window is fully mapped and loaded**, in
a background thread. This makes it the right place for:

- Automated UI testing
- Taking screenshots of the application
- Running end-to-end workflows

The script can import the application and window directly:

```python
from rayforge.uiscript import app, win
```

The script runs in a **background thread** — be mindful of thread safety when accessing GTK widgets
(use `GLib.idle_add` for GTK operations).

### Example: Take a screenshot

```python
"""Capture a screenshot of the main window."""
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

## Using Both Flags

Both `--script` and `--uiscript` can be used together. The `--script` runs first (synchronously),
then the window loads, and then `--uiscript` runs:

```bash
rayforge --script early_setup.py \
    --uiscript automation.py \
    myproject.ryp
```

This is useful when you need to register plugins early and then drive the UI later.
