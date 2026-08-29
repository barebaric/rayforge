---
description: "Text box templates and custom template functions in the Rayforge parametric 2D sketcher."
---

# Text Templates

Text boxes support template expressions enclosed in curly braces. These are
resolved at solve time using the current parameter values, so the text updates
automatically when you change a dimension or input variable.

## Variable Substitution

Reference any sketch parameter or input variable by name:

- `{width}` -- the current value of the "width" parameter
- `{name}` -- the value of a string-type input parameter
- `{count:.0f}` -- formatted with a Python format specifier (no decimals)

## Math Expressions

You can use math functions inside templates:

- `{sqrt(area):.2f}` -- square root of "area", formatted to 2 decimals
- `{width * 2}` -- arithmetic expressions

The standard math functions (`sqrt`, `sin`, `cos`, `tan`, `pi`, etc.) are
available.

## Built-in Functions

| Function         | Return type | Description                                      |
| ---------------- | ----------- | ------------------------------------------------ |
| `{today()}`      | `date`      | Current UTC date (e.g., `2026-08-26`)            |
| `{date()}`       | `date`      | Alias for `today()`                              |
| `{now()}`        | `datetime`  | Current UTC date and time                        |
| `{time()}`       | `time`      | Current UTC time (e.g., `15:30:00.123456+00:00`) |
| `{timestamp()}`  | `float`     | Unix timestamp (seconds since epoch)             |
| `{uuid4()}`      | `str`       | 8-character hex string (e.g., `a1b2c3d4`)        |
| `{uuid8()}`      | `str`       | Alias for `uuid4()`                              |
| `{uuid()}`       | `str`       | Full UUID v4 string (36 chars)                   |

## Format Specs

Python format specs work on any expression result:

- `{width:.1f}` -- one decimal place
- `{timestamp():.0f}` -- no decimals on the timestamp
- `{today()}` -- default string representation

## Example Use Cases

- `Part #{uuid4()}` -- unique serial number on each solve
- `W={width:.1f} H={height:.1f}` -- live dimension labels
- `Date: {today()}` -- date-stamp each piece
- `{name} - {count:.0f}pcs` -- combine string and numeric parameters
- `{timestamp():.0f}` -- Unix timestamp for production logging

## Custom Template Functions

You can register your own functions to use inside text box templates.
This is useful for pulling serial numbers from a database, reading
external data, or generating custom labels.

### Writing the registration script

Create a Python file (e.g. `~/.config/rayforge/my_functions.py`):

```python
"""Register custom template functions for text box expressions."""
import sqlite3

from sketcher.core.template_functions import register_template_function

DB_PATH = "/home/you/production.db"


def next_serial() -> str:
    """Fetch and reserve the next serial number from the database."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "UPDATE counters SET value = value + 1 "
            "WHERE name = 'serial' RETURNING value"
        )
        row = cur.fetchone()
        conn.commit()
        return f"SN-{row[0]:06d}"
    finally:
        conn.close()


register_template_function("next_serial", next_serial)
```

Key points:

- Call `register_template_function(name, callable)` for each function.
- Your function can do anything Python can: open files, connect to
  databases, call APIs, etc.
- The function is called on **every render**, so it should be fast.
  Use caching if the underlying data doesn't change between renders.
- Functions are thread-safe if your callable is.

### Running Rayforge with the script

Use the `--script` flag to load your functions before the window opens:

```bash
rayforge --script ~/.config/rayforge/my_functions.py mydoc.ryp
```

This runs your script early during startup — before addons are loaded and
before the main window is created — so the function is available when the
sketch first solves.

### Using the function in a text box

In the sketcher, create a text box with:

```
{next_serial()}
```

Format specs work too:

```
{next_serial():>20}
```

### Registering functions programmatically

If you're writing an addon or a reusable library, you can call
`register_template_function` from any Python code that runs before the
sketch is solved:

```python
from sketcher.core.template_functions import register_template_function

register_template_function("part_number", lambda: f"P-{hash('x') % 10000:04d}")
```

### Built-in functions cannot be removed

The built-in functions (`today`, `now`, `uuid`, etc.) cannot be
unregistered. If you need to change their behavior, register a function
with a different name.
