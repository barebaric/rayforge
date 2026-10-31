---
description: "Prepare and run jobs from a browser with the Rayforge web UI."
---

# Web UI

Rayforge can serve its workpiece editor in the browser. The web UI offers the same document workflow
as the desktop application — importing, arranging, layer and operation management, toolpath preview,
and machine control — without installing a desktop environment on the machine that runs it.

---

## Starting the Server

Run:

```
rayforge serve
```

or, from a source checkout:

```
python -m rayforge serve
```

Then open `http://127.0.0.1:8080/` in the browser.

| Option        | Description                                                               |
| ------------- | ------------------------------------------------------------------------- |
| `--host ADDR` | Bind address. Defaults to `127.0.0.1` (local access only).                |
| `--port N`    | Port to listen on. Defaults to `8080`.                                    |
| `--token T`   | Require a shared secret; every request must present it as a bearer token. |
| `--config D`  | Configuration directory (same layout as the desktop application).         |
| `--loglevel`  | Logging verbosity. Default: `WARNING`.                                    |

The server reads the same configuration directory as the desktop application, so machines, laser
heads, and settings configured on the desktop are available in the web UI.

---

## Security Notes

By default the server listens on the loopback interface only, which keeps it private to your
computer.

- Only use `--host 0.0.0.0` (exposing the server to your network) on networks you trust, and combine
  it with `--token`.
- When `--token` is set, REST calls must send `Authorization: Bearer <token>` and browser sessions
  are prompted for the token once.
- The machine control endpoints can move your machine. Do not expose the server to the public
  internet.

---

## Browser Requirements

Any current version of Firefox, Chrome, Edge, or Safari with WebGL 2 support works. The 3D bed view
requires hardware or software WebGL; on machines without WebGL the page shows an error banner
instead of the canvas.

---

## Feature Overview

- **Document editing**: import SVG, DXF, and image files; arrange workpieces with move, resize, and
  rotate handles; full undo/redo with a history browser.
- **Layers**: add, rename, recolor, reorder, activate, and hide layers; each layer keeps its own
  workflow of operations.
- **Operations**: add and configure steps such as Contour, Engrave, or Offset Fill through
  schema-driven forms.
- **Toolpath preview**: colored preview of the generated toolpaths with a scrub and playback
  transport.
- **Job control**: start, pause, and cancel jobs with live progress and an ETA estimate.
- **Machine control**: connect to the machine, jog, home, frame the job, switch work coordinate
  systems (G54–G59), set the work origin, and send raw G-code from a console.
- **Projects**: save and re-open `.ryp` project files and export the generated G-code.

The browser keeps a live connection to the server; several tabs share the same document state, and
the view resynchronizes automatically after a connection loss or a server restart.

---

## Current Limitations

- Machines cannot be created or edited in the web UI; configure them in the desktop application or
  edit the configuration files.
- Camera capture, the rotary cylinder view, and workpiece vector editing are desktop-only.
