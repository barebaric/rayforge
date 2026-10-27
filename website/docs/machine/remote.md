---
description:
  "Rayforge always drives machines through one backend/frontend path: locally over a Unix domain
  socket, or across the network with rayforge --server and rayforge --client."
---

# Remote Execution (Backend and Frontend)

Rayforge has exactly one way of driving a machine: a **frontend** computes jobs and a **backend**
executes them through the configured drivers. When the machine is attached to this computer, both
run inside the Rayforge process and talk over a Unix domain socket (a loopback connection where Unix
sockets are unavailable). When the machine is attached to another computer, the backend runs there
as `rayforge --server` and the frontend connects over the network with `rayforge --client`.

Because both cases use the same protocol and the same code, remote machines behave exactly like
local ones — including jogging, homing, framing, probing, and reading device settings. The frontend
copies the backend's machines at startup, so nothing needs to be configured twice.

The backend owns the machine configuration and the running jobs: machine changes made on the
frontend are synced to the backend, which persists them, and you can close the frontend (e.g. your
laptop) while a job is running — reopen it at any time to see the progress of the running or
finished job. One job runs at a time per backend.

Both computers must run the same Rayforge version.

## Starting the Backend Standalone

On the computer connected to the laser cutter, run:

```bash
rayforge --server
```

This is equivalent to running the `rayforge-server` command. The backend loads your regular machine
configuration, connects the configured drivers, and listens on `127.0.0.1` (port `2517` by default).
Use an optional address argument to change it, for example:

```bash
rayforge --server 192.168.1.50
```

:::warning

The server has no authentication. Only bind it to trusted networks, and tunnel the connection (e.g.
via SSH) when you need remote access over untrusted networks.

:::

## Starting the Frontend

On any other computer on the same network, run:

```bash
rayforge --client 192.168.1.50
```

The address is `HOST` or `HOST:PORT`. The UI opens a small waiting window and retries until the
backend is reachable; the machines then appear like local machines and the main window opens.

Starting Rayforge without either flag runs the backend in-process and connects the frontend to it
automatically — no waiting window, no configuration, and no network involved.

## Technical Notes

- The frontend fetches the backend's machine profiles at startup, so its encoding always matches
  what the machine's driver expects.
- Machine configuration changes on the frontend (including machines created by the setup wizard or
  deleted in the settings) are synced to the backend, which owns the configuration directory.
- Device state (positions, connection status, work coordinate systems) streams from the backend to
  the frontend, which mirrors it locally.
- Jobs cross the wire as self-contained payloads; job progress is reported per command, and the
  backend keeps running jobs when the frontend disconnects.
