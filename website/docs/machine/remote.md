---
description:
  "Run Rayforge as a headless backend on the computer connected to your laser, and drive it from a
  frontend anywhere on the network with rayforge --client."
---

# Remote Execution (Backend and Frontend)

Rayforge can run as a **backend** on the computer that is physically connected to the laser cutter,
and as a **frontend** on your laptop or workstation. The frontend does all the demanding computation
— geometry, the pipeline, G-code encoding — and the backend executes the finished jobs with correct
timing through its configured drivers.

The split is transparent: in frontend mode the server's machines appear and behave exactly like
locally connected machines, including jogging, homing, framing, probing, and reading device
settings. No configuration is needed on the frontend: it copies the machines from the backend at
startup.

Because the backend owns the job once it has been accepted, you can close the frontend (e.g. your
laptop) while a job is running. Reopen it at any time to see the progress of the running or finished
job. One job runs at a time per backend.

Both computers must run the same Rayforge version.

## Starting the Backend

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

Starting Rayforge without either flag is completely unchanged: it uses the machine configuration of
that computer and connects directly, exactly as before.

## Technical Notes

- The frontend fetches the backend's machine profiles at startup, so its encoding always matches
  what the machine's driver expects.
- Device state (positions, connection status, work coordinate systems) streams from the backend to
  the frontend, which mirrors it locally.
- Jobs cross the network as self-contained payloads; job progress is reported per command, and the
  backend keeps running jobs when the frontend disconnects.
