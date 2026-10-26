---
description:
  "Run Rayforge as a headless job server on the machine connected to your laser, and compute jobs on
  a separate computer that you can close while a job is running."
---

# Remote Execution (Server and Client)

Rayforge can be split across two computers:

- The **server** is a headless Rayforge process that runs on the computer physically connected to
  the laser cutter. It owns the machine's driver and executes jobs with the correct timing.
- The **client** is the regular Rayforge application. It performs the demanding computations
  (geometry, pipeline, G-code encoding) and hands the finished job to the server.

Because the server owns the job once it has been accepted, you can close the client (e.g. your
laptop) while a job is running. Reconnect at any time to see the current progress of the running or
finished job. One job runs at a time; multiple jobs can be queued by submitting them one after
another.

Both computers must run the same Rayforge version.

## Starting the Server

On the computer connected to the laser cutter, run:

```bash
rayforge-server
```

The server loads your regular machine configuration, connects the configured drivers, and listens on
`127.0.0.1` (port `2517` by default). Use the `--host` and `--port` options to change the address,
for example:

```bash
rayforge-server --host 192.168.1.50
```

:::warning

The server has no authentication. Only bind it to trusted networks, and tunnel the connection (e.g.
via SSH) when you need remote access over untrusted networks.

:::

## Importing the Server's Machine on the Client

On the client, open **Settings → Machines** and press **Import from Server**:

1. Enter the hostname or IP address and port of the server, then press **Connect**.
2. Pick one of the machines the server hosts (a single machine is selected automatically), then
   press **Import Machine**.

The machine's complete profile — kinematics, heads, G-code dialect, speeds — is copied to the
client, so the client encodes jobs exactly the way the machine expects. The imported machine shows
up like any other machine and can be selected, framed, and started normally.

## Limitations

Job execution, framing, stopping, and feed hold (pause/resume) work remotely. Direct interactive
control of the device — jogging, homing, probing, and live power control — is not available through
the remote driver yet and must be done on the server.
