# Plan: One device support library — migrate all drivers to raydriver

Status: in progress (updated 2026-11-12). Goal: make raydriver the single,
vendor-neutral device support library for Rayforge — and eventually
for any CNC/laser software — replacing every remaining in-app
protocol implementation: the GRBL simple, ESP3D network and Ruida
RPA drivers (retiring `ruida-pa` with it).

---

## 1. Vision

Today every CNC application reinvents device support: protocol code is
buried inside each app, tested against whatever hardware a maintainer
happens to own, and lost when a project goes dormant. Raydriver flips
that: an independent, open-source device layer that speaks the
protocols, owns the transports, and ships the tooling to develop
protocols _without_ hardware.

- **One generic API for all machines.** Sessions, transports, device
  state, dialects and events are protocol-neutral. GRBL, Ruida, and
  future firmware families are modules behind the same interface, so
  any CNC software can consume the library instead of growing its own
  stack.
- **Hardware-independent testability.** Every protocol ships with an
  emulator and a protocol analyzer TUI (capture live traffic, decode,
  edit and replay scripts, golden-file fixtures). Development, code
  review and CI run without a laser in the room; golden files captured
  from real controllers become regression fixtures.
- **Remote device access as a first-class capability.** A small
  daemon hosts sessions next to the hardware; the application — or
  any raydriver consumer — drives them over the network. Jobs run
  device-side and survive client disconnects. This is the "remote
  device" direction; it is independent of the "remote UI" direction
  (`rayforge serve` with the web UI addon), and both are wanted.
- **Prototype Fund Innovativ fit** (https://www.prototypefund.de/innovativ):
  - _Open source_: Rust core + Python bindings and TUI; small,
  - auditable, reusable beyond Rayforge.
  - _Datensparsamkeit_: fully local operation — no cloud, no
    accounts, no telemetry; the analyzer captures protocol data that
    never leaves the user's machine unless they choose to share it.
  - _Anwendungsfreundlichkeit_: makers and hardware owners
    participate by capturing golden files with the analyzer TUI,
    directly improving protocol coverage for everyone.
  - _Adaptierbarkeit_: new firmware or controller families are new
    modules behind a stable API; transports (serial, TCP, HTTP/WS,
    USB, UDP) are shared building blocks.

## 2. Current state

### raydriver (released as 1.0.0)

- `raydriver.core`: the protocol-neutral layer — byte transports
  (serial, telnet/tcp, mock), device value types, dialect
  command-template formatting, and the session primitives
  (`SessionEvents`, `SessionError`).
- `raydriver.grbl`: the full character-counting `GrblSession`
  (handshake, status polling, interactive queue, streaming with
  stall detection and deadlock recovery, settings, WCS, probing,
  homing, jogging) and `raydriver.grbl.simple.GrblSimpleSession`,
  the minimal ping-pong session.
  `raydriver.grbl.parser` mirrors the old `grbl_util` parsers;
  `raydriver.grbl.emulator` is a firmware-faithful GRBL 1.1h
  emulator.
- `raydriver.smoothie.SmoothieSession`: telnet/serial, Grbl-style
  status reports, Ctrl+C cancel, `!`/`~` hold, `version`/
  `config-get` probes. `raydriver.marlin.MarlinSession`: `start`
  handshake with phantom-port detection, `M114` polling, `M410`,
  `M149`; response parsers under `raydriver.marlin.parser`.
- Per-protocol firmware emulators for CI: grbl, smoothie, marlin.

Not yet in raydriver: remote sessions, HTTP/WebSocket transports
(ESP3D), USB/UDP transports and the Ruida protocol. The
analyzer/golden-file tooling shipped in 1.0.0.

### Rayforge drivers

| Driver                   | File                          | State                          |
| ------------------------ | ----------------------------- | ------------------------------ |
| `GrblSerialDriver`       | `grbl/grbl_serial.py`         | Rust shell                     |
| `GrblTelnetDriver`       | `grbl/grbl_telnet.py`         | Rust shell                     |
| `SmoothieDriver`         | `smoothie/smoothie_driver.py` | Rust shell                     |
| `MarlinSerialDriver`     | `marlin/marlin_serial.py`     | Rust shell                     |
| `GrblSerialSimpleDriver` | `grbl/grbl_serial_simple.py`  | Rust shell                     |
| `GrblNetworkDriver`      | `grbl/grbl_network.py`        | Python, 891 ln                 |
| `RuidaRPAAdapter`        | `ruidarpa/`                   | Python, 2770 ln + `ruida-pa`   |

- `grbl_util.py` is down to what the drivers still pending cutover
  consume plus the typed `$`-setting VarSet definitions used by the
  settings UI.
- Driver class names are persisted as strings in machine YAML
  (`driver:`), device profiles, the LightBurn importer mapping, the
  wizard icon map, and discovery keys
  (`"{driver_name}:{resource}"`). The shells keep every class name,
  setup var and config key, so existing configs load unchanged.

## 3. Decisions

- **Adopt old class names**: shells keep user-facing driver names,
  setup vars and config keys; no migration code for machine
  configs.
- **Port `GrblSerialSimpleDriver` instead of dropping it**: it
  becomes a thin shell over
  `raydriver.grbl.simple.GrblSimpleSession`, kept as a fallback for
  hardware where character-counting streaming misbehaves. It must
  stay an independent implementation, not a mode of the full
  session — sharing the session would drag along the very
  complexity the simple driver exists to avoid.
- **Remote device access lives in raydriver, at the session level**:
  a small daemon hosts sessions next to the hardware; rayforge keeps
  machine configs and submits encoded jobs. Job streaming is entirely
  Rust-side, so daemon-hosted jobs survive client disconnects. The
  daemon is a session host, not a machine manager — no machine
  configuration moves server-side.
- **PR #643** (remote client/server split, by fllmn): the use case
  and the split — client owns UI and computations, device host owns
  machine and job — are accepted. The proposed mechanism (a
  `rayforge/machine/remote/` package proxying the `Driver` API) is
  superseded by the raydriver remote-session work. Until that lands,
  `rayforge serve` with the web UI covers the headless-machine case.
  The two directions remain independent: remote UI (rayforge serve)
  and remote devices (raydriver).
- **Generalize the API incrementally**: the generic device layer is
  extracted from real consumers instead of speculative up-front
  abstraction. Remote sessions (Phase 2) are the next consumer that
  sizes it; Ruida (Phase 7) follows.

## 4. Phases

### Phase 1 — Port GrblSerialSimpleDriver (rayforge)

Status: landed (PR #658) — `grbl_serial_simple.py` is a thin shell
over `GrblSimpleSession`.

1. Rewrite `grbl_serial_simple.py` as a thin shell over
   `raydriver.grbl.simple.GrblSimpleSession` (same event wiring as
   the grbl serial shell). Keep the class name, setup vars
   (`port`/`baudrate`) and label.
2. Delete the Python implementation and its tests; shell tests
   against the grbl emulator, mirroring the grbl serial shell tests.
3. Changelog note: same fallback role, now Rust-backed.

### Phase 2 — Remote sessions (raydriver)

Detailed design: remote-driver.md (refined per the PR #643 review).

Remote device access, sized by the PR #643 use case: run the machine
from a small computer next to the hardware (e.g. a Raspberry Pi in
the garage) while the application runs elsewhere. The session — and
with it the running job — lives on the device host; a client submits
the program and may disconnect mid-job. Model this as a remote
*session backend*, not a `Transport` implementation: forwarding bytes
ser2net-style keeps streaming in the client and loses the job on
disconnect. Lift what the protocol needs into the generic core layer
rather than building it on Grbl-specific internals.

1. **Control protocol** over TCP (length-prefixed messages):
   hello/version handshake, token auth, device listing (serial ports
   and USB devices), session open with a `SessionConfig`, the full
   session command surface, `SessionEvent` and progress forwarding,
   job submission returning a job handle, and detach/reattach with
   retained progress and final outcome. One controller lease per
   session, so two clients cannot race over hold/cancel.
2. **Device daemon**: ships in the wheel (`python -m raydriver
   daemon`). Hosts sessions bound to local serial/USB/TCP hardware,
   survives client drops, fans events out to all connected clients,
   and supports an optional config file for auto-connecting machines
   at boot.
3. **Client side**: sessions gain remote config keys (`url`, or
   `remote_host` + `remote_port` + `token`) and route to the remote
   backend transparently. The rayforge driver shells change one
   construction line each (the session factory); everything else
   stays unchanged.
4. **Tests**: emulators served by a daemon on localhost (reuse the
   FAST_TIMINGS rigs); job continues after the client drops;
   reattach resyncs progress exactly; lease conflict; auth failure;
   version mismatch.
5. Release raydriver; bump the pin.

### Phase 3 — Remote machines (rayforge)

Detailed design: remote-driver.md (refined per the PR #643 review).

1. Optional remote endpoint setup vars on the migrated shells (host,
   port, token), mapped into the session config. The same driver
   class works locally or remotely, so capabilities need no
   mirroring; remote vs. local is purely a configuration choice.
2. Discovery: a channel that finds daemons on the network (e.g. mDNS
   `_raydriver._tcp`) and lists their attached devices; `probe()`
   runs over the remote session, so the machine wizard works against
   hardware that is not attached to the client — the gap PR #643
   hit with its client-side probing.
3. Console verbose RX/TX across remote sessions (`RawTx`/`RawRx`
   events already exist in the sessions).
4. Device-host cameras: the daemon advertises stream endpoints
   (config, `device/list`, mDNS TXT); Rayforge's existing
   network-camera sources consume them — no video in the control
   protocol (remote-driver.md §9).
5. Changelog and website docs, stating the coverage: every G-code
   driver is remote-capable once its cutover has landed; OctoPrint
   is inherently remote already.

### Phase 4 — Testability framework (raydriver)

The ruida-pa workflow, generalized:

- **Protocol analyzer TUI** (Python, ships with the raydriver
  package): attach to any transport or run as a man-in-the-middle;
  capture streams, decode to a readable listing, edit and replay
  scripts to characterize controller behavior; export captures.
- **Golden files**: every protocol defines its fixture set
  (captured raw frames + decoded listing + rendered output); captured
  from real hardware once, replayed forever in tests.
- **Per-protocol emulators**: the existing grbl/smoothie/marlin
  emulators become the template; add an ESP3D emulator and a Ruida
  emulator. CI runs every protocol session against its emulator; the
  analyzer TUI doubles as the hardware characterization tool for
  contributors.

### Phase 5 — Network / ESP3D support (raydriver)

ESP3D/FluidNC protocol (as implemented by the old Python driver):

- Status polls (`?`) are triggered via HTTP GET
  `/command?commandText=...`; the HTTP response body is ignored — all
  events and responses arrive over a WebSocket (default port 81) as
  `<...>` status reports, `ok`, `error:` and info lines.
- Interactive commands: HTTP GET command trigger, then collect
  response lines on the WebSocket until `ok` / `error:`. The request
  must be registered _before_ the HTTP GET (response race).
- Jobs: multipart POST to `/upload?path=/` (the "Longer" protocol
  variant adds `path`/`size` form fields), then GET
  `/command?commandText=%5BESP220%5D/<filename>` to execute. No flow
  control; per-op completion is reported upfront.
- Cancel: HTTP GET with URL-encoded `\x18` (soft reset), then
  best-effort dialect safety-off commands.
- Connect sequence: `[ESP420]` hw info, `[ESP800]` fw info, `[ESP400]`
  EEPROM, then HTTP poll + WebSocket.
- Discovery stays in rayforge: mDNS `_esp3d._tcp` +
  `fingerprint_grbl_http` (aiohttp).

Work items:

1. **`Esp3dTransport`** implementing the `Transport` trait:
   `write()` → HTTP GET command endpoint (reqwest); background
   WebSocket task (tokio-tungstenite) pushing messages into the rx
   channel; background status-poll loop honoring
   `poll_status_while_running` / `status_poll_interval`;
   `resource_uri()` = `tcp://host:http_port`.
2. **Session network job mode**: `run()` branch for non-streaming
   transports — multipart upload + `[ESP220]` execute, `command_done`
   per op upfront (parity with the old driver), `job_finished`
   handling; `cancel()` = `\x18` command + safety shutdown.
3. **Config**: extend `SessionConfig` and the Python binding with
   `http_port`, `ws_port`, `protocol` (ESP3D/Longer) alongside
   `host`.
4. **Tests**: the ESP3D emulator over real localhost HTTP + WebSocket,
   covering handshake, interactive commands, `$$`/`$#`/`$G`,
   upload/execute, cancel, reconnect.
5. Release raydriver; bump the pin.

### Phase 6 — Network cutover (rayforge)

1. Rewrite `grbl_network.py`: `GrblNetworkDriver` becomes a thin shell
   over the session (same event wiring as the grbl serial shell).
   Keep the class name, `DISCOVERY` (mDNS + fingerprint),
   `reports_granular_progress = False`, and the exact setup-var keys
   `host` / `port` (HTTP, default 80) / `ws_port` (81) / `protocol`
   so existing configs load unchanged; map them to the session
   config.
2. Settings, WCS, parser state, probing and unit detection come from
   the session methods, same as the serial shell.
3. Delete the Python implementation; shell tests via
   `with_transport` + registry/discovery tests (`test_channels.py`,
   `test_session.py`).

### Phase 7 — Ruida support (raydriver)

Port the Ruida protocol from `ruidarpa` + `ruida-pa`:

1. **Transports**: USB (libusb or serial-backed) and UDP, with the
   automatic USB-preferred / UDP-fallback behavior the current driver
   implements.
2. **Protocol codec**: magic number handshake (default `0x88`),
   machine-code encoding (port of `rpa_encoder.py`: gluescript /
   rpascript semantics, vector power floor and image power bias
   compensation), response decode, and the `.rd` / `.cglu` / `.rds`
   file formats (needed by the analyzer and golden files).
3. **Session**: job streaming (ACK handshake over USB, UDP
   resend/recovery), state reporting, keepalive/ping handling.
4. **Emulator + analyzer support**: a Ruida emulator driven by golden
   files; analyzer TUI decodes Ruida captures; migrate the golden
   fixtures from `ruidarpa/golden/` into raydriver's test suite.
5. Release raydriver; bump the pin.

Hardware characterization (RDC6442S-class controllers) is the long
pole; golden files already captured by the community reduce the risk.

### Phase 8 — Ruida cutover (rayforge)

1. Thin Ruida shell(s) in `rayforge/machine/driver/ruida/` replacing
   `ruidarpa`, keeping the user-facing driver names and setup vars
   (UDP host, USB device, magic number) so configs carry over.
2. Delete `rayforge/machine/driver/ruidarpa/` and the
   `ruida-pa` dependency from `pixi.toml`.
3. Update the Ruida docs (`docs/agents` entry, addon README).

### Phase 9 — Final cleanup (rayforge)

1. Shrink `grbl_util.py` to what remains Rayforge-side:
   `get_grbl_setting_varsets` + the typed `$`-setting Var definitions,
   `apply_setting_to_varset`, `grbl_opt_re`; delete everything the
   Rust parser covers (parsers, error/alarm tables, ESP3D URL
   constants, `CommandRequest`, ...).
2. Delete now-unreferenced transports: `transport/grbl.py`
   (`GrblSerialTransport` + Python flow control), `transport/http.py`,
   `transport/websocket.py`, and the Python serial/telnet transport
   implementations freed by the smoothie, marlin and simple driver
   cutovers, and their exports (verify with grep — `aiohttp` stays
   for `grbl_fingerprint` and OctoPrint).
3. Sweep stale references; run `pixi run lint` and the machine/ui test
   suites; update website docs and changelog. The Prototype Fund
   application text derives from the vision section above.

## 5. Risk notes

- Phase 5's HTTP→WebSocket response pairing is the trickiest part;
  preserve the old driver's request-registration-before-send order.
- Phase 2's detach/reattach is the hard part of the remote protocol:
  progress must resume at the exact op index or the UI misreports
  position, and the controller lease must keep two clients from
  racing hold/cancel.
- A LAN-reachable laser needs token auth at minimum; document the
  RTT budget (the character-counting window divided by RTT must
  exceed the baud rate; WiFi jitter counts), stable USB device naming
  on daemon hosts, and a version handshake between daemon and client.
- Phase 7 hardware characterization is the schedule risk; golden
  files from real controllers (existing `ruidarpa/golden/` set plus
  new captures) are the mitigation.
- Hardware smoke tests (serial, grblHAL telnet, ESP3D device, Ruida
  controller) cannot be automated; keep a manual checklist per
  phase.
- Net effect: roughly 4500 lines of Python protocol code still to
  delete from rayforge, replaced by thin shells over the library.