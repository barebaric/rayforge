# Plan: Remote driver support — daemon, control protocol, remote sessions

Status: refined (2026-11-12), incorporating the PR #643 review.
This expands phases 2 (remote sessions) and 3 (remote machines) of
driver-migration.md into an implementable design; that document
remains the master plan and its phase list points here. The debug
infrastructure these phases lean on shipped in raydriver 1.0.0
(analyzer, hardened listener, golden files); rayforge pins
raydriver ==1.0.0.

---

## 1. Goal

Run the machine from a small computer next to the hardware (e.g.
a Raspberry Pi in the workshop) while Rayforge runs anywhere on
the LAN — the use case behind PR #643. The session, and with it
the running job, lives on the device host; the client submits
the program and may disconnect mid-job, then reattach and see
exact progress. The client owns the UI and all machine
configuration; the device host owns the machine connection and
job execution.

Guarantees:

- A job submitted to the daemon runs to completion (or failure)
  even if the submitting client disconnects.
- Reattach resyncs progress at the exact op index.
- Exactly one client holds the control lease per session; others
  may watch.
- One client may hold sessions to any number of daemons at once
  (the "laser farm" case from the PR #643 discussion): each
  machine's config names its endpoint, daemons are independent,
  and no cross-daemon coordination is needed.
- Remote endpoints meet the same security policy as the analyzer:
  loopback unauthenticated by default, token + TLS mandatory for
  anything exposed.

## 2. Non-goals

- Not the "remote UI" direction (`rayforge serve` + web UI addon);
  independent, both wanted.
- Not a machine manager: machine configs, dialects and profiles
  stay client-side. The daemon hosts *sessions*, not machines.
- No video in the control protocol: device-host cameras are
  exposed as ordinary network streams that Rayforge's existing
  network-camera sources consume (§9).
- No WAN routing, NAT traversal or relaying; LAN (or tunnel) only.
- No new mandatory dependencies. Daemon and client are stdlib
  asyncio; mDNS announce is one optional extra (§6.6).

## 3. Architecture

```
Rayforge (client)                     Device host
┌──────────────────┐   TCP (+TLS)  ┌────────────────────────────┐
│ driver shell      │◄────────────►│ raydriver daemon           │
│  (same surface)   │  control     │  ├ listener (bind policy, │
│ RemoteSession     │  protocol    │  │ token auth, TLS)        │
│  (proxy: forwards │              │  ├ session registry       │
│  commands, replays│              │  │  GrblSession(…) etc.   │
│  events)          │              │  │  = Rust protocol stack │
└──────────────────┘              │  ├ lease + event ring     │
                                  │  └ event fan-out          │
                                  └────────────┬───────────────┘
                                               │ serial / tcp (Rust)
                                           ┌───┴────┐
                                           │ machine │
                                           └────────┘
```

A camera attached to the device host (USB camera plus a streamer,
or an IP camera) is reached directly by Rayforge's network-camera
sources; video never crosses the control link (§9).

Decisions, each traced to the master plan:

- **Remote session backend, not a byte pipe.** The daemon hosts
  real sessions (`GrblSession`, `GrblSimpleSession`,
  `SmoothieSession`, `MarlinSession`) over locally attached
  hardware; the client holds a proxy with the same method
  surface. Forwarding bytes ser2net-style would keep streaming
  client-side and lose the job on disconnect — the opposite of
  the goal.
- **Protocol logic stays where it already is.** The proxy
  forwards *commands* and receives *events*; it implements no
  firmware protocol. Session arguments and events are plain data
  at the crate's Python/Rust boundary, so forwarding is generic —
  one proxy serves all protocols. The one exception is
  `StateChanged.state`, a Rust `DeviceState` object: protocol.py
  defines an explicit dict codec (status, error, positions) and
  the proxy rebuilds the event so the shell's `from_session_state`
  consumes it unchanged.
- **Daemon in Python, sessions in Rust.** Job survival comes from
  the sessions being Rust-side inside the daemon process; the
  control listener is plain asyncio, so `python -m raydriver
  daemon` needs no runtime dependency beyond raydriver itself.
- **Client proxy in Python too.** One factory function replaces
  the direct constructor call in each shell — a one-line change
  per shell, an honest relaxation of the master plan's
  "shells stay unchanged" (§7.2). A Rust-native client is a
  possible later hardening; the protocol would not change.
- **Cameras stay out of the control plane.** The daemon neither
  proxies nor re-encodes video; it only advertises camera
  endpoints, and Rayforge's existing network-camera sources
  consume the streams (§9).

## 4. Protocol decision

Custom protocol: length-prefixed JSON frames over TCP, TLS via
stdlib `ssl`, constant-time token auth, mDNS `_raydriver._tcp`
for discovery. Rationale:

- The link is a small control plane (~20 message types) plus a
  modest event stream. There is no bulk data path — the job is
  submitted once and executed device-side — so a framework buys
  nothing.
- JSON on the wire keeps a single encoding story across the
  Python/Rust boundary, needs no schema toolchain, and keeps
  traffic debuggable: the analyzer decodes daemon captures with a
  plain JSON decoder.
- The two open standards worth adopting cost nothing: TLS
  (mandatory off-loopback) and mDNS/DNS-SD (discovery).

Rejected:

- **gRPC** — bidirectional streaming fits, but tonic/prost in
  Rust plus grpcio (C core) for any Python client, plus codegen,
  for a tiny message surface.
- **WebRTC** — its strengths (ICE, NAT traversal, hostile
  networks) are LAN-irrelevant; webrtc-rs is a huge dependency.
- **HTTP/WebSocket** — would carry the same custom message set
  inside an HTTP server stack. Latency would honestly be fine on
  this traffic profile; the layer simply buys no
  interoperability.
- **QUIC (quinn)** — attractive (TLS 1.3 inherent, connection
  migration); kept as a possible future transport swap. The
  envelope (§5) is transport-agnostic.

The envelope is documented as a stable, versioned wire format so
third parties can drive daemons without our library.

## 5. Wire protocol (v1)

### 5.1 Framing

- TCP or TCP+TLS. One frame = 4-byte big-endian payload length +
  one UTF-8 JSON object. Max frame 16 MiB (job submission is a
  single frame; chunked upload is a v2 concern).
- Idle keepalive: `ping`/`pong` frames every 2 s; three missed
  disconnects the peer, so a dead connection is detected within
  ~6 s. Both values are configurable (`keepalive_interval`,
  `keepalive_misses`) on the daemon and the client. The lease
  grace period (§6.3) starts at disconnect *detection*, so fast
  detection bounds the total time until a stale lease releases;
  a reconnect within the grace period keeps control.

### 5.2 Envelope

```
{"id": 7, "type": "session/command", "payload": {…}}    client → daemon
{"id": 7, "ok": true, "payload": {…}}                    response
{"id": 7, "ok": false,
 "error": {"code": "lease_taken", "message": "…"}}       error
{"type": "event", "payload": {…}}                        daemon → client
```

`id` is a client-chosen correlation integer; responses echo it,
events carry none. Unknown `type` ⇒ `unknown_type` error for
requests; clients must ignore unknown *event* kinds (forward
compatibility).

### 5.3 Connection setup

1. The daemon sends `welcome`:
   `{"protocol": 1, "raydriver": "1.0.0", "auth_required": true}`.
2. If `auth_required`, the client sends `auth {"token": "…"}`;
   a wrong token closes the connection (`auth_failed`).
   Comparison is constant-time. The client refuses to send a
   token when `welcome` did not arrive over TLS — no leakage to
   downgrade probes.
3. `protocol` mismatch ⇒ `bad_version` + close. v1 is
   exact-match until a v2 exists.

The listener reuses the analyzer's security primitives
(`raydriver.analyzer.listener`): `validate_bind_policy` applies
verbatim (loopback without auth by default; non-loopback or
wildcard binds require token AND TLS, no override; explicitly
configured auth is never dropped), as do
`make_server_ssl_context` / `make_client_ssl_context` and the
constant-time token comparison. The handshake itself is the JSON
welcome/auth flow above: the analyzer's line-based token
challenge and `connect_authenticated` are diagnostic-endpoint
specific and are not reused. The analyzer listener has no
keepalive; the daemon adds it (§5.1).

### 5.4 Message catalog

Session lifecycle:

- `device/list` → serial ports and TCP endpoints the daemon can
  reach (from the Rust binding, §6.4), plus the cameras
  advertised in its config (§9).
- `session/list` → open sessions with job summaries.
- `session/open {"protocol": "grbl"|"grbl_simple"|"smoothie"|
  "marlin", "config": {…}, "dialect": {…}, "role":
  "control"|"watch", "force": false}` → session info + lease
  granted, or `lease_taken` naming the current holder. `force:
  true` steals the lease and demotes the holder to watcher.
- `session/attach {"session_id", "after_seq": n?, "role": …}` →
  snapshot + replay (§5.5).
- `session/detach {"session_id"}` — the session keeps running.
- `session/close {"session_id"}` — disconnect the hardware and
  drop the session.

Command forwarding — a single frame covers the whole session
surface:

- `session/command {"session_id", "method": "connect" |
  "disconnect" | "run" | "run_raw" | "cancel" | "set_hold" |
  "execute_command" | "execute_interactive_command" | "home" |
  "move_to" | "jog" | "select_tool" | "set_power" |
  "set_focus_power" | "read_settings" | "write_setting" |
  "set_wcs_offset" | "read_wcs_offsets" | "read_parser_state" |
  "run_probe_cycle" | "detect_unit_system" | "update_dialect" |
  "clear_alarm", "args": {…}}` — the daemon returns the session's
  return value in `payload` (lists, dicts, floats, position
  tuples: all JSON). `run` returns a job handle. Integer-keyed
  maps such as `op_line_map` travel as pair lists — JSON keys are
  strings. `clear_alarm` exists on Smoothie only; the daemon
  rejects methods a session does not implement.
- `session/property {"session_id", "name"}` → a session property:
  `state`, `job_running`, `resource_uri`, `buffer_count`,
  `rx_buffer_size`, `pending_commands` (the last three are
  full-GRBL only). The shells read these between commands;
  `state` travels through the DeviceState codec (§3).

Events (daemon → client push):

```
{"type": "event", "payload": {"session": id, "seq": n,
 "event": {"kind": "state" | "connection_status" |
 "command_status" | "job_finished" | "probe_status" |
 "wcs_updated" | "config_changed" | "raw_rx" | "raw_tx" |
 "progress" | "job_error" | "lease", …}}}
```

`progress` bridges the Rust progress callback (op index); the
proxy re-creates `run`'s `progress_callback` locally from these
events — callables never cross the wire. `raw_rx`/`raw_tx` feed
the console verbose channel unchanged.

### 5.5 Event history and reattach

- Per session the daemon keeps a bounded ring of recent events
  (default 1000) with a monotonic `seq`.
- `session/attach {"after_seq": n}` replays everything after `n`
  still in the ring. On first attach, or when the ring no longer
  covers `n`, the daemon sends a `snapshot` instead:
  `{config, dialect, state, connection_status, job: {handle,
  total_ops, current_op, running, submitted_at} | null,
  next_seq}`. The proxy rebuilds exact progress from snapshot +
  replay — the resync guarantee.
- Slow clients: bounded per-client send queues. On overflow the
  daemon drops `raw_*` events first, then disconnects the client.

### 5.6 Job model

- `run` submits gcode + op map + estimates; streaming, progress
  and completion are daemon-side. One job at a time per session
  (matching the session APIs); submitting while running ⇒
  `job_running` error.
- Cancel and hold are regular commands acting on the daemon-side
  session immediately, regardless of any client's presence.

## 6. Daemon design

Module layout (all inside raydriver):

```
python/raydriver/remote/
    __init__.py
    protocol.py     # framing, envelope, catalog, errors
    daemon.py       # listener, registry, lease, ring, fan-out
    client.py       # RemoteSession proxy + create_session()
    mdns.py         # optional announce (extra: daemon)

CLI: raydriver daemon [--config daemon.json] — a subcommand
wired like cli/analyzer.py; python -m raydriver daemon works too.
```

1. **Listener**: the analyzer's bind policy, token comparison and
   TLS contexts with the JSON protocol handler (§5.3). The
   analyzer listener tests keep their meaning; the daemon gains
   protocol-level equivalents.
2. **Session registry**: session id → (Rust session, event ring,
   subscribers, lease). Sessions are constructed exactly as the
   shells construct them today (`GrblSession(config, dialect,
   event_callback)`); the daemon's callback wraps each event and
   fans it out.
3. **Lease**: one `control` role per session. Watchers receive
   events; their commands fail with `lease_held`. The lease
   releases on detach, close, or client disconnect after a grace
   period (default 5 s) so a reconnect during WiFi jitter keeps
   control. Auth is per-connection and has no grace period.
4. **Serial enumeration**: a new Rust binding
   `raydriver.core.list_serial_ports() -> list[dict]` (the
   serialport crate is already a dependency; no new libraries).
   Feeds `device/list` and the machine wizard.
5. **Config file** (JSON — parseable by the stdlib on Python
   3.10, unlike tomllib): bind host/port, token, TLS paths,
   keepalive interval and miss limit (§5.1), sessions to
   auto-open at boot (protocol + full session config),
   advertised cameras (§9), mDNS on/off. Default
   `~/.config/raydriver/daemon.json`. `--setup` writes a config
   with a generated token (`secrets`, 0600) and prints the
   openssl one-liner for a certificate (§10). A systemd unit
   example ships in the docs.
6. **mDNS announce** (optional extra `raydriver[daemon]` →
   python-zeroconf): `_raydriver._tcp` with TXT records
   (protocol versions, requires_auth, tls, camera endpoints
   (§9)). Discovery consumers stay in Rayforge, which already
   runs an mDNS channel for ESP3D.

## 7. Client design

1. **Proxy** (`remote/client.py`): `RemoteSession` implements the
   union of the four sessions' method surfaces (near-identical by
   design). Calls become `session/command` frames awaiting the
   correlated response; events from the reader task re-emerge
   through the constructor-supplied `event_callback` on the event
   loop — so shell signal wiring, including console verbose
   RX/TX, works unmodified. Each proxy owns its connection, so
   one client drives any number of daemons concurrently — one
   session per machine, no shared state between endpoints.
2. **Routing**: `create_session(protocol, config, dialect,
   event_callback)` returns the Rust session when the config has
   no remote keys, otherwise a `RemoteSession`. The five migrated
   shells (grbl serial, grbl telnet, grbl simple, smoothie,
   marlin) switch their single construction line to this factory
   (the master plan's "shells unchanged" becomes "shells change
   one line"; everything else — setup vars, events, capabilities —
   is untouched). Test rigs using `with_transport` bypass the
   factory by design.
3. **Remote config keys** (validated by the factory, forwarded in
   `SessionConfig`): `remote_host` + `remote_port` + `token` +
   `tls_ca`/`tls_hostname`, or a single `url`
   (`raydriver://host:port#token`). Plain `host`/`port` keep
   their local meaning (e.g. Smoothie telnet) — no ambiguity.
4. **Disconnect behavior**: reader-task failure marks the proxy
   sleeping and surfaces through `connection_status_changed`
   exactly as today; commands while detached raise the sessions'
   `ConnectionError`. `proxy.attach()` reattaches using the last
   received `seq`.

## 8. Rayforge integration (master-plan phase 3)

1. **Setup vars**: the five migrated shells gain a connection
   mode var (local/remote) plus `remote_host`, `remote_port` and
   `token`, gated by `visible_when`/`required_when`
   (rayforge/core/varset/var.py) in the existing flat VarSet —
   the framework has no nested groups and needs none. Values map
   straight into the session config. The same driver class works
   locally or remotely; capabilities need no mirroring — remote
   is a configuration choice.
2. **Probing over remote**: `probe()` goes through
   `execute_interactive_command`, which the proxy forwards, so
   the machine wizard works against hardware that is not locally
   attached — the gap PR #643 hit with client-side probing. The
   probers only need `setup` + `execute_interactive_command`
   (the `_GrblProbeDriver` protocol in grbl/grbl_probe.py), which
   the proxy satisfies.
3. **Discovery**: `_raydriver._tcp` is an *endpoint*, not a
   firmware. The mDNS channel
   (rayforge/machine/discovery/mdns_channel.py) browses it,
   speaks welcome + `device/list`, and emits one
   `DiscoveredDevice` per attached device: driver class from the
   session protocol, params carrying `remote_host`/`remote_port`;
   token and TLS are entered on the connection page. Reuses the
   existing mDNS machinery; no new dependencies.
   `test_builtin_mdns_declarations` grows the new service.
4. **Console**: verbose RX/TX already travels as raw events
   through the proxy — verified in tests, no code.
5. **Cameras**: the camera selection dialog
   (rayforge/ui_gtk/camera/selection_dialog.py) offers the
   daemon's advertised endpoints and prefills the network-camera
   URI (§9).
6. **Docs**: a website page (daemon install, token/cert
   generation, systemd, firewall note, camera streamer setup),
   wizard screenshots for the remote vars; the changelog is
   written at release preparation as usual.

## 9. Cameras on the device host

The PR #643 review asked for cameras attached to the device host
without reinventing video streaming. Decision: **video never
enters the control protocol**; the daemon neither proxies nor
re-encodes it. Rayforge already consumes network cameras
(`rayforge/camera/source/network.py`: `http_stream`,
`http_snapshot`, `rtsp`, with reconnect and cached-snapshot
fallback), so the daemon's only camera job is to *advertise
endpoints*.

- **IP cameras / existing streamers**: zero new code. The
  camera's own stream URL is added in Rayforge as a network
  camera, exactly as today.
- **USB cameras on the device host**: run a standard streamer
  (ustreamer or mjpg-streamer) next to the daemon; the website
  docs ship a setup and systemd example. The daemon advertises
  the resulting URL.
- **Advertisement**: `cameras: [{name, uri, type}]` in
  daemon.json (§6.5) surfaces in `device/list` and the mDNS TXT
  records; the wizard and the camera selection dialog prefill
  the URI (§8.5).
- **No model changes**: cameras stay per-machine
  (`Machine.cameras`), driver-independent; only the active
  machine's cameras stream (`CameraManager`). Hardware controls
  (white balance, contrast) already skip network sources.
- **Security**: video endpoints are separate from the TLS
  control channel; keep them LAN-only or firewalled. Streamer
  authentication is out of scope for the control protocol.

A built-in MJPEG bridge (an optional `raydriver[camera]` extra
reading V4L2, pattern: rayforge's
scripts/camera/camera_simulator.py) is deferred (§14) — v1
deliberately ships no video code.

## 10. Security model

- The analyzer bind policy, unchanged (ruida-pa issue #15 lesson):
  loopback may run bare; non-loopback or wildcard binds refuse to
  open without token AND TLS; explicitly configured
  authentication is never dropped.
- Token: generated by `--setup` or operator-supplied; stored 0600;
  compared constant-time.
- TLS: operator certificate via the documented openssl one-liner
  (no certificate-generation dependency); clients pin via
  `tls_ca`, and the wizard embeds the fingerprint in the machine
  config.
- The daemon never opens client-initiated outbound connections;
  mDNS is opt-in; errors surface as error frames, not tracebacks.

## 11. Testing

raydriver (each implementation step lands with its tests):

- **protocol**: framing round-trips, oversize frames, correlation,
  unknown-type tolerance, version and auth failures — codec tests
  over in-memory streams, sockets only where real.
- **daemon**: loopback integration against the firmware emulators
  (reuse the FAST_TIMINGS rigs): open/attach/command/event flows;
  **the job continues after the client socket drops**; reattach
  replays to the exact op index (asserted on progress events);
  lease contention including force-steal; watcher commands
  rejected; slow-client overflow disconnects; a silent peer is
  dropped within the configured keepalive window; two daemons
  driven by one client run independent jobs and leases; TLS +
  token round-trip using the committed test-certificate pattern
  from tests/certs/.
- **client**: a parity test drives one scenario twice — local
  session and proxy — and compares the event sequences.
- **golden (optional, cheap)**: a captured control-protocol
  conversation as a golden fixture; the analyzer already decodes
  JSON frames.

rayforge:

- Shell tests with a remote config against an in-test daemon +
  emulator: connect, run, progress, cancel, probe over remote,
  disconnect → sleeping → attach.
- Discovery tests against a fake `_raydriver._tcp` responder,
  including camera advertisement: `device/list` and discovery
  prefill a network-camera URI.
- The manual hardware checklist per phase: daemon on a Pi, job
  started, UI machine's network unplugged mid-job, reattach with
  exact progress, cancel from a second client, a camera on the
  device host visible in Rayforge while the job runs.

## 12. Implementation steps

Each raydriver step is a reviewable PR. Pin bumps ride the
rayforge PRs as usual (pixi.toml, pixi.lock, requirements.txt,
debian/requirements-bundle.txt — and refresh rayforge.egg-info
before `pixi lock`).

1. **raydriver**: `remote/protocol.py` — framing, envelope,
   catalog, errors (including the DeviceState codec); codec
   tests.
2. **raydriver**: Rust `list_serial_ports` binding (stubs via
   `make stubs`) + tests.
3. **raydriver**: `remote/daemon.py` — listener reuse, registry,
   lease, ring, fan-out, keepalive, camera advertisement;
   emulator-backed integration tests including disconnect
   survival.
4. **raydriver**: `remote/client.py` — proxy, factory, attach;
   parity tests.
5. **raydriver**: `raydriver daemon` subcommand (`--setup`,
   config file, systemd example), optional mDNS extra; release
   the next beta.
6. **rayforge**: pin bump; remote setup vars on the five shells,
   probe-over-remote, shell tests; wizard remote vars; camera
   prefill.
7. **rayforge**: `_raydriver._tcp` discovery channel (endpoint
   semantics) + wizard integration + tests.
8. **rayforge**: website docs (including the camera streamer
   guide) and screenshots; run the manual smoke checklist; tick
   the master-plan phases.

## 13. Risks

- **Reattach exactness** is the hard part (the master plan
  already flags it): mitigated by seq + ring + snapshot fallback,
  and a test that drops the socket mid-job at a random op.
- **Event storms** (raw lines at high poll rates): bounded
  queues, drop-raw-first policy; analyzer captures from real jobs
  will show real rates before v1 ships.
- **Lease vs. WiFi jitter**: grace period tuned by test;
  force-steal is explicit and evented.
- **Windows serial enumeration** differences in the new binding:
  covered by the release CI matrix from day one.
- **Scope creep**: viewer mode is events-only by design; remote
  UI, file transfer and daemon-side machine management stay out.

## 14. Open questions

- Should the daemon ever host *machines* (accepting profiles) or
  forever remain session-only? The master plan says session-only;
  confirmed for v1.
- Chunked job upload for very large files — deferred to v2; the
  16 MiB frame cap applies until then.
- A Rust-native client transport — only if a consumer needs the
  remote link without Python; the wire protocol would not change.
- A built-in MJPEG bridge for USB cameras on the device host
  (optional `raydriver[camera]` extra) — only if operating an
  external streamer proves too clumsy; v1 advertises endpoints
  only (§9).