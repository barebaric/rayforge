# Web Service Architecture

## Goal

A rayforge addon that exposes a REST API for headless automation —
importing files, running layout strategies, executing the pipeline,
and downloading G-code or `.ryp` project files.
Runs in Docker without a display server.

## Architecture

```
HTTP request
    │
    ▼
FastAPI route ──▶ editor.file.import_file(...)
                  editor.layout.execute_layout_async(...)
                  editor.step.set_step_param(...)
                  editor.file.save_project(path)
                  editor.file.export_gcode(path)
    │
    ▼
DocEditor (existing, unchanged API surface)
    │
    ▼
Doc / Pipeline / TaskManager / Worker subprocesses
```

Two layers: HTTP <-> DocEditor commands. No intermediate facade.

The addon calls `bootstrap_headless()` once at startup to create a
`RayforgeContext`, `TaskManager`, and `DocEditor`, then injects the
editor into route handlers via FastAPI dependency injection.

---

## Phase 1: Addon `service` Entry Point (DONE)

The addon system has three entry point types:

| Type       | Main process (GUI) | Main process (headless) | Worker subprocess |
| ---------- | ------------------ | ----------------------- | ----------------- |
| `worker`   | loaded             | loaded                  | loaded            |
| `frontend` | loaded             | skipped                 | skipped           |
| `service`  | loaded             | loaded                  | skipped           |

---

## Phase 2: Addon Skeleton

Create the addon package with bootstrap and a minimal server. The addon should be in `rayforge/private_addons`.

```
rayforge-addon-webservice/
├── rayforge-addon.yaml
├── webservice/
│   ├── __init__.py
│   ├── bootstrap.py         # bootstrap_headless()
│   ├── server.py            # FastAPI app factory + lifespan
│   └── routers/
│       ├── projects.py
│       ├── files.py
│       ├── layout.py
│       ├── pipeline.py
│       └── machines.py
```

Manifest (uses the `service` entry point from Phase 1):

```yaml
name: webservice
display_name: Web Service
description: REST API for headless automation
api_version: 12
provides:
  service: webservice.server
```

### Bootstrap

```python
# webservice/bootstrap.py

def bootstrap_headless(config_dir=None):
    """
    Create a headless rayforge environment.
    Returns (context, task_manager, editor).
    """
    from rayforge.context import get_context
    from rayforge.worker_init import initialize_worker

    context = get_context()
    context._headless = True
    context.initialize_lite_context(config_dir or _default_config_dir())

    _ = context.addon_mgr

    shared_state = context.artifact_store.get_shared_state()

    task_manager = TaskManager(
        main_thread_scheduler=lambda f, *a, **kw: f(*a, **kw),
        worker_initializer=initialize_worker,
        shared_state=shared_state,
    )

    editor = DocEditor(task_manager, context)
    return context, task_manager, editor
```

The headless scheduler runs callbacks synchronously on the calling
thread. For async operations the `TaskManager` already runs its own
event loop in a daemon thread.

### API Endpoints

#### Projects

| Method | Path                         | Description                    |
| ------ | ---------------------------- | ------------------------------ |
| POST   | `/api/v1/projects`           | Create empty project           |
| GET    | `/api/v1/projects/{id}`      | Download `.ryp` file           |
| POST   | `/api/v1/projects/{id}/load` | Load existing `.ryp` from disk |

#### Files

| Method | Path                          | Description                                  |
| ------ | ----------------------------- | -------------------------------------------- |
| POST   | `/api/v1/projects/{id}/files` | Upload file (multipart), import as workpiece |

#### Layout

| Method | Path                           | Description           |
| ------ | ------------------------------ | --------------------- |
| PUT    | `/api/v1/projects/{id}/layout` | Run a layout strategy |

#### Pipeline

| Method | Path                                 | Description               |
| ------ | ------------------------------------ | ------------------------- |
| POST   | `/api/v1/projects/{id}/pipeline/run` | Execute pipeline          |
| GET    | `/api/v1/projects/{id}/gcode`        | Download generated G-code |

#### Machines

| Method | Path                      | Description                     |
| ------ | ------------------------- | ------------------------------- |
| GET    | `/api/v1/machines`        | List available machine profiles |
| PUT    | `/api/v1/machines/{name}` | Set active machine              |

#### WebSocket

| Path         | Description                 |
| ------------ | --------------------------- |
| `/api/v1/ws` | Stream task progress events |

### Example Automation Flow

```
POST /api/v1/projects                         -> {id: "abc"}
POST /api/v1/projects/abc/files  +svg multipart -> imports workpiece
POST /api/v1/projects/abc/files  +svg multipart -> imports workpiece
PUT  /api/v1/projects/abc/layout  {strategy: "nesting"}  -> auto-nests
POST /api/v1/projects/abc/pipeline/run        -> runs pipeline
GET  /api/v1/projects/abc/gcode               -> downloads G-code
GET  /api/v1/projects/abc                     -> downloads .ryp
```

### Session / State Management

```python
# webservice/server.py
sessions: Dict[str, DocEditor] = {}

def create_session() -> str:
    pid = uuid4().hex
    editor = DocEditor(shared_task_manager, context)
    sessions[pid] = editor
    return pid
```

All routes resolve the editor via `sessions[pid]`. Each session owns
one `Doc` but shares the `TaskManager` and `Context`.

### Long-Running Operations

Layout and pipeline execution are async. The API returns `202 Accepted`
immediately and streams progress over the WebSocket:

```
PUT /api/v1/projects/abc/layout  -> 202 {task_id: "xyz"}
WS  /api/v1/ws                   -> {"task_id": "xyz", "progress": 0.5}
WS  /api/v1/ws                   -> {"task_id": "xyz", "progress": 1.0, "status": "done"}
```

### Implementation Plan

1. Create addon directory structure and manifest.
2. Implement `bootstrap.py` — confirm it boots without GTK in a test.
3. Implement `server.py` with FastAPI app factory and lifespan.
4. Implement the `projects` router (create session, save/download `.ryp`).
5. Test: round-trip create -> download `.ryp`.

---

## Phase 3: File Import Endpoint

1. Implement the `files` router — accept multipart upload, call the
   async import API (`DocEditor.import_file_from_path`).
2. Test: upload SVG -> verify workpiece exists in doc.

---

## Phase 4: Layout & Pipeline Endpoints

1. Implement the `layout` router — resolve strategy from
   `layout_registry`, call `execute_layout_async`.
2. Implement the `pipeline` router — trigger pipeline run, expose
   G-code download.
3. Implement `machines` router.
4. Implement WebSocket progress streaming.
5. Integration test: upload -> nest -> run pipeline -> get G-code.

---

## Running

```bash
# From rayforge (addon installed)
rayforge serve --port 8080
```

---

## Docker

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    libcairo2 libvips42 libpoppler-glib8 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["rayforge", "serve", "--port", "8080"]
```

No Xvfb, no display server.
