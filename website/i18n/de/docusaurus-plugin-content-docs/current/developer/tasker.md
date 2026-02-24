# Tasker: Hintergrund-Task-Verwaltung

`tasker` ist ein Modul zum Ausführen lange laufender Aufgaben im Hintergrund einer GTK-Anwendung ohne Einfrieren der UI. Es bietet eine einfache, einheitliche API für sowohl E/A-gebundene (`asyncio`) als auch CPU-gebundene (`multiprocessing`) Arbeit.

## Kernkonzepte

1. **`task_mgr`**: Der globale Singleton-Proxy, den du zum Starten und Abbrechen aller Tasks verwendest
2. **`Task`**: Ein Objekt, das einen einzelnen Hintergrundjob repräsentiert. Du verwendest es zum Verfolgen des Status
3. **`ExecutionContext` (`context`)**: Ein Objekt, das als erstes Argument an deine Hintergrundfunktion übergeben wird. Dein Code verwendet es zum Melden von Fortschritt, Senden von Nachrichten und Prüfen auf Abbruch
4. **`TaskManagerProxy`**: Ein Thread-sicherer Proxy, der Aufrufe an den eigentlichen TaskManager weiterleitet, der im Haupt-Thread läuft

## Schnellstart

Alle Hintergrund-Tasks werden vom globalen `task_mgr` verwaltet.

### Ausführen einer E/A-gebundenen Task (z.B. Netzwerk, Dateizugriff)

Verwende `add_coroutine` für `async`-Funktionen. Diese sind leichtgewichtig und ideal für Tasks, die auf E/A warten.

```python
import asyncio
from rayforge.shared.tasker import task_mgr

# Deine Hintergrundfunktion MUSS `context` als erstes Argument akzeptieren.
async def my_io_task(context, url):
    context.set_message("Wird heruntergeladen...")
    # ... asynchronen Download durchführen ...
    await asyncio.sleep(2) # Arbeit simulieren
    context.set_progress(1.0)
    context.set_message("Download abgeschlossen!")

# Die Task aus deinem UI-Code starten (z.B. ein Button-Klick)
task_mgr.add_coroutine(my_io_task, "http://example.com", key="downloader")
```

### Ausführen einer CPU-gebundenen Task (z.B. schwere Berechnung)

Verwende `run_process` für reguläre Funktionen. Diese laufen in einem separaten Prozess, um den GIL zu vermeiden und die UI reaktionsfähig zu halten.

```python
import time
from rayforge.shared.tasker import task_mgr

# Eine reguläre Funktion, nicht async.
def my_cpu_task(context, iterations):
    context.set_total(iterations)
    context.set_message("Berechne...")
    for i in range(iterations):
        # ... schwere Berechnung durchführen ...
        time.sleep(0.1) # Arbeit simulieren
        context.set_progress(i + 1)
    return "Endergebnis"

# Die Task starten
task_mgr.run_process(my_cpu_task, 50, key="calculator")
```

### Ausführen einer Thread-gebundenen Task

Verwende `run_thread` für Tasks, die in einem Thread laufen sollen, aber keine volle Prozess-Isolation benötigen. Dies ist nützlich für Tasks, die Speicher teilen, aber dennoch die UI nicht blockieren sollten.

```python
import time
from rayforge.shared.tasker import task_mgr

# Eine reguläre Funktion, die in einem Thread laufen wird
def my_thread_task(context, duration):
    context.set_message("Arbeite im Thread...")
    time.sleep(duration) # Arbeit simulieren
    context.set_progress(1.0)
    return "Thread-Task abgeschlossen"

# Die Task in einem Thread starten
task_mgr.run_thread(my_thread_task, 2, key="thread_worker")
```

## Wesentliche Muster

### UI aktualisieren

Verbinde dich mit dem `tasks_updated`-Signal, um auf Änderungen zu reagieren. Der Handler wird sicher im Haupt-GTK-Thread aufgerufen.

```python
def setup_ui(progress_bar, status_label):
    # Dieser Handler aktualisiert die UI basierend auf dem Gesamtfortschritt
    def on_tasks_updated(sender, tasks, progress):
        progress_bar.set_fraction(progress)
        if tasks:
            status_label.set_text(tasks[-1].get_message() or "Arbeite...")
        else:
            status_label.set_text("Leerlauf")

    task_mgr.tasks_updated.connect(on_tasks_updated)

# Später in deiner UI...
# setup_ui(my_progress_bar, my_label)
```

### Abbruch

Gib deinen Tasks einen `key`, um sie später abzubrechen. Deine Hintergrundfunktion sollte regelmäßig `context.is_cancelled()` prüfen.

```python
# In deiner Hintergrundfunktion:
if context.is_cancelled():
    print("Task wurde abgebrochen, stoppe Arbeit.")
    return

# In deinem UI-Code:
task_mgr.cancel_task("calculator")
```

### Abschluss behandeln

Verwende den `when_done`-Callback, um das Ergebnis zu erhalten oder zu sehen, ob ein Fehler aufgetreten ist.

```python
def on_task_finished(task):
    if task.get_status() == 'completed':
        print(f"Task beendet mit Ergebnis: {task.result()}")
    elif task.get_status() == 'failed':
        print(f"Task fehlgeschlagen: {task._task_exception}")

task_mgr.run_process(my_cpu_task, 10, when_done=on_task_finished)
```

## API-Referenz

### `task_mgr` (Der Manager-Proxy)

- `add_coroutine(coro, *args, key=None, when_done=None)`: Eine asyncio-basierte Task hinzufügen
- `run_process(func, *args, key=None, when_done=None, when_event=None)`: Eine CPU-gebundene Task in einem separaten Prozess ausführen
- `run_thread(func, *args, key=None, when_done=None)`: Eine Task in einem Thread ausführen (teilt Speicher mit Hauptprozess)
- `cancel_task(key)`: Eine laufende Task anhand ihres Schlüssels abbrechen
- `tasks_updated` (Signal für UI-Updates): Wird ausgegeben, wenn sich der Task-Status ändert

### `context` (Innerhalb deiner Hintergrundfunktion)

- `set_progress(value)`: Aktuellen Fortschritt melden (z.B. `i + 1`)
- `set_total(total)`: Den Maximalwert für `set_progress` setzen
- `set_message("...")`: Den Statustext aktualisieren
- `is_cancelled()`: Prüfen, ob du anhalten sollst
- `sub_context(...)`: Einen Sub-Task für mehrstufige Operationen erstellen
- `send_event("name", data)`: (Nur Prozess) Benutzerdefinierte Daten zurück an die UI senden
- `flush()`: Alle ausstehenden Updates sofort an die UI senden

## Verwendung in Rayforge

Der Tasker wird in Rayforge durchgehend verwendet für:

- **Pipeline-Verarbeitung**: Ausführen der Dokumenten-Pipeline im Hintergrund
- **Dateioperationen**: Importieren und Exportieren von Dateien ohne Blockieren der UI
- **Gerätekommunikation**: Verwalten lange laufender Operationen mit Laserschneidern
- **Bildverarbeitung**: Durchführen CPU-intensiver Bildverfolgung und -verarbeitung

Wenn du mit dem Tasker in Rayforge arbeitest, stelle sicher, dass deine Hintergrundfunktionen Abbrüche ordnungsgemäß behandeln und aussagekräftige Fortschrittsupdates bereitstellen, um eine reaktionsfähige Benutzererfahrung zu erhalten.
